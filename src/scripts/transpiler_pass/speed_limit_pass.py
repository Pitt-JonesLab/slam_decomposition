import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import sys
sys.path.append("../../../")

from weyl_decompose import RootiSwapWeylDecomposition as decomposer
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks, Unroll3qOrMore, Optimize1qGates
from src.utils.custom_gates import RiSwapGate
from qiskit.circuit.library import CXGate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Operator
from src.basis import MixedOrderBasisCircuitTemplate
from src.utils.polytope_wrap import monodromy_range_from_target
from src.scripts.gate_exploration.bgatev2script import get_group_name, cost_scaling, pick_winner
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import CountOps
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.transpiler import PassManager
from tqdm import tqdm

class fooAnalysis(AnalysisPass):
    """print duration of the circuit (iswap = 1 unit)"""
    def __init__(self, duration_1q):
        super().__init__()
        self.duration_1q = duration_1q

    def run(self, dag):
        d = 0 #tracking critical path duration
        freq = {} #tracking frequency of gates
        for gate in dag.longest_path():
            if isinstance(gate, DAGOpNode):
                d += gate.op.duration if gate.op.duration is not None else 0
                if gate.op.name in ['u', 'u1', 'u2', 'u3']:
                    d += self.duration_1q
                if gate.op.name in ['cx']:
                    d += 1
                # longest path frequency tracking
                if gate.op.name in freq:
                    freq[gate.op.name] += 1
                else:
                    freq[gate.op.name] = 1

        self.property_set['duration'] = d
        self.property_set['longest_path_counts'] = freq

        logging.info("\nTranspilation Results:")
        logging.info(f"Gate Counts: {dag.count_ops()}")
        logging.info(f"Longest Path Gate Counts: {freq}")
        logging.info(f"Duration: {d}")

class SpeedGateSubstitute(TransformationPass):
    def __init__(self, speed_method, duration_1q, strategy, basic_metric, coupling_map):
        super().__init__()
        
        self.speed_method = speed_method
        self.duration_1q = duration_1q
        self.strategy = strategy
        self.basic_metric = basic_metric
        self.coupling_map = coupling_map

        # makes sure the data exists first
        cost_scaling(speed_method=speed_method, duration_1q=duration_1q)
        self.group_name = get_group_name(speed_method, duration_1q)

        # NOTE force requires so that only 2Q ops exist in dag
        # collect 2Q blocks
        self.requires.extend([Unroll3qOrMore(), Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])

    def run(self, dag: DAGCircuit):
        """Run the pass on `dag`."""

        if self.strategy == 'basic_overall':
            """Here, we define a single metric to pick a winner gate to be used for all decompositions
            Metrics pick most efficient for either SWAP, CNOT, or Haar"""
            winner_gate, scaled_winner_gate = pick_winner(self.group_name, metric=self.basic_metric, plot=False)
            #that way we only have to compute a single coverage set
            #NOTE winner_gate goes to constructor so hits the saved polytope coverage set
            template = MixedOrderBasisCircuitTemplate(base_gates=[winner_gate])
            
            logging.info("Found winner, begin substitution")

            #second, make substitutions
            for node in dag.two_qubit_ops():
                target = Operator(node.op).data

                reps = monodromy_range_from_target(template, target_u =target)[0] 
                
                #NOTE, when we build, actually use the scaled_winner_gate which has the proper duration attiriubte
                template.build(reps, scaled_winner_gate)

                #we should set all the U3 gates to be real valued - doesn't matter for sake of counting duration
                sub_qc = template.assign_Xk(template.parameter_guess())
                sub_dag = circuit_to_dag(sub_qc)
                dag.substitute_node_with_dag(node, sub_dag)

        elif self.strategy == 'weighted_overall':
            """Here, we are counting gates that appear in the circuit in order to define a winner metric"""
            #first, need frqeuncy of each gate
            # NOTE this feels unoptimized, because we are consolidating 1Q gates, so more misses (?)
            target_ops = [g.op for g in dag.two_qubit_ops()]
            winner_gate, scaled_winner_gate  = pick_winner(self.group_name, metric=-1, target_ops=target_ops, plot=False) #XXX unoptimized !
            logging.info("Found winner, begin substitution")

            template = MixedOrderBasisCircuitTemplate(base_gates=[winner_gate])

            #second, make substitutions
            for node in dag.two_qubit_ops():
                target = Operator(node.op).data

                reps = monodromy_range_from_target(template, target_u =target)[0] 
                
                template.build(reps, scaled_winner_gate)
                #we should set all the U3 gates to be real valued - doesn't matter for sake of counting duration
                sub_qc = template.assign_Xk(template.parameter_guess())
                sub_dag = circuit_to_dag(sub_qc)
                dag.substitute_node_with_dag(node, sub_dag)

        elif self.strategy == 'weighted_pairwise':
            """Here, we count gates that appear between each pair of qubits in the circuit, define a winner for each pair"""
            # get edges from coupling map

            # turn off logging, too verbose with many winners
            logging.info("Iterating over edges, finding winners")
            logger.setLevel(logging.ERROR)

            edges = self.coupling_map.get_edges()
            # in order to remove duplicates, we need to sort the edges
            # only keep edges if the first qubit is smaller than the second
            edges = [e for e in edges if e[0] < e[1]]
            for edge in tqdm(edges):
                # target_ops = [g.op for g in dag.two_qubit_ops() if (g.qargs[0].index, g.qargs[1].index) == edge]
                # target ops are the 2Q gates that are between the two qubits but the order of the qubits is not important
                target_ops = [g.op for g in dag.two_qubit_ops() if set(edge) == set((g.qargs[0].index, g.qargs[1].index))]
                if len(target_ops) == 0:
                    continue
                
                winner_gate, scaled_winner_gate  = pick_winner(self.group_name, metric=-1, target_ops=target_ops, tqdm_bool=False, plot=False)

                logging.info(f"Found winner for {edge} edge, begin substitution")

                template = MixedOrderBasisCircuitTemplate(base_gates=[winner_gate])

                #second, make substitutions for the 2Q gates between the two qubits
                for node in dag.two_qubit_ops():
                    if set(edge) == set((node.qargs[0].index, node.qargs[1].index)):
                        target = Operator(node.op).data

                        reps = monodromy_range_from_target(template, target_u =target)[0] 
                        
                        template.build(reps, scaled_winner_gate)
                        #we should set all the U3 gates to be real valued - doesn't matter for sake of counting duration
                        sub_qc = template.assign_Xk(template.parameter_guess())
                        sub_dag = circuit_to_dag(sub_qc)
                        dag.substitute_node_with_dag(node, sub_dag)

            # turn logging back on
            logger.setLevel(logging.INFO)

        else:
            raise ValueError("Strategy not recognized")

        logging.warning("1Q gates are not being set to accurate values, just placeholders for fast counting")
        return dag 

#speed-limit aware manager
class pass_manager_slam(PassManager):
    def __init__(self, strategy='basic_overall', speed_method='linear', duration_1q=0, basic_metric=0, coupling_map=None):
        passes = []
        passes.extend([SpeedGateSubstitute(strategy=strategy, speed_method=speed_method, duration_1q=duration_1q, basic_metric=basic_metric, coupling_map=coupling_map)])
        #combine 1Q gates
        passes.extend([Optimize1qGates()])
        passes.extend([CountOps(), fooAnalysis(duration_1q)])
        super().__init__(passes)

class pass_manager_basic(PassManager):
    def __init__(self, gate='sqiswap', duration_1q=0):
        passes = []
        # collect 2Q blocks
        passes.extend([Unroll3qOrMore(), Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
        if gate == 'sqiswap':
            passes.extend([decomposer(basis_gate=RiSwapGate(1/2))])
        elif gate == 'cx':
            passes.extend([decomposer(basis_gate=CXGate())])
        #combine 1Q gates
        passes.extend([Optimize1qGates()])
        passes.extend([CountOps(), fooAnalysis(duration_1q)])
        super().__init__(passes)