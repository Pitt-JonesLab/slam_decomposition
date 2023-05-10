import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    ConsolidateBlocks,
    CountOps,
    Optimize1qGates,
    Unroll3qOrMore,
)
from tqdm import tqdm
from weylchamber import c1c2c3

from slam.basis import MixedOrderBasisCircuitTemplate
from slam.utils.gates.bare_candidates import get_group_name
from slam.utils.gates.custom_gates import ConversionGainGate, CustomCostGate, RiSwapGate
from slam.utils.gates.duraton_scaling import atomic_cost_scaling
from slam.utils.gates.family_extend import recursive_sibling_check
from slam.utils.gates.winner_selection import pick_winner
from slam.utils.polytopes.polytope_wrap import monodromy_range_from_target
from slam.utils.transpiler_pass.weyl_decompose import (
    RootiSwapWeylDecomposition as decomposer,
)


class fooAnalysis(AnalysisPass):
    """print duration of the circuit (iswap = 1 unit)"""

    def __init__(self, duration_1q):
        super().__init__()
        self.duration_1q = duration_1q

    def run(self, dag):
        # XXX this probably didn't cause any issues, but problem in logic. The DAG longest path is not necessarily the duration longest path.

        """Start with overall longest path."""
        freq = {}  # tracking frequency of gates
        d = 0  # tracking critical path duration
        for gate in dag.longest_path():
            if isinstance(gate, DAGOpNode):
                if gate.op.duration is not None:
                    d += gate.op.duration
                elif gate.op.name in ["u", "u1", "u2", "u3"]:
                    d += self.duration_1q
                elif gate.op.name in ["cx"]:
                    d += 1
                # longest path frequency tracking
                freq[gate.op.name] = freq.get(gate.op.name, 0) + 1

        self.property_set["duration"] = d
        self.property_set["gate_counts"] = dag.count_ops()
        self.property_set["longest_path_counts"] = freq

        # XXX for now, just assume all measurements take place together at the end of the circuit

        # """Next, look at longest path on each wire"""
        # # to calculate duration over each wire, start by putting measurements at the end of each wire
        # # then, use remove_nonancestors_of followed by longest_path, to find the critical path for each wire
        # tempc = dag_to_circuit(dag)
        # tempc.measure_all()
        # msmtdag = circuit_to_dag(tempc)
        # # now, we have a dag with measurements at the end of each wire
        # msmtnodes = msmtdag.named_nodes('measure')
        # for i in range(len(msmtnodes)):
        #     wire_d = []
        #     d = 0
        #     # make a working copy of the dag
        #     msmtdag = circuit_to_dag(tempc)
        #     node = msmtdag.named_nodes('measure')[i] # have to remake such that the node is in the copy of the dag, bad code :(
        #     msmtdag.remove_nonancestors_of(node)
        #     # now all paths lead to the measurement
        #     # find the longest path
        #     path = msmtdag.longest_path()
        #     for gate in path:
        #         if isinstance(gate, DAGOpNode):
        #             if gate.op.duration is not None:
        #                 d += gate.op.duration
        #             elif gate.op.name in ['u', 'u1', 'u2', 'u3']:
        #                 d += self.duration_1q
        #             elif gate.op.name in ['cx']:
        #                 d += 1
        #     wire_d.append(d)

        # self.property_set['wire_duration'] = wire_d

        logging.info("\nTranspilation Results:")
        logging.info(f"Gate Counts: {dag.count_ops()}")
        logging.info(f"Longest Path Gate Counts: {freq}")
        logging.info(f"Duration: {d}")
        # logging.info(f"Wire Duration: {wire_d}")
        return 1  # success


class SpeedGateSubstitute(TransformationPass):
    def __init__(
        self,
        speed_method,
        duration_1q,
        strategy,
        basic_metric,
        coupling_map,
        lambda_weight=0.47,
        family_extension=False,
    ):
        super().__init__()

        self.speed_method = speed_method
        self.duration_1q = duration_1q
        self.strategy = strategy
        self.basic_metric = basic_metric
        self.coupling_map = coupling_map
        self.lambda_weight = lambda_weight
        self.family_extension = family_extension

        # makes sure the data exists first # cost scaling is deprecated
        # cost_scaling(speed_method=speed_method, duration_1q=duration_1q)
        self.group_name = get_group_name(speed_method, duration_1q)

        # NOTE force requires so that only 2Q ops exist in dag
        # collect 2Q blocks
        self.requires.extend(
            [
                Unroll3qOrMore(),
                Collect2qBlocks(),
                ConsolidateBlocks(force_consolidate=True),
            ]
        )

    def run(self, dag: DAGCircuit):
        """Run the pass on `dag`."""

        if (
            self.strategy == "basic_overall"
            or self.strategy == "lambda_weight"
            or self.strategy == "basic_smush"
            or self.strategy == "lambda_smush"
        ):
            """Here, we define a single metric to pick a winner gate to be used
            for all decompositions.

            Metrics pick most efficient for either SWAP, CNOT, or Haar
            OR using lambda * d[cnot] + (1-lambda) * d[swap] as winner metric
            """
            metric = (
                self.basic_metric
                if ("basic" in self.strategy)
                else (-1, self.lambda_weight)
            )
            smush_bool = True if ("smush" in self.strategy) else False
            winner_gate, scaled_winner_gate = pick_winner(
                self.group_name,
                metric=metric,
                plot=False,
                smush_bool=smush_bool,
                family_extension=self.family_extension,
            )
            # that way we only have to compute a single coverage set
            # NOTE winner_gate goes to constructor so hits the saved polytope coverage set
            template = MixedOrderBasisCircuitTemplate(
                base_gates=[winner_gate], smush_bool=smush_bool
            )

            logging.info("Found winner, begin substitution")

            # second, make substitutions
            for node in dag.two_qubit_ops():
                target = Operator(node.op).data

                if self.family_extension:
                    ret = recursive_sibling_check(
                        template,
                        target,
                        cost_1q=self.duration_1q,
                        basis_factor=scaled_winner_gate.duration,
                        use_smush=smush_bool,
                    )
                    ret[0]  # should already be built
                    # XXX template isn't using scaled gates will break the fooAnalysis
                    # print("here")
                    # idea is to substitute a dummy gate that contains the duration attribute
                    # note the actual unitary doesn't matter since we're just using the duration
                    # problem is that the 1Q gates are not used so don't get simplified away
                    # need to check if subtemplate has 1Q gates then subtract duration_1q from the dummy cost
                    # this is very hacky, don't like
                    # XXX lets just assume it always has 1Q gates
                    CustomCostGate(
                        str="dummy1q",
                        unitary=XGate(),
                        duration=self.duration_1q,
                        num_qubits=1,
                    )
                    dummy_gate = CustomCostGate(
                        unitary=target,
                        duration=ret[1] - (2 * self.duration_1q),
                        str="dummy",
                    )
                    sub_qc = QuantumCircuit(2)
                    # sub_qc.append(dummy_single, [0])
                    # sub_qc.append(dummy_single, [1])
                    sub_qc.append(dummy_gate, [0, 1])
                    # sub_qc.append(dummy_single, [0])
                    # sub_qc.append(dummy_single, [1])
                    sub_dag = circuit_to_dag(sub_qc)
                    dag.substitute_node_with_dag(node, sub_dag)
                else:
                    reps = monodromy_range_from_target(template, target_u=target)[0]

                    # NOTE, when we build, actually use the scaled_winner_gate which has the proper duration attiriubte
                    template.build(reps, scaled_winner_gate)

                    # we should set all the U3 gates to be real valued - doesn't matter for sake of counting duration
                    sub_qc = template.assign_Xk(template.parameter_guess())
                    sub_dag = circuit_to_dag(sub_qc)
                    dag.substitute_node_with_dag(node, sub_dag)

        elif self.strategy == "weighted_overall":
            """Here, we are counting gates that appear in the circuit in order
            to define a winner metric."""
            # first, need frqeuncy of each gate
            # NOTE this feels unoptimized, because we are consolidating 1Q gates, so more misses (?)
            target_ops = [g.op for g in dag.two_qubit_ops()]
            winner_gate, scaled_winner_gate = pick_winner(
                self.group_name,
                metric=-1,
                target_ops=target_ops,
                plot=False,
                family_extension=self.family_extension,
            )  # XXX unoptimized !
            logging.info("Found winner, begin substitution")

            template = MixedOrderBasisCircuitTemplate(base_gates=[winner_gate])

            # second, make substitutions
            for node in dag.two_qubit_ops():
                target = Operator(node.op).data

                reps = monodromy_range_from_target(template, target_u=target)[0]

                template.build(reps, scaled_winner_gate)
                # we should set all the U3 gates to be real valued - doesn't matter for sake of counting duration
                sub_qc = template.assign_Xk(template.parameter_guess())
                sub_dag = circuit_to_dag(sub_qc)
                dag.substitute_node_with_dag(node, sub_dag)

        elif self.strategy == "weighted_pairwise":
            """Here, we count gates that appear between each pair of qubits in
            the circuit, define a winner for each pair."""
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
                target_ops = [
                    g.op
                    for g in dag.two_qubit_ops()
                    if set(edge) == set((g.qargs[0].index, g.qargs[1].index))
                ]
                if len(target_ops) == 0:
                    continue

                winner_gate, scaled_winner_gate = pick_winner(
                    self.group_name,
                    metric=-1,
                    target_ops=target_ops,
                    tqdm_bool=False,
                    plot=False,
                    family_extension=self.family_extension,
                )

                logging.info(f"Found winner for {edge} edge, begin substitution")

                template = MixedOrderBasisCircuitTemplate(base_gates=[winner_gate])

                # second, make substitutions for the 2Q gates between the two qubits
                for node in dag.two_qubit_ops():
                    if set(edge) == set((node.qargs[0].index, node.qargs[1].index)):
                        target = Operator(node.op).data

                        reps = monodromy_range_from_target(template, target_u=target)[0]

                        template.build(reps, scaled_winner_gate)
                        # we should set all the U3 gates to be real valued - doesn't matter for sake of counting duration
                        sub_qc = template.assign_Xk(template.parameter_guess())
                        sub_dag = circuit_to_dag(sub_qc)
                        dag.substitute_node_with_dag(node, sub_dag)

            # turn logging back on
            logger.setLevel(logging.INFO)

        else:
            raise ValueError("Strategy not recognized")

        logging.warning(
            "1Q gates are not being set to accurate values, just placeholders for fast counting"
        )
        return dag


class OptimizedSqiswapSub(TransformationPass):
    """
    Unoptimized code :(
    Replace CX-family gates with iSwap-fam identity, and SWAP gates with iSwap-fam identity
    This TransformationPass takes advantage of 2 main optimized:
    1. The CX-family gates reduce from a. 2 Sqiswaps build all CX gates to b. iswap^n builds cx^n gates
    For a linear SLF, this changes CX from .1+.5+.1+.5+.1 = 1.3 to .1+1+.1 = 1.2
    For a linear SLF, this changes sqCX from .1+.5+.1+.5+.1 = 1.3 to .1+.5+.1 = 0.7
    ...
    2. The SWAP gates reduce from a. 3 sqiswaps to b. 1 iswap + 1 sqiswap
    For a linear SLF, this changes SWAP from .1+.5+.1+.5+.1+.5+.1 = 1.9 to .1 + 1 + .1 + .5 + .1 = 1.8
    We also believe it is possible to remove the last internal 1Q gate, which would give .1 + 1 +.5 + .1 = 1.7

    3. It could be possible to remove the external gates if unitary equivalence >> local equivalence, but this is not implemented
    This could have very large savings for SWAP gate, but for CX we expect in some cases the external gates to be simplified away
    """

    def __init__(self, duration_1q=0, speed_method="linear"):
        super().__init__()
        self.duration_1q = duration_1q
        self.speed_method = speed_method

    def run(self, dag):
        """Run the OptimizedSqiswapSub pass on `dag`."""
        # first, we need to get a duration scaled iswap gate
        iswap = ConversionGainGate(0, 0, np.pi / 2, 0, t_el=1)
        sqiswap = ConversionGainGate(0, 0, np.pi / 2, 0, t_el=0.5)
        scaled_iswap, _ = atomic_cost_scaling(
            params=iswap.params,
            scores=np.array([0]),
            speed_method=self.speed_method,
            duration_1q=self.duration_1q,
        )
        scaled_sqiswap, _ = atomic_cost_scaling(
            params=sqiswap.params,
            scores=np.array([0]),
            speed_method=self.speed_method,
            duration_1q=self.duration_1q,
        )

        # second load pd coverage of sqiswap
        # we will use this for QV, and any other edge cases (non CX/SWAP gates)
        edge_iswap_template = MixedOrderBasisCircuitTemplate(
            base_gates=[iswap], use_smush_polytope=True
        )
        template = MixedOrderBasisCircuitTemplate(
            base_gates=[sqiswap], use_smush_polytope=True
        )

        # third, we iterate over the 2Q gates and replace them with the scaled iswap gate
        for node in dag.two_qubit_ops():
            # convert node to weyl coordinate
            try:
                target = Operator(node.op).data
                target_coord = c1c2c3(target)
            except:
                # XXX I'm not confident this is always the right thing to do
                logging.warning(
                    "ValueError in c1c2c3, setting target to real part, issue likely due to imaginary numerical error on SWAP gate"
                )
                target = target.real

            sub_qc = QuantumCircuit(2)
            # add random 1Q unitaries to the sub circuit with np.random.random()
            sub_qc.u(np.random.random(), np.random.random(), np.random.random(), 0)
            sub_qc.u(np.random.random(), np.random.random(), np.random.random(), 1)

            # if target coord is a controlled unitary, use derivived 1-1 equivalency to iswap family (up to local equiv)
            if target_coord[1] == 0 and target_coord[2] == 0:
                # with parallel drive, CX==iSwap, sqCX==sqiswap, etc
                scale_factor = (
                    target_coord[0] / 0.5
                )  # divide by .5 to get x-coord of CX
                sub_iswap = ConversionGainGate(
                    *scaled_iswap.params[:-1],
                    t_el=scaled_iswap.params[-1] * scale_factor,
                )
                sub_iswap.normalize_duration(1)
                sub_iswap.duration = scaled_iswap.duration * scale_factor
                sub_qc.append(sub_iswap, [0, 1])

            # if target coord is a swap, use derived equivalency to iswap family iswap_pd + 1Q + sqiswap (up to local equiv)
            # NOTE the best we have found is
            elif target_coord == (0.5, 0.5, 0.5):
                # with parallel drive, SWAP is 1 parallel-driven iSwap followed by a sqiswap
                sub_qc.append(scaled_iswap, [0, 1])

                # XXX we think these can go away if have a perfect decomposition, but hard to find with our approx methods
                PERFECTED_DECOMP_SWAP = 0
                # add random 1Q gates
                if not PERFECTED_DECOMP_SWAP:
                    sub_qc.u(
                        np.random.random(), np.random.random(), np.random.random(), 0
                    )
                    sub_qc.u(
                        np.random.random(), np.random.random(), np.random.random(), 1
                    )

                # add sqiswap
                scale_factor = 1 / 2
                sub_iswap = ConversionGainGate(
                    *scaled_iswap.params[:-1],
                    t_el=scaled_iswap.params[-1] * scale_factor,
                )
                sub_iswap.normalize_duration(1)
                sub_iswap.duration = scaled_iswap.duration * scale_factor
                sub_qc.append(sub_iswap, [0, 1])

            elif target_coord == (0.5, 0.5, 0):  # iSwap may show up from CX+SWAP
                # iswap is just iswap :)
                sub_qc.append(scaled_iswap, [0, 1])
            else:
                # this is edge case
                # for example, in QFT there are some [SWAP+CX] => pSwap
                # pass this to standard monodromy template
                # the idea is that we can still may get an improvement from the \
                # parallel-drive extended coverage polytope

                # XXX missing iSwap parallel driven, family-extension. This code is messy which is why it's not here yet
                # hardcode the cases
                # if its in extended iSwap in 1, do 2 sqiswaps no middle 1Q gate
                reps = monodromy_range_from_target(
                    edge_iswap_template, target_u=target
                )[0]

                if reps == 1:
                    sub_qc.append(scaled_iswap, [0, 1])

                else:  # if its in extended iSwap in 2 more just use the regular extended sqiswap
                    reps = monodromy_range_from_target(template, target_u=target)[0]
                    template.build(reps, scaled_sqiswap)
                    sub_qc = template.assign_Xk(
                        template.parameter_guess()
                    )  # radom real values
                    # this template is going to have an extra set of external 1Q gates,
                    # because cases nested in this for loop have pre-, post- fixed gates
                    # its not a problem to have doubles, as long as Optimized1QGates removes them
                    # we need to make sure that when we simplify that they get cancelled out

            # add random 1Q unitaries to the sub circuit with np.random.random()
            sub_qc.u(np.random.random(), np.random.random(), np.random.random(), 0)
            sub_qc.u(np.random.random(), np.random.random(), np.random.random(), 1)

            # make the substitution
            sub_dag = circuit_to_dag(sub_qc)
            dag.substitute_node_with_dag(node, sub_dag)

        return dag


# optimized sqiswap pass manager (with dummy substitution)
class pass_manager_optimized_sqiswap(PassManager):
    """Note, the sqiswap basis gate is a bit of a misnomer, because the basis
    gate we would calibrate is actually a smaller fraction of an iswap
    (whatever smallest fraction preserves fidelity) For example, if we
    calibrate 1/16 of an iswap, we would build sqiswap and iswap gates by
    repeating the gate with no interleaving 1Q gates 8,16 times."""

    def __init__(self, duration_1q=0, speed_method="linear"):
        passes = []
        passes.extend([CountOps()])
        # collapse 2Q gates
        passes.extend(
            [
                Unroll3qOrMore(),
                Collect2qBlocks(),
                ConsolidateBlocks(force_consolidate=True),
            ]
        )
        # every CX-family gate is replaced using iSwap-fam identity
        # every SWAP gate is repalced using iSwap-fam identity
        passes.extend(
            [OptimizedSqiswapSub(duration_1q=duration_1q, speed_method=speed_method)]
        )
        # collapse 1Q gates
        passes.extend([Optimize1qGates()])
        passes.extend([CountOps(), fooAnalysis(duration_1q)])
        super().__init__(passes)
        logging.warning(
            "1Q gates are not being set to accurate values, just placeholders for fast counting"
        )


# speed-limit aware manager
class pass_manager_slam(PassManager):
    def __init__(
        self,
        strategy="basic_overall",
        speed_method="linear",
        duration_1q=0,
        basic_metric=0,
        family_extension=0,
        coupling_map=None,
    ):
        passes = []
        passes.extend([CountOps()])
        passes.extend(
            [
                SpeedGateSubstitute(
                    strategy=strategy,
                    speed_method=speed_method,
                    duration_1q=duration_1q,
                    basic_metric=basic_metric,
                    coupling_map=coupling_map,
                    family_extension=family_extension,
                )
            ]
        )
        # combine 1Q gates
        passes.extend([Optimize1qGates()])
        passes.extend([CountOps(), fooAnalysis(duration_1q)])
        super().__init__(passes)


class pass_manager_basic(PassManager):
    def __init__(self, gate="sqiswap", duration_1q=0):
        passes = []
        passes.extend([CountOps()])
        # collect 2Q blocks
        # FIXME, it is probably faster to not consolidate, and have some smarter means of duplicate target substitution
        passes.extend(
            [
                Unroll3qOrMore(),
                Collect2qBlocks(),
                ConsolidateBlocks(force_consolidate=True),
            ]
        )
        if gate == "sqiswap":
            passes.extend([decomposer(basis_gate=RiSwapGate(1 / 2))])
        elif gate == "cx":
            passes.extend([decomposer(basis_gate=CXGate())])
        # combine 1Q gates
        passes.extend([Optimize1qGates()])
        passes.extend([CountOps(), fooAnalysis(duration_1q)])
        super().__init__(passes)
