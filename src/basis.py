import logging
from abc import ABC
from dataclasses import dataclass
from inspect import signature
from itertools import cycle
from random import uniform

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info import Operator
from scipy.spatial import KDTree
from weylchamber import c1c2c3

from .hamiltonian import Hamiltonian
from .utils.custom_gates import *
from .utils.data_utils import filename_encode, pickle_load, pickle_save
from .utils.polytope_wrap import (gate_set_to_coverage,
                                  monodromy_range_from_target)

"""
Defines the variational object passed to the optimizer
#TODO: this should extend Qiskit's NLocal class
"""

class VariationalTemplate(ABC):
    def __init__(self, preseed:bool, use_polytopes:bool):
        if self.filename is None:
            raise NotImplementedError
        self.data_dict = pickle_load(self.filename)

        self._construct_tree()

        #messy bit of logic here
        # I want spanning rule to refer to either a function using polytopes given a targer
        # or a constant set range
        # only valid if range is 1 otherwise nuop fails when on boundary
        self.use_polytopes = use_polytopes
        if not self.use_polytopes and self.spanning_range is None:
            raise NotImplementedError
        
        #NOTE: preseeding without polytopes can work if checking that nuop matches spanning range
        #rather than implementing this I'll just force using polytopes so spanning range always matches
        self.preseeded = preseed and self.use_polytopes #(self.use_polytopes or len(self.spanning_range))
        self.seed = None
    
    def eval(self, Xk):
        #evaluate on vector of parameters
        raise NotImplementedError
    
    def parameter_guess(self, temperature=0):
        #return a random vector of parameters
        if self.preseeded and self.seed is not None:
            #add a dash of randomization here, ie +- 5% on each value
            return [el*uniform(1-.05*temperature, 1+ .05*temperature) for el in self.seed]
        return None

    def assign_seed(self, Xk):
        self.seed = Xk
    
    def clear_all_save_data(self):
        self.data_dict = {}
        self._construct_tree()
        self.save_data()

    def save_data(self):
        pickle_save(self.filename, self.data_dict)
    
    def get_spanning_range(self, target_u):
        if not self.use_polytopes:
            return self.spanning_range
        else:
            #call monodromy polytope helper
            return monodromy_range_from_target(self, target_u)

    def _construct_tree(self):
        if len(self.data_dict) > 0:
                #for preseeding, a good data structure to find the closest already known coordinate
                self.coordinate_tree = KDTree(list(self.data_dict.keys()))
        else:
            #no data yet
            self.coordinate_tree = None

    #XXX below will fail for 3Q+
    def target_invariant(self, target_U):
        if not (4,4) == target_U.shape:
            return (-1,-1,-1,-1)
            raise NotImplementedError
        return c1c2c3(target_U)

    def undo_invariant_transform(self, target_U):
        # at this point state of self is locally equivalent to target, 
        # what transformation is needed to move state of template to target basis?
        raise NotImplementedError
        #we need this in a transpiler toolflow, but not for now

@dataclass
class DataDictEntry():
    success_label: int
    loss_result: float
    Xk: list
    cycles: int

class HamiltonianTemplate(VariationalTemplate):
    def __init__(self, h:Hamiltonian):
        self.filename = filename_encode(repr(h))
        self.h = h
        self.spanning_range = range(1)
        super().__init__(preseed=False, use_polytopes=False)
    
    def eval(self, Xk):
        return self.h.construct_U(*Xk).full()
    
    def parameter_guess(self,t=0):
        parent = super().parameter_guess(t)
        if parent is not None:
            return parent
        p_len =  len(signature(self.h.construct_U).parameters) #getting number of parameters from Hamiltonian function definition
        return np.random.random(p_len)

class CircuitTemplate(VariationalTemplate):
    def __init__(self, n_qubits=2, base_gates=[RiSwapGate(1/2)], edge_params=[[(0, 1)]], no_exterior_1q=False, use_polytopes=False, maximum_span_guess=5, preseed=False):
        """Initalizes a qiskit.quantumCircuit object with unbound 1Q gate parameters"""
        hash = str(n_qubits)+ str(base_gates)+ str(edge_params)+ str(no_exterior_1q)
        self.filename = filename_encode(hash)
        self.n_qubits = n_qubits
        self.no_exterior_1q = no_exterior_1q

        self.gate_2q_base = cycle(base_gates)
        #each gate gets its on cycler
        self.gate_2q_edges = cycle([cycle(edge_params_el) for edge_params_el in edge_params])
        self.gen_1q_params = self._param_iter()

        #compliant with basisv2 optimizer changes
        self.using_bounds = False
        self.bounds_list = None
        self.using_constraints = False
        self.constraint_func = None

        #define a range to see how many times we should extend the circuit while in optimization search
        self.spanning_range = None
        if not use_polytopes:
            self.spanning_range = range(1,maximum_span_guess+1)
            self.coverage = None #only precomputed in mixedbasis class
            
        super().__init__(preseed=preseed, use_polytopes=use_polytopes)

        self._reset()

        #deprecated feature
        self.trotter=False
               
    def eval(self, Xk):
        """returns an Operator after binding parameter array to template"""
        return Operator(self.assign_Xk(Xk)).data

    def parameter_guess(self, t=0):
        """returns a np array of random values for each parameter"""
        parent = super().parameter_guess(t)
        if parent is not None:
            return parent
        return np.random.random(len(self.circuit.parameters)) * 2 * np.pi

    def assign_Xk(self, Xk):
        return self.circuit.assign_parameters(
            {parameter: i for parameter, i in zip(self.circuit.parameters, Xk)}
        )

    def _reset(self):
        """Return template to a 0 cycle"""
        self.cycles = 0
        self.circuit = QuantumCircuit(self.n_qubits)
        self.gen_1q_params = self._param_iter() #reset p labels

    def build(self, n_repetitions):
        self._reset()

        if n_repetitions <= 0:
            raise ValueError()

        if self.trotter:
            pass
            # n_repetitions = int(1 / next(self.gate_2q_params))
        for i in range(n_repetitions):
            self._build_cycle(initial=(i == 0), final=(i == n_repetitions - 1))

    def _param_iter(self):
        index = 0
        while True:
            # Check if Parameter already created, then return reference to that variable
            def _filter_param(param):
                return param.name == f"P{index}"

            res = list(filter(_filter_param, self.circuit.parameters))
            if len(res) == 0:
                yield Parameter(f"P{index}")
            else:
                yield res[0]
            index += 1
            if self.trotter:
                index %= 3 * self.n_qubits

    def _build_cycle(self, initial=False, final=False):
        """Extends template by next nonlocal gate"""
        if initial and not self.no_exterior_1q:
            # before build by extend, add first pair of 1Qs
            for qubit in range(self.n_qubits):
                self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
                #self.circuit.ry(*[next(self.gen_1q_params) for _ in range(1)], qubit)

        gate = next(self.gate_2q_base)
        edge = next(next(self.gate_2q_edges)) #call cycle twice to increment gate then edge
        self.circuit.append(gate, edge)
        if not (final and self.no_exterior_1q):
            for qubit in edge:
                #self.circuit.ry(*[next(self.gen_1q_params) for _ in range(1)], qubit)
                self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
        self.cycles += 1


"""this might be deprecated, if I want to use gate costs, should be factored in with circuit polytopes already
for now, instead just use the mixedbasis instead"""

# class CustomCostCircuitTemplate(CircuitTemplate):
#     
#     #assigns a cost value to each repeated call to build()
#     def __init__(self, base_gates=[CustomCostGate]) -> None:
#         logging.warning("deprecated, use mixedorderbasis with cost assigned to cirucit polytopes")
#         for gate in base_gates:
#             if not isinstance(gate, CustomCostGate):
#                 raise ValueError("Gates must have a defined cost")
#         self.cost = {0:0}
#         super().__init__(n_qubits=2, base_gates=base_gates, edge_params=[(0,1)], no_exterior_1q=False,use_polytopes=True, preseed=True)

#     def _build_cycle(self, initial=False, final=False):
#         """Extends template by next nonlocal gate
#         add modification which saves cost to dict"""
#         if initial and not self.no_exterior_1q:
#             # before build by extend, add first pair of 1Qs
#             for qubit in range(self.n_qubits):
#                 self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
#         edge = next(self.gate_2q_edges)
#         gate = next(self.gate_2q_base)
#         self.circuit.append(gate, edge)
#         if not (final and self.no_exterior_1q):
#             for qubit in edge:
#                 self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
#         self.cycles += 1
#         self.cost[self.cycles] = self.cost[self.cycles-1] + gate.cost
        
#     def unit_cost(self, cycles):
#         if not cycles in self.cost.keys():
#             self.build(cycles)
#         return self.cost[cycles]


"""if hetereogenous basis, build is always appending in a set order
we really want to be able to pick and choose freely
when we find monodromy spanning range, we actually need to be checking more combinations of gates
find the minimum cost polytope which contains target
now when we create basis it should precompute the coverage polytopes and order them based on cost"""

class MixedOrderBasisCircuitTemplate(CircuitTemplate):
    #in previous circuit templates, everytime we call build it extends based on a predefined pattern
    #now for mixed basis sets, polytope coverage informs a better way to mix and match
    #this method needs to override the build method
    
    #this means monodromy_range_from_target needs to return a circuit polytope
    #we update template to match circuit polytope shape
    #then tell optimizer range to be range(1) so it knows not to call build again
    def __init__(self, base_gates=[CustomCostGate], no_exterior_1q=False, preseed=False, chatty_build=True, cost_1q=0, bare_cost=False):
        super().__init__(n_qubits=2, base_gates=base_gates, edge_params=[(0,1)], no_exterior_1q=no_exterior_1q, use_polytopes=True, preseed=preseed)
        #precomputed polytopes
        self.coverage, self.gate_hash = gate_set_to_coverage(*base_gates, chatty=chatty_build, cost_1q=cost_1q, bare_cost=bare_cost)

    def set_polytope(self, circuit_polytope):
        self.circuit_polytope = circuit_polytope
        self.cost = circuit_polytope.cost #?
    
    def unit_cost(self, n_):
        return self.cost

    def _reset(self):
        self.circuit_polytope = None
        super()._reset()

    def build(self, n_repetitions):
        assert self.circuit_polytope is not None
        #convert circuit polytope into a qiskit circuit with variation 1q params
        gate_list = [self.gate_hash[gate_key] for gate_key in self.circuit_polytope.operations]
        self.gate_2q_base = cycle(gate_list)
        super().build(n_repetitions=len(gate_list))