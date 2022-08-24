import logging
from abc import ABC
from dataclasses import dataclass
from inspect import signature
from itertools import cycle
from multiprocessing.sharedctypes import Value
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

"""TemplateV2 working on implementing the continuous 2Q search
1. Change base_gate to be a class instead of already initialzied object
2. Eliminate monodromy things (I don't know how it could be used)
Wait actually, what if before we try decomp, we look at the shortest gates that make the monodromy span valid
I'll leave this as a TODO for now because it still breaks for 3Q+ and that's what I want working first
3. Define a cost of the circuit using the Q-params, should be the circuit fidelity which can be used with decomposition fidelity to find total f
"""
from src.basis import VariationalTemplate
class CircuitTemplateV2(VariationalTemplate):
    def __init__(self, n_qubits=2, base_gates=[RiSwapGate], edge_params=[[(0, 1)]], no_exterior_1q=False, use_polytopes=False, maximum_span_guess=5, preseed=False):
        """Initalizes a qiskit.quantumCircuit object with unbound 1Q gate parameters"""
        hash = str(n_qubits)+ str(base_gates)+ str(edge_params)+ str(no_exterior_1q)
        self.filename = filename_encode(hash)
        self.n_qubits = n_qubits
        self.no_exterior_1q = no_exterior_1q

        self.gate_2q_base = cycle(base_gates)
        #each gate gets its on cycler
        self.gate_2q_edges = cycle([cycle(edge_params_el) for edge_params_el in edge_params])
        self.gen_1q_params = self._param_iter()

        #XXX
        self.constraints = {}
        self.bounds = []
        self.using_constraints = False

        #define a range to see how many times we should extend the circuit while in optimization search
        self.spanning_range = None
        if not use_polytopes:
            self.spanning_range = range(1,maximum_span_guess+1)
            self.coverage = None #only precomputed in mixedbasis class
            
        super().__init__(preseed=preseed, use_polytopes=use_polytopes)

        self._reset()

        #deprecated feature
        self.trotter=False
    
    def circuit_fidelity(self, Xk):
        fidelity = 1.0
        qc = self.assign_Xk(Xk)
        #assuming there is only 1 critical path, need to iterate through gates 
        #want to take the product of fidelities
        #for now just hardcode this until we settle on a better way
        for gate in qc:
            if gate[0].name == "riswap":
                a = gate[0].params[0]
                c = RiSwapGate(a).cost() #fidelity
                fidelity = fidelity * c
        return fidelity
               
    def eval(self, Xk):
        """returns an Operator after binding parameter array to template"""
        return Operator(self.assign_Xk(Xk)).data

    #TODO: modify this so the Q-params have a smaller range
    def parameter_guess(self, t=0):
        """returns a np array of random values for each parameter"""
        #parent checking is to handle preseeding
        parent = super().parameter_guess(t)
        if parent is not None:
            return parent
        random_list = []
        #set defaults
        dlow = -2*np.pi
        dhigh = 2*np.pi

        self.bounds = [] #sequence of (min,max) for each element in X passed to scipy optimze
        for parameter in self.circuit.parameters:
            clow, chigh = self.constraints.get(parameter.name, (dlow,dhigh))
            random_list.append(np.random.uniform(clow, chigh, 1))
            self.bounds.append((clow, chigh))
        
        if not self.using_constraints:
            self.bounds = None
        return random_list
        # return np.random.random(len(self.circuit.parameters))* 2 * np.pi

    def add_constraint(self, parameter_name, max=None, min=None):
        self.constraints[parameter_name] = (min, max)
        #con_funs is passed to scipy optimizer
        #quick iteration because I don't have a better way
        p_index = -1
        for index, p in enumerate(self.circuit.parameters):
            if p.name == parameter_name:
                p_index = index
                break
        if p_index == -1:
            raise ValueError("Parameter Name not found")
        
        # self.con_funs.append({'type:ineq':lambda x: max-x[p_index]})
        # self.con_funs.append({'type:ineq':lambda x: x[p_index] - min})
        #flag for safety, can't rebuilt or messes with ordering
        self.using_constraints = True
        return

    def assign_Xk(self, Xk):
        return self.circuit.assign_parameters(
            {parameter: i for parameter, i in zip(self.circuit.parameters, Xk)}
        )

    def _reset(self):
        """Return template to a 0 cycle"""
        self.cycles = 0
        self.circuit = QuantumCircuit(self.n_qubits)
        self.gen_1q_params = self._param_iter() #reset p labels
        self.gen_2q_params = self._param_iter2() #second counter for 2Q gates

    def build(self, n_repetitions):
        # if self.using_constraints:
        #     raise ValueError("Can't build after setting constraints")

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
    
    def _param_iter2(self):
        index = 0
        while True:
            # Check if Parameter already created, then return reference to that variable
            def _filter_param(param):
                return param.name == f"Q{index}"

            res = list(filter(_filter_param, self.circuit.parameters))
            if len(res) == 0:
                yield Parameter(f"Q{index}")
            else:
                yield res[0]
            index += 1


    def _build_cycle(self, initial=False, final=False):
        """Extends template by next nonlocal gate"""
        if initial and not self.no_exterior_1q:
            # before build by extend, add first pair of 1Qs
            for qubit in range(self.n_qubits):
                self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
                #self.circuit.ry(*[next(self.gen_1q_params) for _ in range(1)], qubit)

        gate = next(self.gate_2q_base)
        edge = next(next(self.gate_2q_edges)) #call cycle twice to increment gate then edge
        #inspect to find how many parameters our gate requires
        num2qparams = len(signature(gate).parameters)
        gate_instance = gate(*[next(self.gen_2q_params) for _ in range(num2qparams)])
        self.circuit.append(gate_instance, edge)
        
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

