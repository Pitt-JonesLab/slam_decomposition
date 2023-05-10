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

from slam.hamiltonian import Hamiltonian
from slam.utils.gates.custom_gates import *
from slam.utils.data_utils import filename_encode, pickle_load, pickle_save
from slam.utils.polytopes.polytope_wrap import (gate_set_to_coverage,
                                  monodromy_range_from_target)
import pickle
import os
from typing import List
from config import srcpath
from slam.basis_abc import VariationalTemplate

class HamiltonianTemplate(VariationalTemplate):
    def __init__(self, h:Hamiltonian):
        self.filename = filename_encode(repr(h))
        self.h = h
        self.spanning_range = range(1)
        self.using_bounds = False
        self.using_constraints = False
        self.bounds_list = None
        self.constraint_func = None
        super().__init__(preseed=False, use_polytopes=False)
    
    def get_spanning_range(self, target_u):
        return range(1,2) #only need to build once, in lieu of a circuit template
    
    def eval(self, Xk):
        return self.h.construct_U(*Xk).full()
    
    def parameter_guess(self,t=1):
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
        
    def get_spanning_range(self, target_u):
        if not self.use_polytopes:
            return self.spanning_range
        else:
            #call monodromy polytope helper
            return monodromy_range_from_target(self, target_u)

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
        edge = next(next(self.gate_2q_edges)) #call cycle twice to increment gate index then edge
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
    def __init__(self, base_gates:List[CustomCostGate], chatty_build=True, cost_1q=0, bare_cost=True, coverage_saved_memory=True, use_smush_polytope=False, **kwargs) -> None:
        self.homogenous = len(base_gates) == 1

        if cost_1q != 0 or bare_cost==False:
            logging.warning("rather than setting cost_1q, use bare_cost=True and scale the cost afterwards - that way don't have misses in saved memory.\
                (see bgatev2script.py for implementation)")
            raise ValueError("just don't do this lol")
        
        if not all([isinstance(gate, ConversionGainGate) for gate in base_gates]):
            raise ValueError("all base gates must be ConversionGainGate")

        
        # set gc < gg so that we can use the same polytope for both cases
        #XXX note this means the gate_hash will refer to the wrong gate, but in speedlimit pass we override build() with scaled_gate param anyway
        new_base_gates = []
        for gate in base_gates:
            if gate.params[2] < gate.params[3]:
                new_base_gates.append(gate)
            else:
                new_params = gate.params
                # swap gc and gg
                new_params[2], new_params[3] = new_params[3], new_params[2]
                temp_new_gate = ConversionGainGate(*new_params)
                new_base_gates.append(temp_new_gate)
        base_gates = new_base_gates
        # assuming bare costs we should normalize the gate duration to 1
        for gate in base_gates:
            gate.normalize_duration(1)

        super().__init__(n_qubits=2, base_gates=base_gates, edge_params=[[(0,1)]], no_exterior_1q=False, use_polytopes=True, preseed=False)

        if coverage_saved_memory:
            # used list comprehension so each CG gate has its own str function called
            file_hash = str([str(g) for g in base_gates])
            if use_smush_polytope:
                file_hash += "smush"

            filepath = f"{srcpath}/data/polytopes/polytope_coverage_{file_hash}.pkl"
            
            while True:
                # try load from memory
                if os.path.exists(filepath): #XXX hardcoded file path'
                    logging.debug("loading polytope coverage from memory")
                    with open(filepath, "rb") as f:
                        #NOTE this is hacky monkey patch, if we had more time we would transfer the old values to use this formatting
                        # non smushes use the h5 data loads, but we wanted to make variations of the gate so overriding that h5 data by storing an optional value in the class
                        if use_smush_polytope:
                            self.coverage, self.gate_hash, self.scores = pickle.load(f)
                            return 
                        else:
                            self.coverage, self.gate_hash = pickle.load(f)
                            self.scores = None
                            return
                elif use_smush_polytope:
                    raise ValueError("Smush Polytope not in memory, need to compute using parallel_drive_volume.py")
                    logging.warning("Failed to load smush, using non-smush instead")
                    file_hash = file_hash[:-5]
                    use_smush_polytope = False
                    filepath = f"{srcpath}/data/polytopes/polytope_coverage_{file_hash}.pkl"
                else:
                    # if not in memory, compute and save
                    logging.warning(f"No saved polytope! computing polytope coverage for {file_hash}")
                    self.coverage, self.gate_hash = gate_set_to_coverage(*base_gates, chatty=chatty_build, cost_1q=cost_1q, bare_cost=bare_cost)
                    with open(filepath, "wb") as f:
                        pickle.dump((self.coverage, self.gate_hash), f)
                        logging.debug("saved polytope coverage to file")
                    return
        else:
            self.coverage, self.gate_hash = gate_set_to_coverage(*base_gates, chatty=chatty_build, cost_1q=cost_1q, bare_cost=bare_cost)


    def set_polytope(self, circuit_polytope):
        self.circuit_polytope = circuit_polytope
        self.cost = circuit_polytope.cost #?
    
    def unit_cost(self, n_):
        return self.cost

    def _reset(self):
        self.circuit_polytope = None
        super()._reset()

    def build(self, n_repetitions, scaled_gate = None):
        """the reason the build method is being overriden is specifically for mixed basis sets
        we want to be able to build circuits which have an arbitrary order of basis gates so we have to use info from monodromy
        monodromy communicates the order from a set polytope (which doesn't have access to gate object proper) via a hash
        that lets override the gate_2q_base generator"""

        assert self.circuit_polytope is not None

        # NOTE: overriding the 2Q gate if we want to use a speed limited gate
        # used to manually set a duration attirbute
        if scaled_gate is not None:
            if not self.homogenous:
                raise ValueError("Can't use this hacky substitute method for mixed basis sets")
            gate_list = [scaled_gate]*n_repetitions
        else:
            #convert circuit polytope into a qiskit circuit with variation 1q params
            gate_list = [self.gate_hash[gate_key] for gate_key in self.circuit_polytope.operations]

        self.gate_2q_base = cycle(gate_list)
        assert n_repetitions == len(gate_list)
        super().build(n_repetitions=len(gate_list))