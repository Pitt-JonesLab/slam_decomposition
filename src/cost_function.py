from abc import ABC

import numpy as np
from weylchamber import J_T_LI, bell_basis, c1c2c3, canonical_gate, g1g2g3
import logging
from qiskit.quantum_info import *
from qiskit import QuantumCircuit
"""
Defines functions that the optimizer attempts to minimize, 
Each function is some metric of fidelity between unitaries, where 0 means best and 1 means worst
Experiment to find some metrics perform better/faster than others
"""
class EntanglementCostFunction(ABC):
    #concurrence, mutual info, negativity, entanglement of formation, entropy of entanglement
    def __init__(self, state='w'):
        self.state = state
    def entanglement_monotone(self, qc):
        self.state_prep = QuantumCircuit(3)
        if self.state == "w":
            self.state_prep.ry(2*np.arccos(1/np.sqrt(3)),0)
            self.state_prep.ch(0,1)
            self.state_prep.cx(1,2)
            self.state_prep.cx(0,1)
            self.state_prep.x(0)
        elif self.state == "ghz":
            self.state_prep.h(0)
            self.state_prep.cx(0, 1)
            self.state_prep.cx(0, 2)
        else:
            raise NotImplementedError
            
        self.state_prep.barrier()
        self.full = self.state_prep.compose(qc)
        self.statevector = Statevector(self.full)

class MutualInformation(EntanglementCostFunction):
    # I could code this to be more flexible
    # for now I am going to hardcode 3Q states in with partial tracing
    def entanglement_monotone(self, qc):
        #append on the entangled state circuit
        #the goal of minimizing the cost means undoing the entangled state
        super().entanglement_monotone(qc)
        state1 = partial_trace(self.statevector, [0])
        state2 = partial_trace(self.statevector, [1])
        state3 = partial_trace(self.statevector, [2])
        return sum([mutual_information(state1), mutual_information(state2), mutual_information(state3)])

class MutualInformationSquare(EntanglementCostFunction):
    def entanglement_monotone(self, qc):
        super().entanglement_monotone(qc)
        state1 = partial_trace(self.statevector, [0])
        state2 = partial_trace(self.statevector, [1])
        state3 = partial_trace(self.statevector, [2])
        return sum([mutual_information(state1)**2, mutual_information(state2)**2, mutual_information(state3)**2])

class Negativity(EntanglementCostFunction):
    def entanglement_monotone(self, qc):
        return super().entanglement_monotone(qc)

class Formation(EntanglementCostFunction):
    def entanglement_monotone(self, qc):
        return super().entanglement_monotone(qc)

class EntropyofEntanglement(EntanglementCostFunction):
    def entanglement_monotone(self, qc):
        return super().entanglement_monotone(qc)

class UnitaryCostFunction(ABC):
    def __init__(self):
        #experimenting with this
        #normalize cost using max_cost = c(swap, identity)
        swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]])
        id = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0],[0,0,0,1]])
        self.normalization = 1 #self.unitary_fidelity(id, swap)
        logging.debug(self.normalization)

    def unitary_fidelity(self, current_u, target_u):
        raise NotImplementedError

    # def fidelity_lambda(self, target_u):
    #     return lambda current_u: self.unitary_fidelity(current_u, target_u)

class BasicCostInverse(UnitaryCostFunction):
    #don't subtract 1
    def unitary_fidelity(self, current_u, target_u):
        h = np.matrix(target_u).getH()
        return np.abs(np.trace(np.matmul(h, current_u)))/ np.array(current_u).shape[0]

class BasicCost(UnitaryCostFunction):
    def unitary_fidelity(self, current_u, target_u):
        h = np.matrix(target_u).getH()
        return 1 - np.abs(np.trace(np.matmul(h, current_u)))/ np.array(current_u).shape[0]

class SquareCost(UnitaryCostFunction):
    def unitary_fidelity(self, current_u, target_u):
        h = np.matrix(target_u).getH()
        d = np.array(target_u).shape[0]
        return 1 - (np.abs(np.trace(np.matmul(h, current_u))) ** 2+ d) / (d * (d + 1))

class BasicReducedCost(BasicCost):
    # version that eliminates exterior 1Q gates by converting to can basis
    # need to also convert the template to can basis for similarity
    def unitary_fidelity(self, current_u, target_u):
        can_target = np.matrix(canonical_gate(*c1c2c3(target_u)))
        can_current = np.matrix(canonical_gate(*c1c2c3(current_u)))
        return super().unitary_fidelity(can_current, can_target)

class SquareReducedCost(SquareCost):
    def unitary_fidelity(self, current_u, target_u):
        can_target = np.matrix(canonical_gate(*c1c2c3(target_u)))
        can_current = np.matrix(canonical_gate(*c1c2c3(current_u)))
        return super().unitary_fidelity(can_current, can_target)

class SquareReducedBellCost(SquareCost):
    def unitary_fidelity(self, current_u, target_u):
        bell_target = np.matrix(bell_basis(target_u))
        bell_current = np.matrix(bell_basis(current_u))
        return super().unitary_fidelity(bell_current, bell_target)

class WeylEuclideanCost(UnitaryCostFunction):
    def unitary_fidelity(self, current_u, target_u):
        if (4,4) != current_u.shape:
            raise ValueError("Weyl chamber only for 2Q gates")

        c_target = c1c2c3(target_u)
        c_current = c1c2c3(current_u)
        return np.linalg.norm(np.array(c_target) - np.array(c_current))

class MakhlinEuclideanCost(UnitaryCostFunction):
    def unitary_fidelity(self, current_u, target_u):
        if (4,4) != current_u.shape:
            raise ValueError("Weyl chamber only for 2Q gates")

        g_target = g1g2g3(target_u)
        g_current = g1g2g3(current_u)
        return np.linalg.norm(np.array(g_target) - np.array(g_current))

class MakhlinFunctionalCost(UnitaryCostFunction):
    def unitary_fidelity(self, current_u, target_u):
        return J_T_LI(target_u, current_u)
