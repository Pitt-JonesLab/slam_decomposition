from abc import ABC

import numpy as np
from weylchamber import J_T_LI, bell_basis, c1c2c3, canonical_gate, g1g2g3

"""
Defines functions that the optimizer attempts to minimize, 
Each function is some metric of fidelity between unitaries
Experiment to find some metrics perform better/faster than others
"""

class UnitaryCostFunction(ABC):
    def __init__(self):
        pass

    def unitary_fidelity(self, current_u, target_u):
        raise NotImplementedError

    def fidelity_lambda(self, target_u):
        return lambda current_u: self.unitary_fidelity(current_u, target_u)

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
