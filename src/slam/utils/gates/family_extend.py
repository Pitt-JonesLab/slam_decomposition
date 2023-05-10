
import logging

logger = logging.getLogger()

import numpy as np
from slam.utils.gates.custom_gates import ConversionGainGate
from qiskit import QuantumCircuit
from slam.optimizer import TemplateOptimizer
from slam.utils.polytopes.polytope_wrap import monodromy_range_from_target
from slam.basis import CircuitTemplate, MixedOrderBasisCircuitTemplate
from slam.utils.gates.custom_gates import ConversionGainGate
from slam.cost_function import SquareCost
from slam.basisv2 import CircuitTemplateV2
from weylchamber import c1c2c3


def recursive_sibling_check(basis:CircuitTemplate, target_u, basis_factor = 1, rec_iter_factor=1, cost_1q=.1, use_smush=False):
    """Function used to instantiate a circuit using 1Q gate simplification rules
    basis_factor is duration of root basis gate"""

    # get class of current recursion level gate
    child_gate = next(basis.gate_2q_base)

    # check if target_u is identity
    if np.allclose(target_u, np.eye(4)):
        qc = QuantumCircuit(2)
        return qc, 0
    
    # if child gate is locally equivalent to target gate, then we want to see if they can be equal using phase and VZ gates
    # I'm not checking this for now because slow and only small benefit
    if False and np.all(np.isclose(c1c2c3(child_gate.to_matrix()), c1c2c3(target_u))):
        phase_lambda = lambda p1, p2: ConversionGainGate(p1, p2, child_gate.params[2], child_gate.params[3], t_el=child_gate.params[-1])
        template = CircuitTemplateV2(base_gates = [phase_lambda], maximum_span_guess=1, vz_only=True)
        template.spanning_range = range(1,2)
        optimizer3 = TemplateOptimizer(basis=template, objective=SquareCost(), override_fail=True, success_threshold = 1e-10, training_restarts=1)
        ret3 = optimizer3.approximate_target_U(target_U = target_u)
        if ret3.success_label:
            # basis.no_exterior_1q = True
            basis.vz_only = True
            basis.build(1)
            return basis, basis_factor

    # first get necessary range using basis
    ki = monodromy_range_from_target(basis, target_u)[0]
    # cost to beat
    child_cost = (ki+1)*cost_1q  + ki * basis_factor

    assert ki >= 1, "Monodromy range must be at least 1, taget is identity gate case not implemented"

    # if basis is locally equivalent to target, return k=1
    if ki == 1:
        basis.no_exterior_1q = False
        basis.build(1)
        return basis, 1.2

    # construct the older sibling, based on parity of ki
    if ki % 2 == 0:
        rec_iter_factor = 2
    else:
        rec_iter_factor = 3
    sib_basis_factor = rec_iter_factor * basis_factor
    older_sibling = ConversionGainGate(*child_gate.params[:-1], t_el=child_gate.params[-1] * rec_iter_factor)
    #TODO need to set a duration attribute here
    older_sibling.normalize_duration(1)

    # stop condition, if sibling is bigger than iswap
    #TODO, what we can check is a mixed basis (e.g. rather than 3 sqiswaps becoming 1.5 iswap and fail, make it 1 iswap and 1 sqiswap)
    if older_sibling.params[2] + older_sibling.params[3] <= np.pi/2:
        #new basis using older sibling
        sibling_basis = MixedOrderBasisCircuitTemplate(base_gates=[older_sibling], chatty_build=False, use_smush_polytope=use_smush)
        sibling_decomp, sib_score = recursive_sibling_check(sibling_basis, target_u, use_smush=use_smush, basis_factor=sib_basis_factor, rec_iter_factor=rec_iter_factor, cost_1q=cost_1q)
    else:
        sib_score = np.inf

    # if length of qc is shorter using the siblings decomp template, else use self template
    if sib_score < child_cost:
        return sibling_decomp, sib_score
    else:
        basis.build(ki)
        return basis, child_cost


if __name__ == "__main__":
    from qiskit.circuit.library import CXGate
    from slam.utils.gates.duraton_scaling import atomic_cost_scaling
    target = CXGate().to_matrix()
    params = [0,0, 0, np.pi/16, 1]
    basis = ConversionGainGate(*params)
    template = MixedOrderBasisCircuitTemplate(base_gates=[basis], chatty_build=False, use_smush_polytope=0)
    duration = atomic_cost_scaling(params, 1, speed_method='linear', duration_1q=0)
    ret = recursive_sibling_check(template, target, cost_1q=0.1, basis_factor=duration[1])
    decomp_cost = ret[1]
    print(decomp_cost)