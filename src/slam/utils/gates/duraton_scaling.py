import logging
logger = logging.getLogger()

import numpy as np

from slam.utils.gates.custom_gates import ConversionGainGate
from slam.basis import MixedOrderBasisCircuitTemplate
from qiskit.circuit.library import CXGate, SwapGate
from slam.utils.gates.snail_death_gate import SpeedLimitedGate
import h5py
from slam.utils.gates.family_extend import recursive_sibling_check
from slam.utils.gates.bare_candidates import filename, get_group_name, get_method_duration

def atomic_cost_scaling(
    params, scores, speed_method="linear", duration_1q=0, scaled_gate=None, use_smush=False, family_extension=False, metric=None
):
    if scaled_gate is None:
        # defined speed limit functions
        # equation for offset circle
        # center of circle is (-c, -c) with intercepts at np.pi/2 on x and y axes
        c = np.pi / 4
        mid_sl = lambda x: 0.5 * (
            -2 * c
            + np.sqrt(4 * c**2 - 8 * c * x + 4 * c * np.pi - 4 * x**2 + np.pi**2)
        )
        squared_sl = lambda x: np.sqrt((np.pi / 2) ** 2 - x**2)

        # reconstruct gate object to get cost
        if "hardware" in speed_method:
            gate = SpeedLimitedGate(*params)  # speed_limit_function = spline
        elif "mid" in speed_method:
            gate = SpeedLimitedGate(*params, speed_limit_function=mid_sl)
        elif "squared" in speed_method:
            gate = SpeedLimitedGate(*params, speed_limit_function=squared_sl)
        elif "linear" in speed_method:
            # spline is the hardware characterized speed limit which is constructed inside the module proper
            gate = ConversionGainGate(*params)
        elif "bare" in speed_method:
            gate = ConversionGainGate(*params)
        else:
            raise ValueError("invalid speed_method")
    else:
        gate = scaled_gate

    # compute scaled costs
    # if bare don't scale
    if "bare" in speed_method:
        scaled_scores = scores
    else:
        scaled_scores = scores * gate.cost()  # scale by 2Q gate cost

    # do check over parent family gates to see how many if interior gates are needed
    # TODO compute haar
    if family_extension:
        basis = ConversionGainGate(*params)
        template = MixedOrderBasisCircuitTemplate(base_gates=[basis], chatty_build=False, use_smush_polytope=use_smush)
        from qiskit.circuit.library import SwapGate
        targets = []
        #TODO XXX FIXME, metric is (-1, lambda)
        # custom_score = (lambda_weight * cnot_score + (1 - lambda_weight) * swap_score)
        if metric is None:
            targets = [CXGate().to_matrix(), SwapGate().to_matrix()]
        if metric == 0:
            raise NotImplementedError("Famextend scaling not implemented for haar, calculate it by hand lol")
        if metric == 1:
            targets = [CXGate().to_matrix()]
        if metric == 2:
            targets = [SwapGate().to_matrix()]
        for score_index, gate_target in enumerate(targets):
            ret = recursive_sibling_check(template, gate_target, cost_1q=duration_1q, basis_factor=gate.cost())
            if len(targets) == 1:
                return gate, ret[1]
            else:
                scaled_scores[score_index+1] = ret[1] # add 1 to index to skip over Haar score
        if len(targets) ==1:
            scaled_scores[0] = (scores + 1) * duration_1q  # scale by 1Q gate cost
            return gate, scaled_scores
    else:
        scaled_scores += (scores + 1) * duration_1q  # scale by 1Q gate cost
    return gate, scaled_scores


def cost_scaling(speed_method="linear", duration_1q=0, overwrite=1, query_params=None, family_extension=False, use_smush=False):
    """Use bare costs to add in costs of 2Q gate and 1Q gates"""
    # TODO needs to be deprecated in favor of atomic_cost_scaling
    # loading from saved makes messy using family and smush

    group_name = get_group_name(speed_method, duration_1q)
    with h5py.File(filename, "a") as hf:
        g = hf.require_group("bare_cost")
        g2 = hf.require_group(group_name)

        for v in g.values():

            params = v[0]
            
            # only allow family on cnot, b, swap
            pass_flag = False or not family_extension
            if family_extension:
                logging.warning("Family Extension only covers CNOT, B, and SWAP family gates")
            if family_extension and (params[2] == 0 or params[3] == 0):
                #iswap family
                pass_flag = True
            elif family_extension and (params[2]/params[3] == 3 or params[3]/params[2] == 3):
                #cnot family
                pass_flag = True
            elif family_extension and (params[2] == params[3]):
                # cnot family
                pass_flag = True
            if not pass_flag:
                continue

            base_gate = ConversionGainGate(*params)

            try:
                template = MixedOrderBasisCircuitTemplate(
                    base_gates=[base_gate], chatty_build=0, bare_cost=True, use_smush_polytope=use_smush
                )
            except Exception as e: # this would fail if we we tried to load a smush gate but wasn't precomputed
                if 'Polytope not in memory' in str(e):
                    continue # this is expected since we only precomputed for the 6 main gates, so just skip
                else:
                    raise e
            # grabbing smush scores saved as an attribute
            if template.scores is not None:
                scores = np.array(template.scores)
            else:
                scores = np.array(v[1])

            gate, scaled_scores = atomic_cost_scaling(
                params=params,
                scores=scores,
                speed_method=speed_method,
                duration_1q=duration_1q,
                family_extension=family_extension,
                use_smush=use_smush
            )

            if query_params is not None and np.allclose(params, query_params):
                return gate, scaled_scores
            if str(gate) in g2 and not overwrite:
                # print("already have this gate")
                continue

            # store
            if str(gate) in g2:
                # delete
                del g2[str(gate)]
            g2.create_dataset(str(gate), data=[gate.params, scaled_scores])
