# %%
import logging

logger = logging.getLogger()

import numpy as np


# %%
from slam.utils.gates.custom_gates import ConversionGainGate
from qiskit import QuantumCircuit
from slam.sampler import GateSample
from slam.optimizer import TemplateOptimizer
from slam.utils.monodromy.polytope_wrap import (
    monodromy_range_from_target,
    coverage_to_haar_expectation
)
from slam.basis import CircuitTemplate, MixedOrderBasisCircuitTemplate
from slam.utils.visualize import (
    unitary_to_weyl,
    unitary_2dlist_weyl,
    coordinate_2dlist_weyl,
)
from slam.utils.gates.custom_gates import CustomCostGate, ConversionGainGate
from slam.cost_function import SquareCost
from slam.utils.gates.custom_gates import ConversionGainSmushGate
from slam.basisv2 import CircuitTemplateV2
from qiskit.circuit.library import CXGate, SwapGate
from weylchamber import c1c2c3
from slam.utils.gates.snail_death_gate import SpeedLimitedGate
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time

# %%
# # verifying that relative phase doesn't change 2Q gate location
# unitary = [ConversionGainGate(0, 0, p*0.5*np.pi, (1-p)*0.4*np.pi) for p in np.linspace(0,1,16)]
# print([u.cost() for u in unitary])
# unitary_to_weyl(*unitary);

# %%
# plotting to make sure getting good weyl coverage before running data collection
"""Here we are building the set of gates to collect data over
The math here is translating gates to left side of x-axis to remove duplicates"""

# FIXME want both symmetries of gc and gg even if at same coordinate
def build_gates(elim_extra_weyl=True):
    unitary_list = []
    coordinate_list = []
    for k in np.linspace(0, 0.5, 17):  # 17
        inner_list = []
        for p in np.linspace(0, 1, 21):  # 21
            gate = ConversionGainGate(0, 0, p * k * np.pi, (1 - p) * k * np.pi)
            unitary = gate.to_matrix()
            c = list(c1c2c3(np.array(unitary)))

            if elim_extra_weyl:
                if c[0] > 0.5:
                    c[0] = -1 * c[0] + 1 #1-x

            if c in inner_list or any(c in inner for inner in coordinate_list):
                continue

            inner_list.append(c)
            unitary_list.append(gate)

        coordinate_list.append(inner_list)
    return unitary_list, coordinate_list


# %%
# unitary_list, coordinate_list = build_gates()
# coordinate_2dlist_weyl(*coordinate_list);

# %%
import h5py

filename = "/home/evm9/decomposition_EM/data/cg_gates.h5"


def collect_data(unitary_list, overwrite=False):
    """Using bare costs in mixedorderbasis template - this means the costs are in terms of number of gates
    means we need to scale by costs later - but don't have to recompute each time :)"""
    with h5py.File(filename, "a") as hf:
        g = hf.require_group(
            "bare_cost"
        )  # 'bare_cost' is populated, trying '_coverage' to test if i can save coverage objects

        if overwrite:
            g.clear()

        for base_gate in tqdm(
            unitary_list[1:]
        ):  # skip the identity gate - can't build valid coverage from it
            if str(base_gate) in g:
                logging.debug(f"{str(base_gate)} already in file")
                continue

            template = MixedOrderBasisCircuitTemplate(
                base_gates=[base_gate], chatty_build=0, bare_cost=True
            )

            # track elapsed time
            start = time.time()
            haar_score = coverage_to_haar_expectation(template.coverage, chatty=0)
            haar_time = time.time()
            cnot_score = monodromy_range_from_target(
                template, target_u=CXGate().to_matrix()
            )[0]
            cnot_time = time.time()
            swap_score = monodromy_range_from_target(
                template, target_u=SwapGate().to_matrix()
            )[0]
            swap_time = time.time()

            # log all the elapsed times in seconds
            logging.debug(
                f"TIMING: haar: {haar_time-start}, cnot: {cnot_time-haar_time}, swap: {swap_time-cnot_time}"
            )
            # log all the scores
            logging.debug(
                f"(BARE) SCORES: haar: {haar_score}, cnot: {cnot_score}, swap: {swap_score}"
            )

            # also add the gate when gc and gg are switched
            # NOTE we know that switching gc and gg make a locally invariant gate so we don't need to compute their coverage separately
            # TODO

            # FIXME adding a None to end of score list makes so not a jagged 2d array - can fix better later
            # XXX is a problem if changing size of base_gate.params
            g.create_dataset(
                str(base_gate),
                data=np.array(
                    [base_gate.params, [haar_score, cnot_score, swap_score, -1, -1]]
                ),
            )


def get_group_name(speed_method="linear", duration_1q=0):
    return f"{speed_method}_scaling_1q{duration_1q}"


def get_method_duration(group_name):
    """returns the speed_method and duration_1q from the group name"""
    speed_method = group_name.split("_")[0]
    duration_1q = float(group_name.split("_")[-1].replace("1q", ""))
    return speed_method, duration_1q


def recursive_sibling_check(basis:CircuitTemplate, target_u, basis_factor = 1, rec_iter_factor=1, cost_1q=.1, use_smush=False):
    """Function used to instantiate a circuit using 1Q gate simplification rules
    basis_factor is duration of root basis gate"""
    # check if template basis unitary is equal to target unitary using numpy
   
    child_gate = next(basis.gate_2q_base)

    # check if target_u is identity
    if np.allclose(target_u, np.eye(4)):
        qc = QuantumCircuit(2)
        return qc, 0
    
    # if child gate is locally equivalent to target gate, then we want to see if they can be equal using phase and VZ gates
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

    older_sibling.normalize_duration(1)

    # stop condition, if sibling is bigger than iswap
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
            raise NotImplementedError("Fam scaling not implemented for haar, calculate it by hand lol")
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


# %%
def plot_eharr(group_name, metric=0):
    with h5py.File(filename, "r") as hf:
        g = hf.require_group(group_name)
        x, y, z = [], [], []
        for v in g.values():
            params = v[0]
            scores = np.array(v[1])
            x.append(params[2])  # gain
            y.append(params[3])  # conv
            z.append(
                scores[metric]
            )  # change this to 0 or 1 or 2 for haar or cnot or swap

        # plot E[harr] vs gain and conv
        plt.close()
        plt.figure()
        plt.scatter(x, y, s=100, c=z)
        plt.ylabel("gain")
        plt.xlabel("conv")
        cbar = plt.colorbar()
        cbar.set_label("E[haar]", rotation=90)


# %%
def pick_winner(group_name, metric=0, target_ops=None, tqdm_bool=True, plot=True, smush_bool=False, family_extension=False):
    # TODO add a tiebreaker between any ties
    """pick the gate with the lowest score for the given metric
    params:
        metric = 0 is haar, 1 is cnot, 2 is swap, (-1, lambda) if custom"""
    with h5py.File(filename, "r") as hf:
        # FIXME deprecated use of presaved groups so all the scaling is done in the same place
        g = hf.require_group("bare_cost")  # switching to use bare_cost
        speed_method, duration_1q = get_method_duration(
            group_name
        )  # but still use the custom group name to pass these parameters

        # minimize score over target operations or provided metric
        winner = None

        for v in tqdm(g.values()) if tqdm_bool else g.values():

            base_gate = ConversionGainGate(*v[0])
            try:
                template = MixedOrderBasisCircuitTemplate(
                    base_gates=[base_gate], chatty_build=0, bare_cost=True, use_smush_polytope=smush_bool
                )
            except Exception as e: # this would fail if we we tried to load a smush gate but wasn't precomputed
                if 'Polytope not in memory' in str(e):
                    continue # this is expected since we only precomputed for the 6 main gates, so just skip
                else:
                    raise e

            if np.any(np.array(template.scores) == None):
                continue
 
            candidate_score = 0
            scaled_gate = None  # used for skipping reconstruction in the atomic cost scaling function

            if metric in [0, 1, 2] and target_ops is None:
                if template.scores is not None:
                    target_score = template.scores[metric]
                else:
                    target_score = v[1][metric]
                scaled_gate, scaled_score = atomic_cost_scaling(
                    params=v[0],
                    scores=target_score,
                    speed_method=speed_method,
                    duration_1q=duration_1q,
                    scaled_gate=scaled_gate,
                    family_extension=family_extension,
                    use_smush=smush_bool,
                    metric = metric
                )
                candidate_score = scaled_score

            elif metric[0] == -1 and target_ops is None:  # used for lambda custom
                lambda_weight = metric[1]
                if template.scores is not None:
                    cnot_score = template.scores[1]
                    swap_score = template.scores[2]
                else:
                    cnot_score = v[1][1]  # cnot
                    swap_score = v[1][2]  # swap
                custom_score = (
                    lambda_weight * cnot_score + (1 - lambda_weight) * swap_score
                )
                target_score = custom_score
                scaled_gate, scaled_score = atomic_cost_scaling(
                    params=v[0],
                    scores=target_score,
                    speed_method=speed_method,
                    duration_1q=duration_1q,
                    scaled_gate=scaled_gate,
                    family_extension=family_extension,
                    use_smush=smush_bool,
                    metric=metric
                )
                candidate_score = scaled_score

            else:  # used exact distribution, minimizes cost over every possible gate
                for target in target_ops:
                    target_score = monodromy_range_from_target(
                        template, target_u=target.to_matrix()
                    )[0]
                    scaled_gate, scaled_score = atomic_cost_scaling(
                        params=v[0],
                        scores=target_score,
                        speed_method=speed_method,
                        duration_1q=duration_1q,
                        scaled_gate=scaled_gate,
                        family_extension=family_extension,
                        use_smush=smush_bool,
                        metric=metric
                    )
                    candidate_score += scaled_score

            if winner is None or candidate_score < winner_score:
                winner = v
                winner_score = candidate_score
                winner_scaled_gate = scaled_gate
                winner_scaled_score = scaled_score

        winner_gate = ConversionGainGate(*winner[0])  # 0 is params, 1 is scores
        # log winner params, scores, cost
        logging.info(
            f"winner: {winner_gate}, scores: {winner[1][:-2]}, cost: {winner_gate.cost()}"
        )
        if not target_ops is None:
            # log weigted score and normalized score
            logging.info(
                f"winner score: {winner_score}, normalized score: {winner_score/len(target_ops)}"
            )
        logging.info(
            f"scaled scores: {winner_scaled_score}, scaled cost: {winner_scaled_gate.cost()}"
        )

        if plot:
            unitary_to_weyl(winner_gate.to_matrix())
            # uncomment to see the gate in weyl space
        return winner_gate, winner_scaled_gate


# # %%
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    unitary_list, coordinate_list = build_gates()
    collect_data(unitary_list, overwrite=0)
