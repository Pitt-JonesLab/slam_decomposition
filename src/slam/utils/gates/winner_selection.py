import logging

logger = logging.getLogger()

import h5py
import numpy as np
from tqdm import tqdm

from slam.basis import MixedOrderBasisCircuitTemplate
from slam.utils.gates.bare_candidates import filename, get_method_duration
from slam.utils.gates.custom_gates import ConversionGainGate
from slam.utils.gates.duraton_scaling import atomic_cost_scaling
from slam.utils.polytopes.polytope_wrap import monodromy_range_from_target
from slam.utils.visualize import unitary_to_weyl


def pick_winner(
    group_name,
    metric=0,
    target_ops=None,
    tqdm_bool=True,
    plot=True,
    smush_bool=False,
    family_extension=False,
):
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
                    base_gates=[base_gate],
                    chatty_build=0,
                    bare_cost=True,
                    use_smush_polytope=smush_bool,
                )
            except (
                Exception
            ) as e:  # this would fail if we we tried to load a smush gate but wasn't precomputed
                if "Polytope not in memory" in str(e):
                    continue  # this is expected since we only precomputed for the 6 main gates, so just skip
                else:
                    raise e

            if np.any(np.array(template.scores) is None):
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
                    metric=metric,
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
                    metric=metric,
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
                        metric=metric,
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
        if target_ops is not None:
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
