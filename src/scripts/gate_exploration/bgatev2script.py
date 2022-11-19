# %%
import logging
<<<<<<< HEAD
=======

>>>>>>> master
logger = logging.getLogger()


import sys
<<<<<<< HEAD
=======

>>>>>>> master
sys.path.append("../../../")

import numpy as np


# %%
from src.utils.custom_gates import ConversionGainGate
<<<<<<< HEAD
from src.utils.polytope_wrap import monodromy_range_from_target, coverage_to_haar_expectation
from src.basis import CircuitTemplate, MixedOrderBasisCircuitTemplate
from src.utils.visualize import unitary_to_weyl, unitary_2dlist_weyl, coordinate_2dlist_weyl
from src.utils.custom_gates import CustomCostGate
=======
from src.utils.polytope_wrap import (
    monodromy_range_from_target,
    coverage_to_haar_expectation,
)
from src.basis import CircuitTemplate, MixedOrderBasisCircuitTemplate
from src.utils.visualize import (
    unitary_to_weyl,
    unitary_2dlist_weyl,
    coordinate_2dlist_weyl,
)
from src.utils.custom_gates import CustomCostGate
from src.utils.custom_gates import ConversionGainSmushGate
>>>>>>> master
from qiskit.circuit.library import CXGate, SwapGate
from weylchamber import c1c2c3
from src.utils.snail_death_gate import SpeedLimitedGate
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
<<<<<<< HEAD
#plotting to make sure getting good weyl coverage before running data collection
"""Here we are building the set of gates to collect data over
The math here is translating gates to left side of x-axis to remove duplicates"""

#FIXME want both symmetries of gc and gg even if at same coordinate
def build_gates(elim_extra_weyl=True):
    unitary_list = []
    coordinate_list = []
    for k in np.linspace(0, 0.5, 17): #17
        inner_list = []
        for p in np.linspace(0, 1, 21): #21
            gate = ConversionGainGate(0, 0, p*k*np.pi, (1-p)*k*np.pi)
            unitary= gate.to_matrix()
=======
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
>>>>>>> master
            c = list(c1c2c3(np.array(unitary)))

            if elim_extra_weyl:
                if c[0] > 0.5:
<<<<<<< HEAD
                    c[0] = -1*c[0] + 1
            
=======
                    c[0] = -1 * c[0] + 1 #1-x

>>>>>>> master
            if c in inner_list or any(c in inner for inner in coordinate_list):
                continue

            inner_list.append(c)
            unitary_list.append(gate)

        coordinate_list.append(inner_list)
    return unitary_list, coordinate_list

<<<<<<< HEAD
# %%
#unitary_list, coordinate_list = build_gates()
#coordinate_2dlist_weyl(*coordinate_list);

# %%
import h5py
filename = '/home/evm9/decomposition_EM/data/cg_gates.h5'
=======

# %%
# unitary_list, coordinate_list = build_gates()
# coordinate_2dlist_weyl(*coordinate_list);

# %%
import h5py

filename = "/home/evm9/decomposition_EM/data/cg_gates.h5"

>>>>>>> master

def collect_data(unitary_list, overwrite=False):
    """Using bare costs in mixedorderbasis template - this means the costs are in terms of number of gates
    means we need to scale by costs later - but don't have to recompute each time :)"""
<<<<<<< HEAD
    with h5py.File(filename, 'a') as hf:
        g = hf.require_group('bare_cost') # 'bare_cost' is populated, trying '_coverage' to test if i can save coverage objects
    
        if overwrite:
            g.clear()

        for base_gate in tqdm(unitary_list[1:]): #skip the identity gate - can't build valid coverage from it
            if str(base_gate) in g:
                logging.debug(f'{str(base_gate)} already in file')
                continue

            template = MixedOrderBasisCircuitTemplate(base_gates=[base_gate], chatty_build=0, bare_cost=True)

            #track elapsed time 
            start = time.time()
            haar_score = coverage_to_haar_expectation(template.coverage, chatty=0)
            haar_time = time.time()
            cnot_score = monodromy_range_from_target(template, target_u = CXGate().to_matrix())[0] 
            cnot_time = time.time()
            swap_score = monodromy_range_from_target(template, target_u = SwapGate().to_matrix())[0]
            swap_time = time.time()

            # log all the elapsed times in seconds
            logging.debug(f"TIMING: haar: {haar_time-start}, cnot: {cnot_time-haar_time}, swap: {swap_time-cnot_time}")
            # log all the scores
            logging.debug(f"(BARE) SCORES: haar: {haar_score}, cnot: {cnot_score}, swap: {swap_score}")

            #FIXME adding a None to end of score list makes so not a jagged 2d array - can fix better later
            #XXX is a problem if changing size of base_gate.params
            g.create_dataset(str(base_gate), data=np.array([base_gate.params, [haar_score, cnot_score, swap_score, -1, -1]]))

def get_group_name(speed_method='linear', duration_1q=0):
    return f'{speed_method}_scaling_1q{duration_1q}'

def get_method_duration(group_name):
    """returns the speed_method and duration_1q from the group name"""
    speed_method = group_name.split('_')[0]
    duration_1q = float(group_name.split('_')[-1].replace('1q',''))
    return speed_method, duration_1q

def atomic_cost_scaling(params, scores, speed_method='linear', duration_1q=0, scaled_gate=None):
    if scaled_gate is None:
        #defined speed limit functions
        #equation for offset circle
        # center of circle is (-c, -c) with intercepts at np.pi/2 on x and y axes
        c = np.pi/4
        mid_sl = lambda x: 0.5 * (-2*c + np.sqrt(4*c**2 - 8*c*x  + 4*c*np.pi - 4*x**2 + np.pi**2))
        squared_sl = lambda x: np.sqrt((np.pi/2)**2 - x**2)

        #reconstruct gate object to get cost
        if 'hardware' in speed_method:
            gate = SpeedLimitedGate(*params) # speed_limit_function = spline
        elif 'mid' in speed_method:
            gate = SpeedLimitedGate(*params, speed_limit_function=mid_sl)
        elif 'squared' in speed_method:
            gate = SpeedLimitedGate(*params, speed_limit_function=squared_sl)
        elif 'linear' in speed_method:
            # spline is the hardware characterized speed limit which is constructed inside the module proper
            gate = ConversionGainGate(*params) 
        elif 'bare' in speed_method:
            gate = ConversionGainGate(*params) 
=======
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


def atomic_cost_scaling(
    params, scores, speed_method="linear", duration_1q=0, scaled_gate=None
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
>>>>>>> master
        else:
            raise ValueError("invalid speed_method")
    else:
        gate = scaled_gate

<<<<<<< HEAD
    #compute scaled costs
    # if bare don't scale
    if 'bare' in speed_method:
        scaled_scores = scores
    else:
        scaled_scores = scores * gate.cost() #scale by 2Q gate cost

    scaled_scores += (scores + 1) * duration_1q #scale by 1Q gate cost
    return gate, scaled_scores


def cost_scaling(speed_method='linear', duration_1q=0, overwrite=1, query_params=None):
    """Use bare costs to add in costs of 2Q gate and 1Q gates"""
    group_name = get_group_name(speed_method, duration_1q)
    with h5py.File(filename, 'a') as hf:
        g = hf.require_group('bare_cost')
=======
    # compute scaled costs
    # if bare don't scale
    if "bare" in speed_method:
        scaled_scores = scores
    else:
        scaled_scores = scores * gate.cost()  # scale by 2Q gate cost

    scaled_scores += (scores + 1) * duration_1q  # scale by 1Q gate cost
    return gate, scaled_scores


def cost_scaling(speed_method="linear", duration_1q=0, overwrite=1, query_params=None):
    """Use bare costs to add in costs of 2Q gate and 1Q gates"""
    group_name = get_group_name(speed_method, duration_1q)
    with h5py.File(filename, "a") as hf:
        g = hf.require_group("bare_cost")
>>>>>>> master
        g2 = hf.require_group(group_name)

        for v in g.values():
            params = v[0]
            scores = np.array(v[1])
<<<<<<< HEAD
            gate, scaled_scores = atomic_cost_scaling(params=params, scores=scores, speed_method=speed_method, duration_1q=duration_1q)
=======
            gate, scaled_scores = atomic_cost_scaling(
                params=params,
                scores=scores,
                speed_method=speed_method,
                duration_1q=duration_1q,
            )
>>>>>>> master

            if query_params is not None and np.allclose(params, query_params):
                return gate, scaled_scores
            if str(gate) in g2 and not overwrite:
<<<<<<< HEAD
                #print("already have this gate")
                continue

            #store
=======
                # print("already have this gate")
                continue

            # store
>>>>>>> master
            if str(gate) in g2:
                # delete
                del g2[str(gate)]
            g2.create_dataset(str(gate), data=[gate.params, scaled_scores])

<<<<<<< HEAD
# %%
def plot_eharr(group_name, metric=0):
    with h5py.File(filename, 'r') as hf:
        g = hf.require_group(group_name)
        x,y, z = [], [], []
        for v in g.values():
            params = v[0]
            scores = np.array(v[1])
            x.append(params[2]) #gain
            y.append(params[3]) #conv
            z.append(scores[metric]) #change this to 0 or 1 or 2 for haar or cnot or swap

        #plot E[harr] vs gain and conv
        plt.close()    
        plt.figure()
        plt.scatter(x, y, s=100, c=z)
        plt.ylabel("gain")
        plt.xlabel('conv')
        cbar = plt.colorbar()
        cbar.set_label("E[haar]", rotation=90)

# %%
def pick_winner(group_name, metric=0, target_ops=None, tqdm_bool=True, plot=True):
    # TODO add a tiebreaker between any ties
    """pick the gate with the lowest score for the given metric
    params:
        metric = 0 is haar, 1 is cnot, 2 is swap"""
    with h5py.File(filename, 'r') as hf:
        # FIXME deprecated use of presaved groups so all the scaling is done in the same place
        g = hf.require_group('bare_cost') #switching to use bare_cost
        speed_method, duration_1q = get_method_duration(group_name) #but still use the custom group name to pass these parameters

        # minimize score over target operations or provided metric
        winner = None
    
        for v in tqdm(g.values()) if tqdm_bool else g.values():

            base_gate = ConversionGainGate(*v[0])
            template = MixedOrderBasisCircuitTemplate(base_gates=[base_gate], chatty_build=0, bare_cost=True)
            candidate_score = 0
            scaled_gate = None # used for skipping reconstruction in the atomic cost scaling function

            if metric in [0,1,2] and target_ops is None:
                target_score = v[1][metric] 
                scaled_gate, scaled_score = atomic_cost_scaling(params=v[0], scores=target_score, speed_method=speed_method, duration_1q=duration_1q, scaled_gate=scaled_gate)
                candidate_score = scaled_score

            else:
                for target in target_ops:
                        target_score = monodromy_range_from_target(template, target_u = target.to_matrix())[0] 
                        scaled_gate, scaled_score = atomic_cost_scaling(params=v[0], scores=target_score, speed_method=speed_method, duration_1q=duration_1q, scaled_gate=scaled_gate)
                        candidate_score += scaled_score
            
=======

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
def pick_winner(group_name, metric=0, target_ops=None, tqdm_bool=True, plot=True, smush_bool=False):
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
 
            candidate_score = 0
            scaled_gate = None  # used for skipping reconstruction in the atomic cost scaling function

            if metric in [0, 1, 2] and target_ops is None:
                if template.scores is not None:
                    target_score = template.scores
                else:
                    target_score = v[1][metric]
                scaled_gate, scaled_score = atomic_cost_scaling(
                    params=v[0],
                    scores=target_score,
                    speed_method=speed_method,
                    duration_1q=duration_1q,
                    scaled_gate=scaled_gate,
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
                    )
                    candidate_score += scaled_score

>>>>>>> master
            if winner is None or candidate_score < winner_score:
                winner = v
                winner_score = candidate_score
                winner_scaled_gate = scaled_gate
                winner_scaled_score = scaled_score

<<<<<<< HEAD

        winner_gate = ConversionGainGate(*winner[0]) #0 is params, 1 is scores
        # log winner params, scores, cost
        logging.info(f'winner: {winner_gate}, scores: {winner[1][:-2]}, cost: {winner_gate.cost()}')
        if not (metric in [0,1,2] and target_ops is None):
            #log weigted score and normalized score
            logging.info(f"winner score: {winner_score}, normalized score: {winner_score/len(target_ops)}")
        logging.info(f'scaled scores: {winner_scaled_score}, scaled cost: {winner_scaled_gate.cost()}')

        if plot:
            unitary_to_weyl(winner_gate.to_matrix()); #uncomment to see the gate in weyl space
        return winner_gate, winner_scaled_gate

=======
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


>>>>>>> master
# # %%
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    unitary_list, coordinate_list = build_gates()
<<<<<<< HEAD
    collect_data(unitary_list, overwrite=0)  
=======
    collect_data(unitary_list, overwrite=0)
>>>>>>> master
