import logging
logger = logging.getLogger()

import numpy as np
from slam.utils.gates.custom_gates import ConversionGainGate
from slam.utils.polytopes.polytope_wrap import monodromy_range_from_target, coverage_to_haar_expectation
from slam.basis import MixedOrderBasisCircuitTemplate
from slam.utils.gates.custom_gates import ConversionGainGate
from qiskit.circuit.library import CXGate, SwapGate
from weylchamber import c1c2c3
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import h5py
from config import srcpath
filename = f"{srcpath}/data/cg_gates.h5"

# # verifying that relative phase doesn't change 2Q gate location
# unitary = [ConversionGainGate(0, 0, p*0.5*np.pi, (1-p)*0.4*np.pi) for p in np.linspace(0,1,16)]
# print([u.cost() for u in unitary])
# unitary_to_weyl(*unitary);

# plotting to make sure getting good weyl coverage before running data collection
"""Here we are building the set of gates to collect data over
The math here is translating gates to left side of x-axis to remove duplicates"""


def get_group_name(speed_method="linear", duration_1q=0):
    return f"{speed_method}_scaling_1q{duration_1q}"


def get_method_duration(group_name):
    """returns the speed_method and duration_1q from the group name"""
    speed_method = group_name.split("_")[0]
    duration_1q = float(group_name.split("_")[-1].replace("1q", ""))
    return speed_method, duration_1q


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


# unitary_list, coordinate_list = build_gates()
# coordinate_2dlist_weyl(*coordinate_list);

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

            # FIXME adding a None to end of score list makes so not a jagged 2d array - can fix better later
            # XXX is a problem if changing size of base_gate.params
            g.create_dataset(
                str(base_gate),
                data=np.array(
                    [base_gate.params, [haar_score, cnot_score, swap_score, -1, -1]]
                ),
            )

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

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    unitary_list, coordinate_list = build_gates()
    print(len(unitary_list))
    collect_data(unitary_list, overwrite=0)
