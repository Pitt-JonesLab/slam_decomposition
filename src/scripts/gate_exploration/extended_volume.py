import logging
from typing import no_type_check

logger = logging.getLogger()
logging.basicConfig(filename="gg.log", level=logging.INFO)


import sys

sys.path.append("../../../")

import pickle
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as ss
from src.basis import MixedOrderBasisCircuitTemplate
from monodromy.backend.lrs import LRSBackend
from monodromy.coordinates import (monodromy_to_positive_canonical_coordinate,
                                   monodromy_to_positive_canonical_polytope,
                                   positive_canonical_to_monodromy_coordinate,
                                   unitary_to_monodromy_coordinate)
from monodromy.coverage import CircuitPolytope, deduce_qlr_consequences
from monodromy.haar import distance_polynomial_integrals
from monodromy.static.examples import (everything_polytope, exactly,
                                       identity_polytope)
from qiskit.circuit.library import (CPhaseGate, CXGate, IGate, SwapGate,
                                    iSwapGate)
from qiskit.quantum_info import Operator
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm
from weylchamber import WeylChamber, c1c2c3

from src.basisv2 import CircuitTemplateV2
from src.cost_function import SquareCost
from src.optimizer import TemplateOptimizer
from src.sampler import GateSample
from src.utils.custom_gates import (BerkeleyGate, ConversionGainGate,
                                    ConversionGainSmushGate)
from src.utils.visualize import _plot_circuit_polytope as debug_plot
from src.utils.visualize import coordinate_2dlist_weyl, unitary_2dlist_weyl

fpath = "/home/evm9/decomposition_EM/images"


# super hacky because haar_volume wasn't working
# XXX this works because function only uses convex_subpolytopes attribute which we use
def haar_volume(polytope):
    # circuit_poly = get_circuit_polytope(*[iSwapGate().power(1/2)]*k)
    return list(distance_polynomial_integrals([polytope]).values())[0][0]


def get_circuit_polytope(*basis_gate):
    circuit_polytope = identity_polytope
    for gate in basis_gate:
        b_polytope = exactly(
            *(
                Fraction(x).limit_denominator(10_000)
                for x in unitary_to_monodromy_coordinate(gate.to_matrix())[:-1]
            )
        )
        circuit_polytope = deduce_qlr_consequences(
            target="c",
            a_polytope=circuit_polytope,
            b_polytope=b_polytope,
            c_polytope=everything_polytope,
        )
    # return circuit_polytope

    return CircuitPolytope(
        operations="gate",
        cost=1,
        convex_subpolytopes=circuit_polytope.convex_subpolytopes,
    )


duration_1q = 0.25
N = 50
if __name__ == "__main__":
    # NOTE iters should be k when polytope is everything_polytope
    # gc, gg, t, str, iters
    iswap = np.pi / 2, 0, 1, "iSwap", 3
    sqiswap = np.pi / 2, 0, 1 / 2, "sqiSwap", 3
    cnot = np.pi / 4, np.pi / 4, 1, "CNOT", 3
    sqcnot = np.pi / 4, np.pi / 4, 1 / 2, "sqCNOT", 6
    b = 3 * np.pi / 8, np.pi / 8, 1, "B", 2
    sqb = 3 * np.pi / 8, np.pi / 8, 1 / 2, "sqB", 4
    gate_list = [iswap]  # , sqiswap, cnot, sqcnot, b, sqb]
    no_save = 1

    results = {}
    for gate in gate_list:
        gc, gg, t, gate_str, iters = gate
        logging.info("=========================================")
        logging.info(f"Gate: {gate_str}")
        gate_dict = {}
        unitary_list = (
            []
        )  # NOTE don't reset unitary list if want to stack between each K
        base_vol = None
        coverage_set = [identity_polytope] # needs to start like this to match monodromy formatting
        cnot_score = None
        swap_score = None
        haar_score = 0
        running_vol = 0


        # try loading from previous runs the coverage set
        MixedOrderBasisCircuitTemplate()


        for k in range(1, iters + 1):

            logging.info(f"K = {k}")
            
            # if k is at end, set it to full coverage and skip the rest
            # if full coverage, parallel drive of course can't extend volume
            if k == iters:
                base_vol, extended_vol = 1, 1
                haar_score += k * (extended_vol - running_vol)
                running_vol += extended_vol
                logging.info(f"Extended {extended_vol}")
                coverage_set.append(everything_polytope)
                gate_dict[str(k)] = [1, 1, 1, 1, 1]
                break

            # Setting up the template
            p_expand = [
                2,
                round(t / duration_1q),
                round(t / duration_1q),
            ]  # NOTE first variable is tracking an offset)
            pp2 = lambda *vargs: ConversionGainSmushGate(
                vargs[0],
                vargs[1],
                gc,
                gg,
                vargs[2 : 2 + round(t / duration_1q)],
                vargs[2 + round(t / duration_1q) :],
                t_el=t,
            )
            basis = CircuitTemplateV2(
                n_qubits=2,
                base_gates=[pp2],
                no_exterior_1q=1,
                vz_only=0,
                param_vec_expand=p_expand,
            )
            basis.build(k)
            # 1Q gate constraints
            # bounds_1q = 4 * np.pi  # arbitrary
            # for el in basis.circuit.parameters:
            #     s_el = str(el)
            #     if "Q" in s_el:
            #         basis.add_bound(s_el, bounds_1q, 0)
            # basis.circuit.draw()

            # Extending points via randomization
            progress = tqdm(range(N))
            for i in progress:
                params = basis.parameter_guess()
                qc_init = basis.assign_Xk(params)
                unitary_list.append(Operator(qc_init).data)
                progress.update()
                progress.refresh()
                if i % 1000 == 0:
                    # print(i/N)
                    pass
            # fig = unitary_2dlist_weyl(unitary_list, c='red', no_bar=1)
            # convert extended points to monodromy coordinates in order to work with LRS
            coordinate_list = list(map(lambda x: np.array(c1c2c3(x)), unitary_list))


            # second, we want to extend the points AGAIN using optimizer
            # the idea is that if CX or SWAP are far away, the randomizer won't be able to find them
            # a simple idea is to train to each of the vertics of the weyl chamber
            # every point we hit along the way is a new point that is added to the extended points
            # NOTE the template will use exterior 1Q gates such that can use SquareCost rather than coordinate optimizer
            
            for target_vertex in [CPhaseGate(theta=0), CXGate(), SwapGate(), iSwapGate()]:
                varg_offset = 0 #set to 4 if want to use phase, and change 0s to vargs in pp2 constructor below
                pp2 =lambda *vargs: ConversionGainSmushGate(0,0 , gc, gg, vargs[varg_offset:varg_offset+round(t/duration_1q)], vargs[varg_offset+round(t/duration_1q):], t_el=t)
                basis = CircuitTemplateV2(n_qubits=2, base_gates = [pp2], edge_params=[[(0,1)]], vz_only=False, param_vec_expand=[varg_offset,round(t/duration_1q),round(t/duration_1q)])
                basis.build(k)
                basis.spanning_range = range(k, k+1)
                sampler = GateSample(gate = target_vertex)
                s = [s for s in sampler][0]
                optimizer3 = TemplateOptimizer(basis=basis, objective=SquareCost(), use_callback=True, override_fail=True, success_threshold = 1e-10, training_restarts=3)
                ret3 = optimizer3.approximate_from_distribution(sampler) 
                coordinate_list += ret3[1][0] #get coordinate list


            # third, extend points via symmetry
            # TODO, take every coordinate and add its conjugate reflection to the unitary_list
            # the effect should be 
            for coordinate in coordinate_list:
                x,y,z = coordinate
                if x != 0.5:
                    coordinate_list.append(np.array([1-x, y, z]))

            # convert coordinates to monodromy
            coordinate_list = list(
                map(lambda x: np.array(x) * np.pi / 2, coordinate_list)
            )  # weylchamber package normalization
            coordinate_list = list(
                map(
                    lambda x: positive_canonical_to_monodromy_coordinate(*x),
                    coordinate_list,
                )
            )

            # save fig as svg and pdf
            color = ["black", "red", "blue"][(k - 1) % 3]
            fig = unitary_2dlist_weyl(unitary_list, c=color, no_bar=1)
            # fig.show()

            if not no_save:
                name = f"smush_{k}_k_{t}_t_{gc}_gc_{gg}_gg"
                fig.savefig(f"{fpath}/extended_{gate_str}_k{k}.svg", format="svg")
                fig.savefig(f"{fpath}/extended_{gate_str}_k{k}.pdf", format="pdf")

            """Solve for volumes"""
            # first, create the base case volume
            circuit_poly = get_circuit_polytope(
                *([ConversionGainGate(0, 0, gc, gg, t)] * k)
            )
            base_vol = haar_volume(circuit_poly)
            logging.info(f"Base {base_vol}")

            # then, convert coordinates to fractions
            convert_frac = lambda x: [
                Fraction(xi).limit_denominator(10_000) for xi in x
            ]
            extended_points = list(map(convert_frac, coordinate_list))

            # finally, create the extended polytope by appending convex hull to base case
            convex_polytope = LRSBackend.convex_hull(extended_points)
            # circuit_poly.convex_subpolytopes.append(convex_polytope) #XXX hacky
            extened_poly_list = circuit_poly.convex_subpolytopes + [convex_polytope]
            # to be less hacky just reconstruct
            circuit_poly = CircuitPolytope(
                convex_subpolytopes=extened_poly_list,
                cost=k,
                operations=gate_str,
            )
            # Haar calcs
            extended_vol = haar_volume(circuit_poly)
            haar_score += k * (extended_vol - running_vol)
            running_vol += extended_vol
            logging.info(f"Extended {extended_vol}")

            # check for CNOT and SWAP
            target_u = CXGate().to_matrix()
            target_coords = unitary_to_monodromy_coordinate(target_u)
            cnot_bool = circuit_poly.has_element(target_coords)
            if cnot_score is None and cnot_bool:
                cnot_score = k

            target_u = SwapGate().to_matrix()
            target_coords = unitary_to_monodromy_coordinate(target_u)
            swap_bool = circuit_poly.has_element(target_coords)
            if swap_score is None and swap_bool:
                swap_score = k

            # checking for B because may help improving SWAP decomp
            target_u = BerkeleyGate().to_matrix()
            target_coords = unitary_to_monodromy_coordinate(target_u)
            b_bool = circuit_poly.has_element(target_coords)

            logging.info(f"D[CNOT] {cnot_bool}")
            logging.info(f"D[SWAP] {swap_bool}")
            logging.info(f"D[B] {b_bool}")
            logging.info(f"Volume: {extended_vol}")  # idk what circuit_poly.volume does

            coverage_set.append(circuit_poly)

            # debug_plot(circuit_poly)
            gate_dict[str(k)] = [base_vol, extended_vol, cnot_bool, swap_bool, b_bool]

        results[gate_str] = gate_dict
        logging.info(f"Haar Score: {haar_score}")

        # save coverage set
        # matching syntax from mixedbasistemplate, need to create a gate hash for reconstruction
        if not no_save:
            
            gate = ConversionGainGate(0, 0, gc, gg, t)
            gate.normalize_duration(1)
            gate_hash = {}
            gate_hash[str(gate)] = gate
            # seems like from monodromy.coverage all we need is the list of circuit polytopes
            logging.warning(
                "Assuming smush template is irreundant, see monodromy.coverage.build_coverage_set for more details"
            )
            file_hash = (
                str([str(gate)]) + "smush"
            )  # wrap in list for consistency with mixedbasistemplate
            print(file_hash)
            filepath = f"/home/evm9/decomposition_EM/data/polytopes/polytope_coverage_{file_hash}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump((coverage_set, gate_hash, [haar_score, cnot_score, swap_score]), f)
                logging.debug("saved polytope coverage to file")

    # save results
    if not no_save:
        import json

        with open(f"extended_results.json", "w") as fp:
            json.dump(results, fp)
