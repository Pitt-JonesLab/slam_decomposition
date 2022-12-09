# %%
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import numpy as np
from slam.utils.visualize import plotMatrix

import matplotlib.pyplot as plt
# %matplotlib widget
from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import mutual_information, Statevector, partial_trace, concurrence, entanglement_of_formation
from slam.basisv2 import CircuitTemplateV2
from slam.utils.gates.custom_gates import CirculatorSNAILGate
from slam.cost_function import BasicCostInverse, BasicCost, BasicReducedCost
from slam.optimizer import TemplateOptimizer
import h5py

# %%
# use the ideas from smush volume to find a gaet taht can include sqiswap in its extended primative set

from slam.hamiltonian import ConversionGainPhaseHamiltonian
from slam.basisv2 import CircuitTemplateV2
from slam.utils.gates.custom_gates import ConversionGainGate
pp =lambda p1, p2: ConversionGainGate(p1, p2, np.pi/4, np.pi/4)
basis = CircuitTemplateV2(n_qubits=2, base_gates=[pp], no_exterior_1q=0, vz_only=1)
basis.build(1)


# %%
t = 1
duration_1q = .1
# gc = np.pi/2
# gg = 0*np.pi/4

# %%
from slam.utils.gates.custom_gates import ConversionGainSmushGate

# NOTE first variable is tracking an offset (basically set it to 2 if counting the phase variables)
offset = 4 # 2 if only phase, 4 if c and g
p_expand = [offset, round(t/duration_1q), round(t/duration_1q)]
# XXX turn p_expand into indices is tricky

pp2 =lambda *vargs: ConversionGainSmushGate(vargs[0], vargs[1], vargs[2], vargs[3], vargs[offset:offset+round(t/duration_1q)], vargs[offset+round(t/duration_1q):], t_el=t)
#pp2 =lambda *vargs: ConversionGainSmushGate(0, 0, np.pi/2, 0, vargs[:round(t/du}ration_1q)], vargs[round(t/duration_1q):], t_el=t)

# circuittemplate builds # of parameters by checking the number of parameters in the lambda function to build the gate
# because the parametes for gx and gy are vectors they only get counted once and it messes up
# we can add an extra parameter called param_vec_expand
# we need to use this to tell it to expand the number of parameters we should include
# however, this will get really messy because we don't know which parameters are the vectors or not
# be careful, this is going to be a mess :(
basis = CircuitTemplateV2(n_qubits=2, base_gates=[pp2], no_exterior_1q=1, vz_only=0, param_vec_expand = p_expand)
basis.spanning_range = range(1,2)

basis.build(1)
#adding constraint of 1Q params need to be positive valued
#this could be optional let's compare what happens if include it or not
for el in basis.circuit.parameters:
    s_el = str(el)
    if 'Q' in s_el:
        basis.add_bound(s_el, 2*np.pi, -2*np.pi)
#basis.add_bound("Q0", 4*np.pi, 0)
basis.circuit.draw()

# %%
# from slam.sampler import HaarSample
# sampler = HaarSample(seed=0,n_samples=1)
# s = [s for s in sampler][0]
from slam.utils.gates.custom_gates import RiSwapGate, BerkeleyGate
from slam.sampler import HaarSample, GateSample
#sampler = GateSample(gate = RiSwapGate(1/2))
sampler = GateSample(gate = BerkeleyGate())
s = [s for s in sampler][0]

# %%
# Here we want to save the best cost as a function of the success threshold
objective1 = BasicCost()
# keep trying until success_threshold converges:
i=0
LB = 0
UB = 0.75
current_cost = (UB - LB)/2

# keep trying until success_threshold converges:
while i == 0 or np.abs(current_cost - previous_cost) > 0.0001:
    if not current_cost is None:
        basis.set_constraint(param_max_cost=current_cost)
        #pass
    #rebuild optimizer to refresh the updated f_basis obj
    #NOTE setting the success threshold low since SWAP is very hard to find exactly
    optimizer3 = TemplateOptimizer(basis=basis, objective=objective1, use_callback=False, override_fail=True, success_threshold = 1e-8, training_restarts=50)

    _ret3 = optimizer3.approximate_target_U(s)
    current_cost = basis.circuit_cost(_ret3.Xk)
    print(f"Iteration:{i}, Decomposition Result:{_ret3.loss_result}, Cost:{current_cost},")

    #search using 2 steps forward, 1 step back approach BAD
    # I want to do a binary search between 0 and 1.5 
    # if success, then set next cost to be current - LB / 2
    # if fail, then set next cost to be current + (UB - current) / 2
    
    if _ret3.success_label:
        ret3 = _ret3
        #fidelities[k] = ret3

        #success means can tighten the constraint
        previous_cost = current_cost
        current_cost = (current_cost - LB)/2
        UB = previous_cost
        print("Success, new cost:", current_cost)

        # save best cost as we find it 
        with h5py.File(f'sqiswap_fidelity.h5', 'w') as hf:
            #save ret3
            hf.create_dataset('loss', data=[ret3.loss_result])
            hf.create_dataset('Xk', data=ret3.Xk)
            hf.create_dataset('cost', data=[current_cost])
            hf.create_dataset('cycles', data=[ret3.cycles])

    else:
        #fail means loosen the constraint
        previous_cost = current_cost
        current_cost = current_cost + (UB - current_cost)/2
        LB = previous_cost
        print("Fail, new cost:", current_cost)

    i+=1

# # %%
# # load best from h5
# with h5py.File(f'sqiswap_fidelity.h5', 'r') as hf:
#     #save ret3
#     loss = hf['loss'][0]
#     Xk = hf['Xk'][:]
#     cycles = hf['cycles'][0]
#     cost = hf['cost'][0]

# # build the circuit
# print(cost, loss)
# #show the result of training
# basis.build(cycles)
# circuit =basis.assign_Xk(Xk)
# circuit.draw()


