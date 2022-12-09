# %%
#this notebook is for testing when we smush 1Q gates into a 2Q gate (and changing its weyl coordinates)
# whether the new coordinates it gets access to gives the overall 2Q+smush additional volume had it not had otherwise

# %%
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import sys
sys.path.append("../../../")

import numpy as np

# %%
from src.hamiltonian import ConversionGainPhaseHamiltonian
from src.basisv2 import CircuitTemplateV2
from src.utils.custom_gates import ConversionGainGate
pp =lambda p1, p2: ConversionGainGate(p1, p2, np.pi/4, np.pi/4)
basis = CircuitTemplateV2(n_qubits=2, base_gates=[pp], no_exterior_1q=0, vz_only=1)
basis.build(1)

# %%
t = 1
duration_1q = 0.1
gc = 0*np.pi/2
gg = 0*np.pi

# %%
from src.utils.custom_gates import ConversionGainSmushGate

# NOTE first variable is tracking an offset (basically set it to 2 if counting the phase variables)
p_expand = [4, round(t/duration_1q), round(t/duration_1q)]
# XXX turn p_expand into indices is tricky

pp2 =lambda *vargs: ConversionGainSmushGate(vargs[0], vargs[1], vargs[2], vargs[3], vargs[4:4+round(t/duration_1q)], vargs[4+round(t/duration_1q):], t_el=t)
#pp2 =lambda *vargs: ConversionGainSmushGate(0, 0, np.pi/2, 0, vargs[:round(t/duration_1q)], vargs[round(t/duration_1q):], t_el=t)

# circuittemplate builds # of parameters by checking the number of parameters in the lambda function to build the gate
# because the parametes for gx and gy are vectors they only get counted once and it messes up
# we can add an extra parameter called param_vec_expand
# we need to use this to tell it to expand the number of parameters we should include
# however, this will get really messy because we don't know which parameters are the vectors or not
# be careful, this is going to be a mess :(
basis = CircuitTemplateV2(n_qubits=2, base_gates=[pp2], no_exterior_1q=1, vz_only=0, param_vec_expand = p_expand)
basis.build(1)
#adding constraint of 1Q params need to be positive valued
#this could be optional let's compare what happens if include it or not
for el in basis.circuit.parameters:
    s_el = str(el)
    if 'Q' in s_el:
        basis.add_bound(s_el, 4*np.pi, 0)

# %%
basis.circuit.draw()

# %%
from src.utils.visualize import unitary_2dlist_weyl, coordinate_2dlist_weyl
from qiskit.quantum_info import Operator
from weylchamber import c1c2c3
import h5py
#from tqdm.notebook import tqdm as tqdm
from tqdm import tqdm

filename = "/home/evm9/decomposition_EM/data/smush_unitary.h5"
#load from hdf5 file
with h5py.File(filename, 'w') as f:
    data = f.require_group('unitary_cost_data')
    if 'unitary_list' in data and 'cost_list' in data:
        unitary_list = list(data['unitary_list'][:])
        cost_list = list(data['cost_list'][:])
    else:
        unitary_list = []
        cost_list = []

#outer loop is sweeping over gc and gc
for m in tqdm(np.linspace(0, 0.5, 9)): #17
    if m ==0:
        continue #identity is not interesting case
    
    for n in np.linspace(0, 1, 13): #21
        for i in range(400):
            
            gc = n*m*np.pi
            gg = (1-n)*m*np.pi
            cost = ConversionGainGate(0,0, gc, gg, t_el=t).cost()

            #need to convert strings to Parameter objects
            #the reason need to do this is because need to get reference to the exact parameter object via iteration
            pstrs = ["Q2", "Q3"]
            pdict = {str(pstr):[p for p in basis.circuit.parameters if p.name == pstr][0] for pstr in pstrs}

            qc_init = basis.circuit.assign_parameters({pdict["Q2"]:gc})
            qc_init = qc_init.assign_parameters({pdict["Q3"]:gg})
            #randomize the remaining parameters
            #NOTE there isn't a 1-1 mapping from p to remaining parameters bc have replaced q2 and q3 already but doesnt matter since is random
            p = basis.parameter_guess()
            #qc_init = qc_init.assign_Xk(p)
            qc_init = qc_init.assign_parameters({k:v for k,v in zip(qc_init.parameters, p)})

            #eliminating x-axis symmetry
            c = list(c1c2c3(Operator(qc_init).data))
            if c[0] > 0.5:
                c[0] = -1*c[0] + 1

            # #when checking for duplicates, keep the lower cost one
            # #instead of checking for duplicates, check if already a coordinate close by
            # if any([np.linalg.norm(np.array(c)-np.array(c2)) < 0.025 for c2 in unitary_list]):
            #     c2 = unitary_list[np.argmin([np.linalg.norm(np.array(c)-np.array(c2)) for c2 in unitary_list])]
            #     ind = unitary_list.index(c2)
            #     if cost < cost_list[ind]:
            #         unitary_list[ind] = c
            #         cost_list[ind] = cost
            #         continue
            else:        
                unitary_list.append(c)
                cost_list.append(cost)

#do some preprocessing that takes the smaller cost value if their is a match of coordinates
#no not a good idea

# save unitary list into a hdf5 file
with h5py.File(filename, 'w') as f:
    f.create_dataset('unitary_cost_data/unitary_list', data=unitary_list)
    f.create_dataset('unitary_cost_data/cost_list', data=cost_list)