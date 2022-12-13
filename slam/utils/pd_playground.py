import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from weylchamber import c1c2c3
from slam.utils.visualize import coordinate_2dlist_weyl, update_coordinate_2dlist_weyl
from slam.basisv2 import CircuitTemplateV2
from slam.utils.gates.custom_gates import ConversionGainSmushGate


class ParallelDrivenGateWidget():
    def __init__(self, N=10, gc=np.pi/2, gg=0, phase_c=0, phase_g=0) -> None:
        self.N = N
        self.gc = gc
        self.gg = gg
        self.phase_c = phase_c
        self.phase_g = phase_g
        self.t= .1
        self.timesteps = 1
        self.duration_1q = self.t/self.timesteps
        self.construct_basis()
        self.prepare_parameters(0,0)
        self.fig = None

    def construct_basis(self):

        varg_offset = 0
        pp2 =lambda *vargs: ConversionGainSmushGate(self.phase_c,self.phase_g, self.gc, self.gg, vargs[varg_offset:varg_offset+round(self.t/self.duration_1q)], vargs[varg_offset+round(self.t/self.duration_1q):], t_el=self.t)
        basis = CircuitTemplateV2(n_qubits=2, base_gates = [pp2], edge_params=[[(0,1)]], no_exterior_1q=True, param_vec_expand=[varg_offset,round(self.t/self.duration_1q),round(self.t/self.duration_1q)])
        basis.build(1)
        # basis.circuit.draw(output='mpl');

        # repeat the atomic pd gate multiple times
        qc = QuantumCircuit(2)
        for _ in range(self.N):
            qc = qc.compose(basis.circuit)
        self.qc = qc

    def widget_wrap(self, q0, q1):
        self.prepare_parameters(q0, q1)
        self.iterate_time()
        if self.fig is not None:
            self.fig = update_coordinate_2dlist_weyl(self.fig, *self.coordinate_list)
            self.fig.show()
        else:
            self.fig = coordinate_2dlist_weyl(*self.coordinate_list);
            self.fig.show()

    def prepare_parameters(self, q0, q1):
        i = 0
        out = self.qc.copy()
        for instr, qargs, cargs in out:
            if instr.params and instr.name =="2QSmushGate":
                instr.params[4:6] = [q0, q1]
                instr.params[-1] = Parameter(f't{i}')
                i +=1
        self.prep_qc = out
    
    def iterate_time(self):
        R = 5 # resolution
        endpoints = range(1, self.N+1)
        coordinate_list = []

        for end in endpoints:
            temp_coords = [] 
            qc = QuantumCircuit(2)
            for gate in self.prep_qc[0:end]:
                qc.append(gate[0], gate[1])
            
            qc2 = qc.copy()
            # for all prior 2Q gates, set time parameter to full length
            for i in [el for el in endpoints if el < end]:
                qc2 = qc2.bind_parameters({qc2[i-1][0].params[-1] : self.duration_1q} )
            # for current 2Q gate, iterate over time and append coordinate
            for t in np.linspace(0, self.duration_1q, R):
                qc3 = qc2.bind_parameters({qc2[end-1][0].params[-1]: t})
                #eliminating x-axis symmetry
                c = list(c1c2c3(Operator(qc3).data))
                if c[0] > 0.5:
                    c[0] = -1*c[0] + 1
                temp_coords.append(c)
            coordinate_list.append(temp_coords)
        self.coordinate_list = coordinate_list
        # qc2.draw(output='mpl');
