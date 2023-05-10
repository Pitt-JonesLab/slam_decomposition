import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from weylchamber import c1c2c3

from slam.basisv2 import CircuitTemplateV2
from slam.utils.gates.custom_gates import ConversionGainSmush1QPhaseGate
from slam.utils.visualize import coordinate_2dlist_weyl, update_coordinate_2dlist_weyl


class ParallelDrivenGateWidget:
    def __init__(
        self,
        N=10,
        gc=np.pi / 2,
        gg=0,
        gz1=0,
        gz2=0,
        phase_a=0,
        phase_b=0,
        phase_c=0,
        phase_g=0,
    ) -> None:
        self.N = N
        self.gc = gc
        self.gg = gg
        self.gz1 = gz1
        self.gz2 = gz2
        self.phase_a = phase_a
        self.phase_b = phase_b
        self.phase_c = phase_c
        self.phase_g = phase_g
        self.t = 0.1
        self.timesteps = 1
        self.duration_1q = self.t / self.timesteps
        self.construct_basis()
        self.prepare_parameters(0, 0)
        self.fig = None

    def __add__(self, other):
        if self.qc is None:
            self.construct_basis()
        if other.qc is None:
            other.construct_basis()
        # need to rename the parameters to avoid conflicts
        for p in self.qc.parameters:
            self.qc = self.qc.assign_parameters({p: Parameter(str(p) + "_")})

        ret = ParallelDrivenGateWidget()
        ret.qc = self.qc.compose(other.qc)
        ret.N = self.N + other.N
        return ret

    def construct_basis(self):
        varg_offset = 0

        # ConversionGainSmushGate
        def pp2(*vargs):
            return ConversionGainSmush1QPhaseGate(
                self.phase_a,
                self.phase_b,
                self.phase_c,
                self.phase_g,
                self.gc,
                self.gg,
                self.gz1,
                self.gz2,
                vargs[varg_offset : varg_offset + round(self.t / self.duration_1q)],
                vargs[varg_offset + round(self.t / self.duration_1q) :],
                t_el=self.t,
            )

        self.basis = CircuitTemplateV2(
            n_qubits=2,
            base_gates=[pp2],
            edge_params=[[(0, 1)]],
            no_exterior_1q=True,
            param_vec_expand=[
                varg_offset,
                round(self.t / self.duration_1q),
                round(self.t / self.duration_1q),
            ],
        )
        self.basis.build(1)
        # basis.circuit.draw(output='mpl');

        # repeat the atomic pd gate multiple times
        qc = QuantumCircuit(2)
        for _ in range(self.N):
            qc = qc.compose(self.basis.circuit)
        self.qc = qc

    def plot(self):
        self.fig = coordinate_2dlist_weyl(*self.coordinate_list)
        plt.show()

    def widget_wrap(self, q0, q1, pa, pb, pc, pg, gz1, gz2):
        self.gz1 = gz1
        self.gz2 = gz2
        self.phase_a = pa
        self.phase_b = pb
        self.phase_c = pc
        self.phase_g = pg
        self.construct_basis()
        self.prepare_parameters(q0, q1)
        self.iterate_time()
        if self.fig is not None:
            self.fig = update_coordinate_2dlist_weyl(self.fig, *self.coordinate_list)
            # self.fig.show()
        else:
            self.fig = coordinate_2dlist_weyl(*self.coordinate_list)
            plt.show()

    def widget_wrap_different_start(self, q0, q1):
        self.prepare_parameters(q0, q1)
        self.iterate_time_different_start(R=25)
        if self.fig is not None:
            self.fig = update_coordinate_2dlist_weyl(self.fig, *self.coordinate_list)
            # self.fig.show()
        else:
            self.fig = coordinate_2dlist_weyl(*self.coordinate_list)
            plt.show()

    def widget_wrap_nonuniform(self, g0_vector, g1_vector):
        self.prepare_parameters_nonuniform(g0_vector, g1_vector)
        self.iterate_time()
        if self.fig is not None:
            self.fig = update_coordinate_2dlist_weyl(self.fig, *self.coordinate_list)
            # self.fig.show()
        else:
            self.fig = coordinate_2dlist_weyl(*self.coordinate_list)
            plt.show()

    def prepare_parameters(self, q0, q1):
        i = 0
        out = self.qc.copy()
        for instr, qargs, cargs in out:
            if instr.params and instr.name == "2QSmushGate":
                instr.params[4:6] = [q0, q1]
                instr.params[-1] = Parameter(f"t{i}")
                i += 1
            elif instr.params and instr.name == "2QSmushGate1QPhase":
                instr.params[8:10] = [q0, q1]
                instr.params[-1] = Parameter(f"t{i}")
                i += 1
        self.prep_qc = out

    def prepare_parameters_nonuniform(self, g0_vector, g1_vector):
        assert len(g0_vector) == len(g1_vector) == self.N
        i = 0
        out = self.qc.copy()
        for instr, qargs, cargs in out:
            if instr.params and instr.name == "2QSmushGate":
                instr.params[4:6] = [g0_vector[i], g1_vector[i]]
                instr.params[-1] = Parameter(f"t{i}")
                i += 1
            elif instr.params and instr.name == "2QSmushGate1QPhase":
                instr.params[8:10] = [g0_vector[i], g1_vector[i]]
                instr.params[-1] = Parameter(f"t{i}")
                i += 1
        self.prep_qc = out

    def solve_end(self):
        # bind each time to full duration and evalute
        qc = QuantumCircuit(2)
        # copying the circuit manually this way moves t parameters into the parameterview object
        for gate in self.prep_qc:
            qc.append(gate[0], gate[1])
        for i in range(self.N):
            qc = qc.bind_parameters({qc[i][0].params[-1]: self.duration_1q})
        return Operator(qc).data

    def iterate_time(self, R=5):
        # R = 5 # resolution
        endpoints = range(1, self.N + 1)
        coordinate_list = []
        end_segment_list = []

        for end in endpoints:
            temp_coords = []
            qc = QuantumCircuit(2)
            for gate in self.prep_qc[0:end]:
                qc.append(gate[0], gate[1])

            qc2 = qc.copy()
            # for all prior 2Q gates, set time parameter to full length
            for i in [el for el in endpoints if el < end]:
                qc2 = qc2.bind_parameters({qc2[i - 1][0].params[-1]: self.duration_1q})
            # for current 2Q gate, iterate over time and append coordinate
            for t in np.linspace(0, self.duration_1q, R):
                qc3 = qc2.bind_parameters({qc2[end - 1][0].params[-1]: t})
                # eliminating x-axis symmetry
                c = list(c1c2c3(Operator(qc3).data))
                if c[0] > 0.5:
                    c[0] = -1 * c[0] + 1
                temp_coords.append(c)
            coordinate_list.append(temp_coords)
            end_segment_list.append(c)
        self.coordinate_list = coordinate_list
        self.end_segment_list = end_segment_list
        self.final_unitary = Operator(qc3).data
        # qc2.draw(output='mpl');

    def iterate_time_different_start(self, R=5):
        # R = 5 # resolution
        endpoints = range(1, self.N + 1)
        coordinate_list = []
        end_segment_list = []

        for end_iter, end in enumerate(endpoints):
            temp_coords = []
            qc = QuantumCircuit(2)
            # for gate in self.prep_qc[0:end]:
            #     qc.append(gate[0], gate[1])
            qc.rzx(np.pi / 5 * end_iter, 0, 1)

            # append the self.prep_qc
            for gate in self.prep_qc[end_iter : end_iter + 1]:
                qc.append(gate[0], gate[1])

            qc2 = qc.copy()
            # # for all prior 2Q gates, set time parameter to full length
            # for i in [el for el in endpoints if el < end]:
            #     qc2 = qc2.bind_parameters({qc2[i-1][0].params[-1] : self.duration_1q})
            # for current 2Q gate, iterate over time and append coordinate
            for t in np.linspace(0, self.duration_1q, R):
                qc3 = qc2.bind_parameters({qc2[1][0].params[-1]: t})
                # eliminating x-axis symmetry
                c = list(c1c2c3(Operator(qc3).data))
                if c[0] > 0.5:
                    c[0] = -1 * c[0] + 1
                temp_coords.append(c)
            coordinate_list.append(temp_coords)
            end_segment_list.append(c)
        self.coordinate_list = coordinate_list
        self.end_segment_list = end_segment_list
        self.final_unitary = Operator(qc3).data
        # qc2.draw(output='mpl');


class ImprovedCX(ParallelDrivenGateWidget):
    def __init__(self):
        super().__init__()
        self.prepare_parameters_nonuniform([3] * self.N, [0] * self.N)
        self.iterate_time()
        R = 5
        # append 1Q at (0,0,0) and (0.5, 0 ,0)
        self.coordinate_list.append([[0, 0, 0]] * R)
        self.coordinate_list.append([(0.5, 0, 0)] * R)

        # want to plot comparing the nonoptimized version
        # spline between (0,0,0), (.25, .25, 0), (.5, 0, 0)
        baseline_coords = []

        # interpolate endpoints inclusive
        i_steps = 25  # hardcoded but this is R * t/duration_1q
        baseline_coords.append(
            [
                [0.25 / (i_steps - 1) * i, 0.25 / (i_steps - 1) * i, 0]
                for i in range(i_steps)
            ]
        )

        # interpolate 5 points between (.25 .25 0) and (.5, 0, 0)
        baseline_coords.append(
            [
                [0.25 + 0.25 / (i_steps - 1) * i, 0.25 - 0.25 / (i_steps - 1) * i, 0]
                for i in range(i_steps)
            ]
        )

        baseline_coords.append([[0, 0, 0]] * R)
        baseline_coords.append([[0.25, 0.25, 0]] * R)
        baseline_coords.append([[0.5, 0, 0]] * R)

        self.baseline_coords = baseline_coords

        # self.fig = coordinate_2dlist_weyl(*baseline_coords, c='cyan', no_bar=1, fig=self.fig);
        # self.fig = coordinate_2dlist_weyl(*self.coordinate_list, c='red', no_bar=1);
        # plt.show()


class ImprovedSWAP(ParallelDrivenGateWidget):
    def __init__(self):
        super().__init__()
        self.prepare_parameters_nonuniform([np.pi] * self.N, [np.pi] * self.N)
        self.iterate_time()

        R = 5
        # tack on the final sqiswap, found from decomp_trajectory.ipynb
        # we tried to optimize this away but could not perfectly do so yet
        # start with the previous circuit all binded
        extended_qc = self.prep_qc.copy()
        # for gate in extended_qc:
        #     gate[0].params[-1] = self.duration_1q

        # XXX ugly hardcoding because we are messing with parameters in an unsafe way
        # ie circuit parameter table is out of date
        for gate in range(self.N):
            extended_qc[gate][0].params[-1] = self.duration_1q

        # next build the remaining gates
        from qiskit.circuit.library import U3Gate

        extended_qc.append(
            U3Gate(7.84862563826406, 9.44285614361501, 2.30856826810552), [0]
        )
        extended_qc.append(
            U3Gate(7.85928560541358, 9.44027709402712, -3.9923157086907), [0]
        )
        for _ in np.linspace(0, R):
            c = list(c1c2c3(Operator(extended_qc).data))
            if c[0] > 0.5:
                c[0] = -1 * c[0] + 1
            self.coordinate_list.append(c)

        # for _ in range(self.N/2): # should be N=10, and N/2
        #     self.qc = self.qc.compose(self.basis.circuit)

        # now need to compute final part of trajectory
        # this is messy because in parent class, we assumed there were no interior 1Q gates
        # hardcode, lets just go back 6 timesteps from the end

        # finally
        # append 1Q at (0,0,0) and (0.5, 0.5 ,0.5)
        self.coordinate_list.append([[0, 0, 0]] * 5)
        self.coordinate_list.append([(0.5, 0.5, 0.5)] * 5)
        # self.coordinate_list = coordinate_list
        # self.end_segment_list = end_segment_list
        # self.final_unitary = Operator(qc3).data

        # self.fig = coordinate_2dlist_weyl(*self.coordinate_list);
        # plt.show()
