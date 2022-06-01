# %%
from custom_gates import *
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from itertools import cycle
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info import Operator

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TemplateCircuit:
    def __init__(
        self,
        n_qubits=2,
        base_gate_class=[RiSwapGate],
        gate_2q_params=[1 / 2],
        edge_params=[(0, 1)],
        trotter=False,
        no_exterior_1q=False,
    ):
        """Initalizes a qiskit.quantumCircuit object with unbound 1Q gate parameters
        Args:
            n_qubits: size of target unitary,
            base_gate_class: Gate class of 2Q gate,
            gate_2q_params: List of params to define template gate cycle sequence
            edge_params: List of edges to define topology cycle sequence
            trotter: if true, only use gate_2q_params[0], override cycle length and edge_params, each 1Q gate share parameters per qubit row
        """
        self.hash = (
            str(n_qubits)
            + str(base_gate_class)
            + str(gate_2q_params)
            + str(edge_params)
            + str(trotter)
            + str(no_exterior_1q)
        )

        if n_qubits != 2 and trotter:
            raise NotImplementedError
        self.n_qubits = n_qubits
        self.trotter = trotter
        self.circuit = QuantumCircuit(n_qubits)
        self.gate_2q_base = base_gate_class
        self.no_exterior_1q = no_exterior_1q

        self.cycles = 0

        if self.trotter:
            # raise NotImplementedError
            logging.warning("Trotter may not work as intended")

        # else:
        self.gate_2q_base = cycle(base_gate_class)
        self.gate_2q_params = cycle(gate_2q_params)
        self.gate_2q_edges = cycle(edge_params)
        self.cycle_length = max(len(gate_2q_params), len(edge_params))

        self.gen_1q_params = self._param_iter()

    # def __str__(self):
    #     s = ""
    #     for param in self.gate_2q_params:
    #         s += self.gate_2q_base.latex_string(param)
    #     return s

    def build(self, n_repetitions):
        self._reset()

        if n_repetitions <= 0:
            return

        if self.trotter:
            pass
            # n_repetitions = int(1 / next(self.gate_2q_params))
        for i in range(n_repetitions):
            self._build_cycle(initial=(i == 0), final=(i == n_repetitions - 1))

    def _reset(self):
        """Return template to a 0 cycle"""
        self.cycles = 0
        self.circuit = QuantumCircuit(self.n_qubits)

    def initial_guess(self):
        """returns a np array of random values for each parameter"""
        return np.random.random(len(self.circuit.parameters)) * 2 * np.pi

    def assign_Xk(self, Xk):

        return self.circuit.assign_parameters(
            {parameter: i for parameter, i in zip(self.circuit.parameters, Xk)}
        )

    def eval(self, Xk):
        """returns an Operator after binding parameter array to template"""
        return Operator(self.assign_Xk(Xk)).data

    def _param_iter(self):
        index = 0
        while True:
            # Check if Parameter already created, then return reference to that variable
            def _filter_param(param):
                return param.name == f"P{index}"

            res = list(filter(_filter_param, self.circuit.parameters))
            if len(res) == 0:
                yield Parameter(f"P{index}")
            else:
                yield res[0]
            index += 1
            if self.trotter:
                index %= 3 * self.n_qubits

    def _build_cycle(self, initial=False, final=False):
        """Extends tempalte by one full cycle"""
        if initial and not self.no_exterior_1q:
            # before build by extend, add first pair of 1Qs
            for qubit in range(self.n_qubits):
                self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
        for _ in range(self.cycle_length):
            edge = next(self.gate_2q_edges)
            self.circuit.append(
                next(self.gate_2q_base)(next(self.gate_2q_params)), edge
            )
            if not (final and self.no_exterior_1q):
                for qubit in edge:
                    self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
        self.cycles += 1


# %%
# a = TemplateCircuit(
#     gate_2q_params=[1 /3, 1 / 3], n_qubits=2, edge_params=[(0, 1), (0, 2), (1, 2)], trotter=True
# )
# a.build(2)
# a.circuit.draw(output="mpl")
