import numpy as np
import weylchamber
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType

from src.hamiltonian import CirculatorHamiltonian

"""
Library of useful gates that aren't defined natively in qiskit

Example Usage:
from custom_gates import *
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.append(RiSwapGate(1/4), [0,1])
qc.draw(output='mpl')
"""


class VSwap(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("vswap", 3, [], "VSWAP")
        v_nn = np.sqrt(2) * np.pi / np.arccos(1 / np.sqrt(3))
        v_params = [np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0]

        self._array = CirculatorHamiltonian.construct_U(*v_params)

    def __array__(self, dtype=None):
        # v_nn = np.sqrt(2)*np.pi/np.arccos(1/np.sqrt(3))
        # params = [np.pi/2,np.pi/2,0, np.pi/v_nn, np.pi/v_nn, 0]
        # U = build_circulator_U(*params)
        return self._array.full()


class CParitySwap(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("cpswap", 3, [], "TBD")

        # an alternative defintion using hamiltonian
        # we just have it hardcoded instead

        # nn = 3 * np.sqrt(3) / 2
        # params = [-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn]
        # from hamiltonian import CirculatorHamiltonian
        # self._array = CirculatorHamiltonian.construct_U(*params)

    def __array__(self, dtype=None):
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=dtype,
        )


class Margolus(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("margolus", 3, [], "Margolus")

    def __array__(Self, dtype=None):
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=dtype,
        )


class CanonicalGate(Gate):
    def __init__(self, alpha, beta, gamma, name="can"):
        super().__init__(name, 2, [alpha, beta, gamma], name)
        # normalize to convention
        alpha, beta, gamma = [2 * x / np.pi for x in (alpha, beta, gamma)]
        self.data = weylchamber.canonical_gate(alpha, beta, gamma).full()

    def __array__(self, dtype=None):
        return np.array(self.data, dtype=dtype)


class BerkeleyGate(CanonicalGate):
    def __init__(self):
        super().__init__(np.pi / 4, np.pi / 8, 0, name="B")

    # alternative definition
    # def __init__(self):
    #     from hamiltonian import ConversionGainHamiltonian
    #     ConversionGainHamiltonian.construct_U(3*np.pi/8, np.pi/8)
    #     ...


class CCZGate(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("ccz", 3, [], "CCZGate")

    def __array__(self, dtype=None):
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ],
            dtype=dtype,
        )


class CCiXGate(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("ccix", 3, [], "CCiXGate")

    def __array__(Self, dtype=None):
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1j],
                [0, 0, 0, 0, 0, 0, 1j, 0],
            ],
            dtype=dtype,
        )


class CiSwap(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("ciswap", 3, [], "CiSwap")

    def __array__(Self, dtype=None):
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1j, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=dtype,
        )


class Peres(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("peres", 3, [], "PERES")

    def __array__(Self, dtype=None):
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=dtype,
        )


class FSim(Gate):
    def __init__(self, theta: ParameterValueType, phi: ParameterValueType):
        """SYC: FSim(theta=np.pi/2, phi=np.pi/6)"""
        super().__init__("fsim", 2, [theta, phi])

    def __array__(self, dtype=None):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(self.params[0]), -1j * np.sin(self.params[0]), 0],
                [0, -1j * np.sin(self.params[0]), np.cos(self.params[0]), 0],
                [0, 0, 0, np.exp(1j * self.params[1])],
            ],
            dtype=dtype,
        )


class SYC(FSim):
    def __init__(self, _):
        super().__init__(np.pi / 2, np.pi / 6)

    @staticmethod
    def latex_string(gate_params):
        return "SYC"


class RiSwapGate(Gate):
    r"""RiSWAP gate.

    **Circuit Symbol:**

    .. parsed-literal::

        q_0: ─⨂─
           R(alpha)
        q_1: ─⨂─

    """

    def __init__(self, alpha: ParameterValueType):
        """Create new iSwap gate."""
        super().__init__(
            "riswap", 2, [alpha], label=r"$\sqrt[" + str(int(1 / alpha)) + r"]{iSwap}$"
        )

    def __array__(self, dtype=None):
        """Return a numpy.array for the RiSWAP gate."""
        alpha = self.params[0]
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(np.pi * alpha / 2), 1j * np.sin(np.pi * alpha / 2), 0],
                [0, 1j * np.sin(np.pi * alpha / 2), np.cos(np.pi * alpha / 2), 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
        )

    @staticmethod
    def latex_string(gate_params=None):
        if gate_params is None:
            return r"$\sqrt[n]{iSwap}$"
        else:
            n = 1 / gate_params[0]
            return r"$\sqrt[" + str(int(n)) + r"]{iSwap}$"
