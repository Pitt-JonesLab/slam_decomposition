import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
import qutip

"""Example Usage:
from custom_gates import *
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.append(RiSwapGate(1/4), [0,1])
qc.draw(output='mpl')
"""


def build_circulator_U(
    phi_ab=0, phi_ac=0, phi_bc=np.pi / 2, g_ab=1.0, g_ac=1.0, g_bc=1.0
):
    # TODO: when sweeping phase parameters leave time-independent
    # when sweeping coupling parameters make time-dependet i.e gaussian shaped pusle
    # build raising and lowering operations
    a = qutip.operators.create(N=2)
    I2 = qutip.operators.identity(2)
    A = qutip.tensor(a, I2, I2)
    B = qutip.tensor(I2, a, I2)
    C = qutip.tensor(I2, I2, a)

    # construct circulator Hamiltonian
    H_ab = np.exp(1j * phi_ab) * A * B.dag() + np.exp(-1j * phi_ab) * A.dag() * B
    H_ac = np.exp(1j * phi_ac) * A * C.dag() + np.exp(-1j * phi_ac) * A.dag() * C
    H_bc = np.exp(1j * phi_bc) * B * C.dag() + np.exp(-1j * phi_bc) * B.dag() * C
    H = g_ab * H_ab + g_ac * H_ac + g_bc * H_bc

    # time evolution, if time dependent need to use master-equation
    # qutip.mesolve()
    # U = expm(1j*np.array(H))
    U = (1j * H).expm()
    return U


v_nn = np.sqrt(2) * np.pi / np.arccos(1 / np.sqrt(3))
v_params = [np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0]
_Vswap_array = build_circulator_U(*v_params)

nn = 3 * np.sqrt(3) / 2
params = [-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn]
_Cparity_array = build_circulator_U(*params)


class VSwap(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("vswap", 3, [], "VSWAP")
        v_nn = np.sqrt(2) * np.pi / np.arccos(1 / np.sqrt(3))
        v_params = [np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0]
        self._array = build_circulator_U(*v_params)

    def __array__(self, dtype=None):
        # v_nn = np.sqrt(2)*np.pi/np.arccos(1/np.sqrt(3))
        # params = [np.pi/2,np.pi/2,0, np.pi/v_nn, np.pi/v_nn, 0]
        # U = build_circulator_U(*params)
        return self._array.full()


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


class CCZGate(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("ccz", 3, [], "CCZGate")

    def __array__(Self, dtype=None):
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


class CParitySwap(Gate):
    r"""Yet to be determined 3Q gate"""

    def __init__(self, _: ParameterValueType = None):
        super().__init__("cpswap", 3, [], "TBD")
        nn = 3 * np.sqrt(3) / 2
        params = [-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn]
        self._array = build_circulator_U(*params)

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
