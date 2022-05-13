import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType

"""Example Usage:
from custom_gates import *
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.append(RiSwapGate(1/4), [0,1])
qc.draw(output='mpl')
"""


class CParitySwap(Gate):
    r"""Yet to be determined 3Q gate"""

    def __init__(self, _: ParameterValueType = None):
        super().__init__("cpswap", 3, [], "TBD")

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
    def latex_string(n=None):
        if n is None:
            return r"$\sqrt[n]{iSwap}$"
        else:
            return r"$\sqrt[" + str(int(n)) + r"]{iSwap}$"
