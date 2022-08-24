from re import L
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

class CustomCostGate(Gate):
    #want to build a gate progamatically from a unitary
    #cost value used in expected haar calcuation
    def __init__(self, unitary, str, cost=1):
        self.unitary = unitary
        self.str= str
        self.cost = cost #i.e. duration
        super().__init__(str, num_qubits=2, params=[], label=str)
    
    @classmethod
    def from_gate(cls, gate:Gate, cost:float):
        return cls(gate.to_matrix(), str(gate), cost=cost)

    def __str__(self):
        return self.str

    def __array__(self, dtype=None):
        return np.array(self.unitary,dtype)

class VSwap(Gate):
    def __init__(self, t_el: ParameterValueType = 1):
        super().__init__("vswap", 3, [t_el], "VSWAP")
        v_nn = np.sqrt(2) * np.pi / np.arccos(1 / np.sqrt(3))
        self.v_params = [np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0]

        #alternative normalization
        v_nn = 4/np.sqrt(2) #1.5iswap
        self.v_params = [np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0] #V-swap

        #self._array = CirculatorHamiltonian.construct_U(*v_params,t=t_el)

    def __array__(self, dtype=None):
        # v_nn = np.sqrt(2)*np.pi/np.arccos(1/np.sqrt(3))
        # params = [np.pi/2,np.pi/2,0, np.pi/v_nn, np.pi/v_nn, 0]
        self._array = CirculatorHamiltonian.construct_U(*self.v_params,t=self.params[0])
        return self._array.full()

class DeltaSwap(Gate):
    def __init__(self, t_el: ParameterValueType = 1):
        super().__init__("Δswap", 3, [t_el], "ΔSWAP")
        nn = 3 * np.sqrt(3) / 2
        self.v_params = [np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn]   #smiley

    def __array__(self, dtype=None):
        self._array = CirculatorHamiltonian.construct_U(*self.v_params,t=self.params[0])
        return self._array.full()

class CirculatorSNAILGate(Gate):
    def __init__(self, p1:ParameterValueType, p2:ParameterValueType, p3:ParameterValueType, g1:ParameterValueType, g2:ParameterValueType, g3:ParameterValueType, t_el: ParameterValueType = 1):
        super().__init__("3QGate", 3, [p1, p2, p3, g1, g2, g3, t_el], "3QGate")
 
    def __array__(self, dtype=None):
        self._array = CirculatorHamiltonian.construct_U(*[float(el) for el in self.params[0:-1]], t= float(self.params[-1]))
        return self._array.full()

    def cost(self):
        # #something to prevent infinitely small/negative values
        # if all([float(el) <= (1/20) for el in self.params[3:-1]]):
        #     return 0 
        base = .999
        norm = np.pi/2
        c = (sum(self.params[3:-1]) * self.params[-1])/norm
        return np.max(1 - (1-base)*float(c) , 0)

class CParitySwap(Gate):
    def __init__(self, _: ParameterValueType = None):
        super().__init__("cpswap", 3, [], "CParitySwap")

        # an alternative defintion using hamiltonian
        # we just have it hardcoded instead

        # nn = 3 * np.sqrt(3) / 2
        # params = [-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn]  #frowny
        # params = [np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn]   #smiley
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
    def _define(self):
        """
        gate ccx a,b,c
        {
        h c; cx b,c; tdg c; cx a,c;
        t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c; cx a,b;
        t a; tdg b; cx a,b;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit import QuantumRegister
        from qiskit.circuit.library.standard_gates import SwapGate, CSwapGate

        # q_0: ─X──X──X──X─
        #     │  │  │  │ 
        # q_1: ─X──■──X──┼─
        #     │  │     │ 
        # q_2: ─■──X─────X─ 
        
        qc = QuantumCircuit(3, name=self.name)
        qc.cswap(2,0,1)
        qc.cswap(1,0,2)
        qc.swap(0,1)
        qc.swap(0,2)
        self.definition = qc


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

    def __str__(self):
        return "B"
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
    def __init__(self):
        super().__init__(np.pi / 2, np.pi / 6)

    def __str__(self):
        return self.latex_string(self.params)
    @staticmethod
    def latex_string(gate_params):
        return "SYC"


class RiSwapGate(Gate):
    #turns out you can also do qiskit.iSwapGate().power(1/n)
    #but I didnt know about the power fucntion until recently :(

    r"""RiSWAP gate.

    **Circuit Symbol:**

    .. parsed-literal::

        q_0: ─⨂─
           R(alpha)
        q_1: ─⨂─

    """

    def __init__(self, alpha: ParameterValueType):
        """Create new iSwap gate."""
        # super().__init__(
        #     "riswap", 2, [alpha], label=r"$\sqrt[" + str(int(1 / alpha)) + r"]{iSwap}$"
        # )
        super().__init__(
            "riswap", 2, [alpha], label="riswap"
        )
    
    def _define(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.iswap(0,1)
        self.definition = qc
    
    def cost(self):
        if float(self.params[0]) <= (1/20): #something to prevent infinitely small/negative values
            return 0 
        base = .999
        return np.max(1 - (1-base)*float(self.params[0]),0)

    def __array__(self, dtype=None):
        """Return a numpy.array for the RiSWAP gate."""
        alpha = float(self.params[0]) / 2
        cos = np.cos(np.pi * alpha)
        isin = 1j * np.sin(np.pi* alpha)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, cos, isin, 0],
                [0, isin, cos, 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
        )

    def __str__(self):
        return RiSwapGate.latex_string(self.params)
        
    @staticmethod
    def latex_string(gate_params=None):
        if gate_params is None:
            return r"$\sqrt[n]{iSwap}$"
        else:
            n = 1 / gate_params[0]
            return r"$\sqrt[" + str(int(n)) + r"]{iSwap}$"
