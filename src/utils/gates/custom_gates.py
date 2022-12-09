from re import L
import numpy as np
import weylchamber
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from typing import List
from qiskit.extensions import UnitaryGate
from src.hamiltonian import CirculatorHamiltonian, ConversionGainPhaseHamiltonian, ConversionGainSmush

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
    def __init__(self, unitary, str, cost=1, duration=1, num_qubits=2):
        self.unitary = unitary
        self.str= str
        self.c = cost
        
        # self.global_phase = 0 #idk why the dag method requires this
        super().__init__(str, num_qubits=num_qubits, params=[], label=str)
        self.duration  = duration # duration is attribution not duration
    @classmethod
    def from_gate(cls, gate:Gate, cost:float):
        return cls(gate.to_matrix(), str(gate), cost=cost)

    # we use this duration property in the speed limit pass sub
    # we build a dummy gate that sets duration such that fooanalysis counts correctly
    # the fam substitution messes up our custom scaled gates but this is a nice work around
    # def duration(self):
    #     return self.d


    def cost(self):
        return self.c

    def __str__(self):
        return self.str

    def __array__(self, dtype=None):
        return np.array(self.unitary,dtype)

# class VSwap(Gate):
#     def __init__(self, t_el: ParameterValueType = 1):
#         super().__init__("vswap", 3, [t_el], "VSWAP")
#         # v_nn = np.sqrt(2) * np.pi / np.arccos(1 / np.sqrt(3))
#         # self.v_params = [np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0]

#         #use a more standardized normalization
#         v_nn = 4/np.sqrt(2) #1.5iswap
#         self.v_params = [np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0] #V-swap

#         #self._array = CirculatorHamiltonian.construct_U(*v_params,t=t_el)

#     def __array__(self, dtype=None):
#         # v_nn = np.sqrt(2)*np.pi/np.arccos(1/np.sqrt(3))
#         # params = [np.pi/2,np.pi/2,0, np.pi/v_nn, np.pi/v_nn, 0]
#         self._array = CirculatorHamiltonian.construct_U(*self.v_params,t=self.params[0])
#         return self._array.full()

#     def inverse(self):
#         return UnitaryGate(np.matrix.getH(self.__array__()))

# class DeltaSwap(Gate):
#     def __init__(self, t_el: ParameterValueType = 1):
#         super().__init__("Δswap", 3, [t_el], "ΔSWAP")
#         nn = 3 * np.sqrt(3) / 2
#         self.v_params = [np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn]   #smiley

#     def __array__(self, dtype=None):
#         self._array = CirculatorHamiltonian.construct_U(*self.v_params,t=self.params[0])
#         return self._array.full()

#     def inverse(self):
#         return UnitaryGate(np.matrix.getH(self.__array__()))

class CirculatorSNAILGate(Gate):
    def __init__(self, p1:ParameterValueType, p2:ParameterValueType, p3:ParameterValueType, g1:ParameterValueType, g2:ParameterValueType, g3:ParameterValueType, t_el: ParameterValueType = 1):
        super().__init__("3QGate", 3, [p1, p2, p3, g1, g2, g3, t_el], "3QGate")
        # XXX can only assign duration after init with real values
        if all([isinstance(p, (int, float)) for p in self.params]):
            self.duration = self.cost()

    #NOTE: we don't want this param in the constr ctor, it messes len(signature(gate).parameters) in template construction
    def set_str(self, str):
        self._label = str
        self._name = str

    def __array__(self, dtype=None):
        self._array = CirculatorHamiltonian.construct_U(*[float(el) for el in self.params[0:-1]], t= float(self.params[-1]))
        return self._array.full()

    def cost(self):
        # #something to prevent infinitely small/negative values
        # if all([float(el) <= (1/20) for el in self.params[3:-1]]):
        #     return 0 
        
        norm = np.pi/2
        #abs because g can be negative, just consider its absolute strength
        c = (sum(abs(np.array(self.params[3:-1]))) * self.params[-1])/norm
        return c
    
    def fidelity(self):
        c = self.cost()
        base = .999
        return np.max(1 - (1-base)*float(c) , 0)

    def inverse(self):
        return UnitaryGate(np.matrix.getH(self.__array__()))

class VSwap(CirculatorSNAILGate):
    def __init__(self,t_el: ParameterValueType = 1,) -> None:
        v_nn = 4/np.sqrt(2) #1.5iswap
        super().__init__(*[np.pi / 2, np.pi / 2, 0, np.pi / v_nn, np.pi / v_nn, 0], t_el=t_el)
        self.set_str("VSWAP")

class DeltaSwap(CirculatorSNAILGate):
    def __init__(self, t_el: ParameterValueType = 1) -> None:
        nn = 3 * np.sqrt(3) / 2
        super().__init__(*[np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / nn, np.pi / nn, np.pi / nn], t_el=t_el)
        self.set_str("Δ-iSWAP")

class ConversionGainGate(Gate):
    def __init__(self, p1:ParameterValueType, p2:ParameterValueType, g1:ParameterValueType, g2:ParameterValueType, t_el: ParameterValueType = 1):
        super().__init__("2QGate", 2, [p1, p2, g1, g2, t_el], "2QGate")
        # XXX can only assign duration after init with real values
        if all([isinstance(p, (int, float)) for p in self.params]):
            self.duration = self.cost() # XXX not really duration since always normalized to norm=1
            self.name = str(self)

    def __array__(self, dtype=None):
        self._array = ConversionGainPhaseHamiltonian.construct_U(*[float(el) for el in self.params[0:-1]], t= float(self.params[-1]))
        return self._array.full()

    #overwrite string representation
    def __str__(self):
        g1 = self.params[2]
        g2 = self.params[3]
        t = self.params[4]
        # truncate to 8 decimal places
        s= f"2QGate({g1:.8f}, {g2:.8f}, {t:.8f})"
        return s
    
    def normalize_duration(self, new_duration):
        # scales g terms such that t is new_duration
        # this is useful for loading gates from a file, matching file hashes
        # save the old duration
        old_duration = self.duration
        t = self.params[-1]
        self.params[2] = self.params[2] * t / new_duration
        self.params[3] = self.params[3] * t / new_duration
        self.params[-1] = new_duration
        # assert the duration has not hcnaged
        assert self.duration == old_duration
        self = ConversionGainGate(*self.params)
    
    def cost(self):
        norm = np.pi/2
        #sum the g terms
        c = (sum(abs(np.array(self.params[2:4]))) * self.params[-1])/norm
        return c

class ConversionGainSmushGate(Gate):
    def __init__(self, pc:ParameterValueType, pg: ParameterValueType, gc: ParameterValueType, gg: ParameterValueType, gx: List[ParameterValueType], gy: List[ParameterValueType], t_el: ParameterValueType=1):
        self.xy_len = len(gx)
        assert len(gx) == len(gy)
        self.t_el = t_el
        super().__init__("2QSmushGate", 2, [pc, pg, gc, gg, *gx, *gy, t_el], "2QSmushGate")
        # XXX can only assign duration after init with real values
        # XXX vectors will break this type checking
        # XXX not checking if time is real valued!! 
        if all([isinstance(p, (int, float)) for p in self.params[0:4]]):
            self.duration = self.cost()

    def __array__(self, dtype=None):
        self._array = ConversionGainSmush.construct_U(float(self.params[0]), float(self.params[1]), float(self.params[2]), float(self.params[3]), [float(el) for el in self.params[4:4+self.xy_len]], [float(el) for el in self.params[4+self.xy_len:-1]], t= float(self.params[-1]))
        return self._array #don't need full() since multiplication happens inside the construct_U function
    
    def cost(self):
        norm = np.pi/2
        #sum the g terms, ignore the x and y terms
        #idea is that 1Q gates drive qubits not SNAIL so don't contribute to speed limit
        c = (sum(abs(np.array(self.params[2:4]))) * self.params[-1])/norm
        return c

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

from src.hamiltonian import FSimHamiltonian
class FSimHamiltonianGate(Gate):
    def __init__(self, g: ParameterValueType, eta: ParameterValueType, t: ParameterValueType):
        super().__init__("fsim", 2, [g, eta, t])

    def __array__(self, dtype=None):
        self._array = FSimHamiltonian.construct_U(*[float(el) for el in self.params[0:-1]], t= float(self.params[-1]))
        return self._array.full()


class SYC(FSim):
    def __init__(self):
        super().__init__(np.pi / 2, np.pi / 6)

    def __str__(self):
        return self.latex_string(self.params)
    @staticmethod
    def latex_string(gate_params):
        return "SYC"


class   RiSwapGate(Gate):
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
        # XXX can only assign duration after init with real values
        if all([isinstance(p, (int, float)) for p in self.params]):
            self.duration = self.cost()
    
    #including this seems to break the decompose method
    # we don't want to define riswap in terms of other gates, leave it as riswap primitives
    # def _define(self):
    #     from qiskit import QuantumCircuit
    #     qc = QuantumCircuit(2)
    #     from qiskit.circuit.library.standard_gates import iSwapGate
    #     qc.append(iSwapGate().power(1/self.params[0]), [0, 1])
    #     self.definition = qc
    
    def cost(self):
        # norm = np.pi/2
        #I don't need to use nrom bc already considered in parameter definition
        #e.g. sqisw has params[0] := 1/2
        return float(self.params[0])

    def fidelity(self):
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
