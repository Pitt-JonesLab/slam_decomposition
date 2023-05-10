from abc import ABC
from ast import Pass
from hashlib import sha1

import numpy as np
import qutip

from config import srcpath


def filename_encode(arg):
    hash = sha1(arg.encode()).hexdigest()
    return f"{srcpath}/data/{hash}.pkl"


"""
Hamiltonians defined in terms of raising/lowering operators
Modifications to coefficients realize different unitaries
"""


class Hamiltonian(ABC):
    def __init__(self):
        self.H = lambda: None

    def __repr__(self):
        return filename_encode(type(self).__name__)

    def _construct_H(self, *args):
        return self.H(*args)

    def _construct_U_lambda(self, *args):
        return lambda t: (-1j * t * self._construct_H(*args)).expm()

    @staticmethod
    def construct_U(*args):
        raise NotImplementedError


class FluxQubitHamiltonian(Hamiltonian):
    Pass


class SnailEffectiveHamiltonian(Hamiltonian):
    """Used to find iSwap family gates."""

    # same as conversion gain but H_g=0
    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2)
        B = qutip.tensor(I2, a)
        H_int = A * B.dag() + A.dag() * B
        self.H = lambda geff: geff * H_int

    # static method creates an instance of class, acting like a factory
    @staticmethod
    def construct_U(geff):  # , t=1):
        t = 1
        h_instance = SnailEffectiveHamiltonian()
        return h_instance._construct_U_lambda(geff)(t)


class ConversionGainHamiltonian(Hamiltonian):
    """Used to find B Gate."""

    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2)
        B = qutip.tensor(I2, a)
        H_c = A * B.dag() + A.dag() * B
        H_g = A * B + A.dag() * B.dag()
        self.H = lambda gc, gg: gc * H_c + gg * H_g

    # static method creates an instance of class, acting like a factory
    @staticmethod
    def construct_U(gc, gg):  # , t=1):
        t = 1
        h_instance = ConversionGainHamiltonian()
        return h_instance._construct_U_lambda(gc, gg)(t)


class ConversionGainPhaseHamiltonian(Hamiltonian):
    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2)
        B = qutip.tensor(I2, a)
        # H_c = A * B.dag() + A.dag() * B
        # H_g = A * B + A.dag() * B.dag()

        # construct Hamiltonian
        # fmt: off
        def foo_H(phi_c, phi_g, gc, gg):
            # H_ab = np.exp(1j * phi_ab) * A * B.dag() + np.exp(-1j * phi_ab) * A.dag() * B
            # H_ac = np.exp(1j * phi_ac) * A * C.dag() + np.exp(-1j * phi_ac) * A.dag() * C
            # H_bc = np.exp(1j * phi_bc) * B * C.dag() + np.exp(-1j * phi_bc) * B.dag() * C
            H_c = np.exp(1j * phi_c) * A * B.dag() + np.exp(-1j * phi_c) * A.dag() * B
            H_g = np.exp(1j * phi_g) * A * B + np.exp(-1j * phi_g) * A.dag() * B.dag()
            return gc * H_c + gg * H_g
        # fmt: on

        self.H = foo_H

    # static method creates an instance of class, acting like a factory
    @staticmethod
    def construct_U(gc, gg, phi_c, phi_g, t=1):
        t = float(t)
        h_instance = ConversionGainPhaseHamiltonian()
        return h_instance._construct_U_lambda(gc, gg, phi_c, phi_g)(t)


class ConversionGainSmush(Hamiltonian):
    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2)
        B = qutip.tensor(I2, a)
        # construct Hamiltonian
        # fmt: off
        def foo_H(phi_c, phi_g, gc, gg, gx, gy):
            H_x = (A + A.dag())
            H_y = (B + B.dag())
            H_c = np.exp(1j * phi_c) * A * B.dag() + np.exp(-1j * phi_c) * A.dag() * B
            H_g = np.exp(1j * phi_g) * A * B + np.exp(-1j * phi_g) * A.dag() * B.dag()
            return gx* H_x + gy * H_y + gc * H_c + gg * H_g
        # fmt: on
        self.H = foo_H

    @staticmethod
    def construct_U(phi_c, phi_g, gc, gg, gxvector, gyvector, t=1):
        h_instance = ConversionGainSmush()
        assert len(gxvector) == len(gyvector)
        N = len(gxvector)
        timestep = t / N
        # timestep = 0.1 #1:10 1Q to 2Q gate duration
        totalUi = np.eye(4)
        for it in range(N):
            Ui = h_instance._construct_U_lambda(
                phi_c, phi_g, gc, gg, gxvector[it], gyvector[it]
            )(timestep).full()
            totalUi = Ui @ totalUi
        return totalUi


class ConversionGainSmush1QPhase(Hamiltonian):
    # TODO could make it so that the 1Q phase variables are vectors different at each time step
    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2)
        B = qutip.tensor(I2, a)
        # construct Hamiltonian
        # fmt: off
        def foo_H(phi_a, phi_b, phi_c, phi_g, gc, gg, gz1, gz2, gx, gy):
            H_x = (np.exp(1j * phi_a)*A + np.exp(-1j * phi_a)*A.dag())
            H_y = (np.exp(1j * phi_b)*B + np.exp(-1j * phi_b)*B.dag())
            H_z1 = A.dag()*A
            H_z2 = B.dag()*B
            H_c = np.exp(1j * phi_c) * A * B.dag() + np.exp(-1j * phi_c) * A.dag() * B
            H_g = np.exp(1j * phi_g) * A * B + np.exp(-1j * phi_g) * A.dag() * B.dag()
            return gx* H_x + gy * H_y + gc * H_c + gg * H_g + gz1 * H_z1 + gz2 * H_z2
        # fmt: on
        self.H = foo_H

    @staticmethod
    def construct_U(
        phi_a, phi_b, phi_c, phi_g, gc, gg, gz1, gz2, gxvector, gyvector, t=1
    ):
        h_instance = ConversionGainSmush1QPhase()
        assert len(gxvector) == len(gyvector)
        N = len(gxvector)
        timestep = t / N
        # timestep = 0.1 #1:10 1Q to 2Q gate duration
        totalUi = np.eye(4)
        for it in range(N):
            Ui = h_instance._construct_U_lambda(
                phi_a, phi_b, phi_c, phi_g, gc, gg, gz1, gz2, gxvector[it], gyvector[it]
            )(timestep).full()
            totalUi = Ui @ totalUi
        return totalUi


# #XXX not working yet, how do I want the gxvector, gyvector params to see by Basis?
# class TimeDependentHamiltonian(Hamiltonian):
#     def __init__(self, timesteps):
#         raise NotImplementedError
#         self.timesteps = timesteps
#         super().__init__()

# class Simul1QGatesHamiltonian(TimeDependentHamiltonian):
#     """Smush together 1Q gates to build a new family of 2Q basis gates"""

#     def __init__(self, timesteps=10):
#         super().__init__(timesteps)
#         a = qutip.operators.create(N=2)
#         I2 = qutip.operators.identity(2)
#         A = qutip.tensor(a, I2)
#         B = qutip.tensor(I2, a)
#         self.H = (
#             lambda gx, gy, gz: gx * (A + A.dag())
#             + gy * (B + B.dag())
#             + gz * (A * B.dag() + A.dag() * B)
#         )

#     @staticmethod
#     def construct_U(gxvector, gyvector, full_time=np.pi/2):
#         h_instance = Simul1QGatesHamiltonian()
#         assert len(gxvector) == len(gyvector)
#         N = len(gxvector)
#         timestep = full_time/N
#         totalUi = np.eye(4)
#         for it in range(N):
#             Ui = h_instance._construct_U_lambda(gx=gxvector[it], gy=gyvector[it], gz=1)(timestep).full()
#             totalUi = Ui @ totalUi
#         return totalUi


class FSimHamiltonian(Hamiltonian):
    """https://arxiv.org/pdf/1910.11333.pdf"""

    def __init__(self):
        # a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        sp1 = qutip.tensor(qutip.operators.sigmap(), I2)
        sp2 = qutip.tensor(I2, qutip.operators.sigmap())
        sm1 = qutip.tensor(qutip.operators.sigmam(), I2)
        sm2 = qutip.tensor(I2, qutip.operators.sigmam())
        sz1 = qutip.tensor(qutip.operators.sigmaz(), I2)
        sz2 = qutip.tensor(I2, qutip.operators.sigmaz())

        H_1 = sp1 * sm2 + sm1 * sp2
        H_2 = sz1 * sz2
        self.H = lambda g, eta: g * H_1 + (g**2 / np.abs(eta)) * H_2

    # static method creates an instance of class, acting like a factory
    @staticmethod
    def construct_U(g, eta, t=1):
        h_instance = FSimHamiltonian()
        return h_instance._construct_U_lambda(g, eta)(t)


class CirculatorHamiltonian(Hamiltonian):
    """Use to find VSwap and CParitySwap."""

    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2, I2)
        B = qutip.tensor(I2, a, I2)
        C = qutip.tensor(I2, I2, a)

        # construct circulator Hamiltonian
        # fmt: off
        def foo_H(phi_ab, phi_ac, phi_bc, g_ab, g_ac, g_bc):
            H_ab = np.exp(1j * phi_ab) * A * B.dag() + np.exp(-1j * phi_ab) * A.dag() * B
            H_ac = np.exp(1j * phi_ac) * A * C.dag() + np.exp(-1j * phi_ac) * A.dag() * C
            H_bc = np.exp(1j * phi_bc) * B * C.dag() + np.exp(-1j * phi_bc) * B.dag() * C
            return g_ab * H_ab + g_ac * H_ac + g_bc * H_bc
        # fmt: on

        self.H = foo_H

    @staticmethod
    def construct_U(phi_ab, phi_ac, phi_bc, g_ab, g_ac, g_bc, t):
        # t=1
        t = float(t)  # convert from ParameterExpression
        h_instance = CirculatorHamiltonian()
        return h_instance._construct_U_lambda(phi_ab, phi_ac, phi_bc, g_ab, g_ac, g_bc)(
            t
        )


class DeltaConversionGainHamiltonian(Hamiltonian):
    """Searching for error parity detection gate."""

    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2, I2)
        B = qutip.tensor(I2, a, I2)
        C = qutip.tensor(I2, I2, a)

        # construct circulator Hamiltonian
        # fmt: off
        def foo_H(gphi_ab, gphi_ac, gphi_bc, g_ab, g_ac, g_bc, cphi_ab, cphi_ac, cphi_bc, c_ab, c_ac, c_bc):
            H_c = np.exp(1j * cphi_ac) * A * B.dag() + np.exp(-1j * cphi_ac) * A.dag() * B
            H_g = np.exp(1j * gphi_ab) * A * B + np.exp(-1j * gphi_ab) * A.dag() * B.dag()
            H_ab= c_ab * H_c + g_ab * H_g

            H_c = np.exp(1j * cphi_ac) * A * C.dag() + np.exp(-1j * cphi_ac) * A.dag() * C
            H_g = np.exp(1j * gphi_ac) * A * C + np.exp(-1j * gphi_ac) * A.dag() * C.dag()
            H_ac = c_ac * H_c + g_ac * H_g

            H_c = np.exp(1j * cphi_bc) * B * C.dag() + np.exp(-1j * cphi_bc) * B.dag() * C
            H_g = np.exp(1j * gphi_bc) * B * C + np.exp(-1j * gphi_bc) * B.dag() * C.dag()
            H_bc = c_bc * H_c + g_bc * H_g

            return H_ab + H_ac + H_bc
        # fmt: on

        self.H = foo_H

    @staticmethod
    def construct_U(
        gphi_ab,
        gphi_ac,
        gphi_bc,
        g_ab,
        g_ac,
        g_bc,
        cphi_ab,
        cphi_ac,
        cphi_bc,
        c_ab,
        c_ac,
        c_bc,
    ):
        t = 1
        h_instance = DeltaConversionGainHamiltonian()
        return h_instance._construct_U_lambda(
            gphi_ab,
            gphi_ac,
            gphi_bc,
            g_ab,
            g_ac,
            g_bc,
            cphi_ab,
            cphi_ac,
            cphi_bc,
            c_ab,
            c_ac,
            c_bc,
        )(t)

    # def gaussian(t, A, s):
    #     return A * np.exp(-((t / s) ** 2))

    # def _wrap_build_time_dependent_U(
    #     phi_ab=0,
    #     phi_ac=0,
    #     phi_bc=np.pi / 2,
    #     g_ab_A=0,
    #     g_ab_s=0.1,
    #     g_ac_A=0,
    #     g_ac_s=0.1,
    #     g_bc_A=0,
    #     g_bc_s=0.1,
    # ):
    #     """
    #     Example usage
    #     init = build_init_state(0, 1, 1)
    #     build_time_dependent_U = _wrap_build_time_dependent_U()
    #     result = qutip.mesolve(build_time_dependent_U, init, t_list)
    #     """
    #     a = qutip.operators.create(N=2)
    #     I2 = qutip.operators.identity(2)
    #     A = qutip.tensor(a, I2, I2)
    #     B = qutip.tensor(I2, a, I2)
    #     C = qutip.tensor(I2, I2, a)

    #     # construct circulator Hamiltonian
    #     H_ab = np.exp(1j * phi_ab) * A * B.dag() + np.exp(-1j * phi_ab) * A.dag() * B
    #     H_ac = np.exp(1j * phi_ac) * A * C.dag() + np.exp(-1j * phi_ac) * A.dag() * C
    #     H_bc = np.exp(1j * phi_bc) * B * C.dag() + np.exp(-1j * phi_bc) * B.dag() * C
    #     build_time_dependent_U = (
    #         lambda t, args: gaussian(t, g_ab_A, g_ab_s) * H_ab
    #         + gaussian(t, g_ac_A, g_ac_s) * H_ac
    #         + gaussian(t, g_bc_A, g_bc_s) * H_bc
    #     )
    #     return build_time_dependent_U


# class FluxQubit(Hamiltonian):
#     """https://arxiv.org/pdf/2107.02343.pdf"""
#     def __init__(self):

#         a = qutip.operators.destroy(N=2)
#         I2 = qutip.operators.identity(2)
#         A = qutip.tensor(a, I2, I2) #qubit1
#         B = qutip.tensor(I2, a, I2) #qubit2
#         C = qutip.tensor(I2, I2, a) #coupler
#         alpha = 0 #coupler anharmonicity

#         H_a = lambda wa: wa * A.dag() * A
#         #self.H = lambda t: H_a + H_b + H_c(t) + H_g

#     def __repr__(self):
#         return filename_encode(type(self).__name__)

#     def _construct_H(self, *args):
#         return self.H(*args)

#     def _construct_U_lambda(self, *args):
#         return lambda t: (-1j * t * self._construct_H(*args)).expm()

#     @staticmethod
#     def construct_U(*args):
#         raise NotImplementedError
