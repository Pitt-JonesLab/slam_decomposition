from abc import ABC

import numpy as np
import qutip

from src.utils.data_utils import filename_encode

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
        return lambda t: (1j * t * self._construct_H(*args)).expm()

    @staticmethod
    def construct_U(*args):
        raise NotImplementedError

class ConversionGainHamiltonian(Hamiltonian):
    """Used to find B Gate"""

    def __init__(self):
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2)
        B = qutip.tensor(I2, a)
        H_c = A * B.dag() + A.dag() * B
        H_g = A * B + A.dag() * B.dag()
        self.H = lambda gc, gg: gc * H_c + gg * H_g
    
    #static method creates an instance of class, acting like a factory
    @staticmethod
    def construct_U(gc, gg):#, t=1):
        t = 1
        h_instance = ConversionGainHamiltonian()
        return h_instance._construct_U_lambda(gc, gg)(t)


#XXX not working yet, how do I want the gxvector, gyvector params to see by Basis?
class TimeDependentHamiltonian(Hamiltonian):
    def __init__(self, timesteps):
        raise NotImplementedError
        self.timesteps = timesteps
        super().__init__()

class Simul1QGatesHamiltonian(TimeDependentHamiltonian):
    """Smush together 1Q gates to build a new family of 2Q basis gates"""

    def __init__(self, timesteps=10):
        super().__init__(timesteps)
        a = qutip.operators.create(N=2)
        I2 = qutip.operators.identity(2)
        A = qutip.tensor(a, I2)
        B = qutip.tensor(I2, a)
        self.H = (
            lambda gx, gy, gz: gx * (A + A.dag())
            + gy * (B + B.dag())
            + gz * (A * B.dag() + A.dag() * B)
        )

    @staticmethod
    def construct_U(gxvector, gyvector, full_time=np.pi/2):
        h_instance = Simul1QGatesHamiltonian()
        assert len(gxvector) == len(gyvector)
        N = len(gxvector)
        timestep = full_time/N
        totalUi = np.eye(4)
        for it in range(N):
            Ui = h_instance._construct_U_lambda(gx=gxvector[it], gy=gyvector[it], gz=1)(timestep).full()
            totalUi = Ui @ totalUi
        return totalUi

class CirculatorHamiltonian(Hamiltonian):
    """Use to find VSwap and CParitySwap"""

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
            return g_ac * H_ac + g_bc * H_bc + g_ab * H_ab
        # fmt: on

        self.H = foo_H

    @staticmethod
    def construct_U(phi_ab, phi_ac, phi_bc, g_ab, g_ac, g_b, t=1):
        h_instance = CirculatorHamiltonian()
        return h_instance._construct_U_lambda(phi_ab, phi_ac, phi_bc, g_ab, g_ac, g_b)(t)

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

