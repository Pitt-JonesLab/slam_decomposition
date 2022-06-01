# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Weyl decomposition of two-qubit gates in terms of echoed cross-resonance gates."""

from qiskit import QuantumCircuit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag

from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer

from qiskit.quantum_info.synthesis.weyl import weyl_coordinates
from qiskit.quantum_info.operators import Operator

# from qiskit.circuit.library.standard_gates import *
from custom_gates import RiSwapGate
from qiskit import QuantumCircuit
import numpy as np

import cmath
from qiskit.quantum_info.synthesis.two_qubit_decompose import *
import scipy.linalg as la

from qiskit.circuit.library import RZGate, RXGate, RYGate, IGate
from qiskit.extensions.unitary import UnitaryGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


class RootiSwapWeylDecomposition(TransformationPass):
    """Rewrite two-qubit gates using the Weyl decomposition.
    This transpiler pass rewrites two-qubit gates in terms of root iswap gates according
    to the Weyl decomposition. A two-qubit gate will be replaced with at most 3 root i swap gates.
    """

    def __init__(self, basis_gate=False):
        """RootiSwapWeylDecomposition pass.
        Args:
            instruction_schedule_map (InstructionScheduleMap): the mapping from circuit
                :class:`~.circuit.Instruction` names and arguments to :class:`.Schedule`\\ s.
        """
        super().__init__()
        # self.requires = [
        #     BasisTranslator(_sel, ["u3", "cu3", "cp", "swap", "riswap", "id"])
        # ]
        # self.decompose_swaps = decompose_swaps
        self.basis_gate = basis_gate

    # reference: https://arxiv.org/pdf/2105.06074.pdf

    # from Qiskits two_qubit_decomp #FIXME moving functions around still this won't need to be copied once SQiSwap inside of that same pass

    def KAKDecomp(self, unitary_matrix, *, fidelity=(1.0 - 1.0e-9)):
        _ipx = np.array([[0, 1j], [1j, 0]], dtype=complex)
        _ipy = np.array([[0, 1], [-1, 0]], dtype=complex)
        _ipz = np.array([[1j, 0], [0, -1j]], dtype=complex)
        _id = np.array([[1, 0], [0, 1]], dtype=complex)

        """Perform the Weyl chamber decomposition, and optionally choose a specialized subclass.

        The flip into the Weyl Chamber is described in B. Kraus and J. I. Cirac, Phys. Rev. A 63,
        062309 (2001).

        FIXME: There's a cleaner-seeming method based on choosing branch cuts carefully, in Andrew
        M. Childs, Henry L. Haselgrove, and Michael A. Nielsen, Phys. Rev. A 68, 052311, but I
        wasn't able to get that to work.

        The overall decomposition scheme is taken from Drury and Love, arXiv:0806.4015 [quant-ph].
        """
        pi = np.pi
        pi2 = np.pi / 2
        pi4 = np.pi / 4

        # Make U be in SU(4)
        U = np.array(unitary_matrix, dtype=complex, copy=True)
        detU = la.det(U)
        U *= detU ** (-0.25)
        global_phase = cmath.phase(detU) / 4

        Up = transform_to_magic_basis(U, reverse=True)
        M2 = Up.T.dot(Up)

        # M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
        # P ∈ SO(4), D is diagonal with unit-magnitude elements.
        #
        # We can't use raw `eig` directly because it isn't guaranteed to give us real or othogonal
        # eigenvectors.  Instead, since `M2` is complex-symmetric,
        #   M2 = A + iB
        # for real-symmetric `A` and `B`, and as
        #   M2^+ @ M2 = A^2 + B^2 + i [A, B] = 1
        # we must have `A` and `B` commute, and consequently they are simultaneously diagonalizable.
        # Mixing them together _should_ account for any degeneracy problems, but it's not
        # guaranteed, so we repeat it a little bit.  The fixed seed is to make failures
        # deterministic; the value is not important.
        state = np.random.default_rng(2020)
        for _ in range(100):  # FIXME: this randomized algorithm is horrendous
            M2real = state.normal() * M2.real + state.normal() * M2.imag
            _, P = np.linalg.eigh(M2real)
            D = P.T.dot(M2).dot(P).diagonal()
            if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=0, atol=1.0e-13):
                break
        else:
            raise ValueError

        d = -np.angle(D) / 2
        d[3] = -d[0] - d[1] - d[2]
        cs = np.mod((d[:3] + d[3]) / 2, 2 * np.pi)

        # Reorder the eigenvalues to get in the Weyl chamber
        cstemp = np.mod(cs, pi2)
        np.minimum(cstemp, pi2 - cstemp, cstemp)
        order = np.argsort(cstemp)[[1, 2, 0]]
        cs = cs[order]
        d[:3] = d[order]
        P[:, :3] = P[:, order]

        # Fix the sign of P to be in SO(4)
        if np.real(la.det(P)) < 0:
            P[:, -1] = -P[:, -1]

        # Find K1, K2 so that U = K1.A.K2, with K being product of single-qubit unitaries
        K1 = transform_to_magic_basis(Up @ P @ np.diag(np.exp(1j * d)))
        K2 = transform_to_magic_basis(P.T)

        K1l, K1r, phase_l = decompose_two_qubit_product_gate(K1)
        K2l, K2r, phase_r = decompose_two_qubit_product_gate(K2)
        global_phase += phase_l + phase_r

        K1l = K1l.copy()

        # Flip into Weyl chamber
        if cs[0] > pi2:
            cs[0] -= 3 * pi2
            K1l = K1l.dot(_ipy)
            K1r = K1r.dot(_ipy)
            global_phase += pi2
        if cs[1] > pi2:
            cs[1] -= 3 * pi2
            K1l = K1l.dot(_ipx)
            K1r = K1r.dot(_ipx)
            global_phase += pi2
        conjs = 0
        if cs[0] > pi4:
            cs[0] = pi2 - cs[0]
            K1l = K1l.dot(_ipy)
            K2r = _ipy.dot(K2r)
            conjs += 1
            global_phase -= pi2
        if cs[1] > pi4:
            cs[1] = pi2 - cs[1]
            K1l = K1l.dot(_ipx)
            K2r = _ipx.dot(K2r)
            conjs += 1
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if cs[2] > pi2:
            cs[2] -= 3 * pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if conjs == 1:
            cs[2] = pi2 - cs[2]
            K1l = K1l.dot(_ipz)
            K2r = _ipz.dot(K2r)
            global_phase += pi2
        if cs[2] > pi4:
            cs[2] -= pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase -= pi2

        a, b, c = cs[1], cs[0], cs[2]
        return global_phase, (a, b, c), K1l, K1r, K2l, K2r

    # Reference: https://arxiv.org/pdf/2105.06074.pdf
    def riswapWeylDecomp(self, U):
        """Decompose U into single qubit gates and the SQiSW gates"""
        qc = QuantumCircuit(2)

        _, (x, y, z), A1, A2, B1, B2 = self.KAKDecomp(U)
        if np.abs(z) <= x - y:
            C1, C2 = self.interleavingSingleQubitRotations(x, y, z)
            V = (
                RiSwapGate(0.5).to_matrix()
                @ np.kron(C1, C2)
                @ RiSwapGate(0.5).to_matrix()
            )
            _, (x, y, z), D1, D2, E1, E2 = self.KAKDecomp(V)

            qc.append(UnitaryGate(np.matrix(E1).H @ B1), [1])
            qc.append(UnitaryGate(np.matrix(E2).H @ B2), [0])
            qc.append(RiSwapGate(0.5), [0, 1])
            qc.append(UnitaryGate(C1), [1])
            qc.append(UnitaryGate(C2), [0])
            qc.append(RiSwapGate(0.5), [0, 1])
            qc.append(UnitaryGate(A1 @ np.matrix(D1).H), [1])
            qc.append(UnitaryGate(A2 @ np.matrix(D2).H), [0])
        else:
            (x, y, z), F1, F2, G1, G2, H1, H2 = self.canonicalize(x, y, z)
            C1, C2 = self.interleavingSingleQubitRotations(x, y, z)
            V = (
                RiSwapGate(0.5).to_matrix()
                @ np.kron(C1, C2)
                @ RiSwapGate(0.5).to_matrix()
            )
            _, (x, y, z), D1, D2, E1, E2 = self.KAKDecomp(V)

            qc.append(UnitaryGate(H1 @ B1), [1])
            qc.append(UnitaryGate(H2 @ B2), [0])
            qc.append(RiSwapGate(0.5), [0, 1])
            qc.append(UnitaryGate(np.matrix(E1).H @ G1), [1])
            qc.append(UnitaryGate(np.matrix(E2).H @ G2), [0])
            qc.append(RiSwapGate(0.5), [0, 1])
            qc.append(UnitaryGate(C1), [1])
            qc.append(UnitaryGate(C2), [0])
            qc.append(RiSwapGate(0.5), [0, 1])
            qc.append(UnitaryGate(A1 @ F1 @ np.matrix(D1).H), [1])
            qc.append(UnitaryGate(A2 @ F2 @ np.matrix(D2).H), [0])

        return qc.decompose()

    def interleavingSingleQubitRotations(self, x, y, z):
        """Output the single qubit rotations given the interaction coefficients (x,y,z) \in W' when sandiwched by two SQiSW gates"""
        C = (
            np.sin(x + y - z)
            * np.sin(x - y + z)
            * np.sin(-x - y - z)
            * np.sin(-x + y + z)
        )
        alpha = np.arccos(
            np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) + 2 * np.sqrt(C)
        )
        beta = np.arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) - 2 * np.sqrt(C))
        _num = 4 * (np.cos(x) ** 2) * (np.cos(z) ** 2) * (np.cos(y) ** 2)
        _den = _num + np.cos(2 * x) + np.cos(2 * y) * np.cos(2 * z)
        gamma = np.arccos(np.sign(z) * np.sqrt(_num / _den))
        return (
            RZGate(gamma).to_matrix()
            @ RXGate(alpha).to_matrix()
            @ RZGate(gamma).to_matrix(),
            RXGate(beta).to_matrix(),
        )

    def canonicalize(self, x, y, z):
        """Decompose an arbitrary gate into one SQISW and one L(x,y',z) where (x',y',z') \in W' and output the coefficients (x',y',z') and the interleaving single qubit rotations"""
        A1 = IGate().to_matrix()
        A2 = IGate().to_matrix()
        B1 = RYGate(-np.pi / 2).to_matrix()
        B2 = RYGate(np.pi / 2).to_matrix()
        C1 = RYGate(np.pi / 2).to_matrix()
        C2 = RYGate(-np.pi / 2).to_matrix()
        s = np.sign(z)
        z = np.abs(z)
        if x > np.pi / 8:
            y = y - np.pi / 8
            z = z - np.pi / 8
            B1 = RZGate(np.pi / 2).to_matrix() @ B1
            B2 = RZGate(-np.pi / 2).to_matrix() @ B2
            C1 = C1 @ RZGate(-np.pi / 2).to_matrix()
            C2 = C2 @ RZGate(np.pi / 2).to_matrix()

        else:
            x = x + np.pi / 8
            z = z - np.pi / 8

        if np.abs(y) < np.abs(z):
            # XXX typo in alibaba here (?)
            z = -z
            A1 = RXGate(np.pi / 2).to_matrix()
            A2 = RXGate(-np.pi / 2).to_matrix()
            B1 = RXGate(-np.pi / 2).to_matrix() @ B1
            B2 = RXGate(np.pi / 2).to_matrix() @ B2
        if s < 0:
            z = -z
            A1 = RZGate(np.pi).to_matrix() @ A1 @ RZGate(np.pi).to_matrix()
            B1 = RZGate(np.pi).to_matrix() @ B1 @ RZGate(np.pi).to_matrix()
            C1 = RZGate(np.pi).to_matrix() @ C1 @ RZGate(np.pi).to_matrix()

        return (x, y, z), A1, A2, B1, B2, C1, C2

    def run(self, dag: DAGCircuit):
        """Run the RootiSwapWeylDecomposition pass on `dag`.
        Rewrites two-qubit gates in an arbitrary circuit in terms of echoed cross-resonance
        gates by computing the Weyl decomposition of the corresponding unitary. Modifies the
        input dag.
        Args:
            dag (DAGCircuit): DAG to rewrite.
        Returns:
            DAGCircuit: The modified dag.
        Raises:
            TranspilerError: If the circuit cannot be rewritten.
        """
        # pylint: disable=cyclic-import

        if len(dag.qregs) > 1:
            raise TranspilerError(
                "RootiSwapWeylDecomposition expects a single qreg input DAG,"
                f"but input DAG had qregs: {dag.qregs}."
            )

        if isinstance(self.basis_gate, RiSwapGate):
            self.decomposer = self.riswapWeylDecomp
        else:
            self.decomposer = TwoQubitBasisDecomposer(self.basis_gate)

        for node in dag.two_qubit_ops():
            unitary = Operator(node.op).data
            dag_sub = circuit_to_dag(self.decomposer(unitary))
            dag.substitute_node_with_dag(node, dag_sub)

        return dag
