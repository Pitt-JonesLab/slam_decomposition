import logging
import random
from abc import ABC
from sys import maxsize

from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import Operator, random_clifford, random_unitary
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks, CountOps
from qiskit.transpiler.passmanager import PassManager

from slam.utils.gates.custom_gates import RiSwapGate
from slam.utils.transpiler_pass.weyl_decompose import RootiSwapWeylDecomposition

logger = logging.getLogger()
"""Define functions which act as distributions for a template to train
against."""


class SampleFunction(ABC):
    def __init__(self, n_qubits=2, n_samples=1):
        self.n_qubits = n_qubits
        self.n_samples = n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            yield self._get_unitary()

    def _get_unitary(self):
        raise NotImplementedError


class GateSample(SampleFunction):
    def __init__(self, gate: Gate, n_samples=1):
        self.gate = gate
        super().__init__(gate.num_qubits, n_samples)

    def _get_unitary(self):
        return Operator(self.gate).data


class CircuitSample(SampleFunction):
    def __init__(self, circuit: QuantumCircuit):
        pm = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
        self.transpiled_circuit = pm.run(circuit)
        super().__init__(n_qubits=2, n_samples=len(self.transpiled_circuit))
        logging.info(f"Created sampler with {self.n_samples} 2Q gates")

    def __iter__(self):
        for instruction in self.transpiled_circuit:
            yield self._get_unitary(instruction)

    def _get_unitary(self, instruction):
        return instruction[0].to_matrix()


class Clifford(SampleFunction):
    def _get_unitary(self):
        return Operator(random_clifford(num_qubits=self.n_qubits)).data


class HaarSample(SampleFunction):
    def __init__(self, seed=None, n_samples=1, n_qubits=2):
        self.seed = seed
        super().__init__(n_samples=n_samples, n_qubits=n_qubits)

    def _get_unitary(self):
        random.seed(self.seed)
        return random_unitary(
            dims=2**self.n_qubits, seed=random.randint(0, maxsize)
        ).data

    def _haar_ground_truth(self, haar_exact=2):
        """When using sqrt[2] iswap, we might want to do a haar sample where we
        know ahead of time if it will take 2 or 3 uses this is used for
        establishing the effectiveness of our optimizer, but won't work for any
        other basis gate."""
        logging.warning("This sampler only works for \sqrt[2]iSwap")
        pm0 = PassManager()
        pm0.append(RootiSwapWeylDecomposition(basis_gate=RiSwapGate(0.5)))
        pm0.append(CountOps())
        logger.setLevel(
            logging.CRITICAL
        )  # turn off logging here so we don't see lots of irrelevant things
        while True:
            qc = QuantumCircuit(2)
            qc.append(random_unitary(dims=4), [0, 1])
            pm0.run(qc)
            if haar_exact == pm0.property_set["count_ops"]["riswap"]:
                logger.setLevel(logging.INFO)
                return qc


class Haar2Sample(HaarSample):
    def __init__(self, seed=None, n_samples=1):
        super().__init__(seed=seed, n_samples=n_samples)

    def _get_unitary(self):
        return Operator(self._haar_ground_truth(2)).data


class Haar3Sample(HaarSample):
    def __init__(self, seed=None, n_samples=1):
        super().__init__(seed=seed, n_samples=n_samples)

    def _get_unitary(self):
        return Operator(self._haar_ground_truth(3)).data
