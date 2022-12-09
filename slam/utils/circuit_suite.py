"""Define circuits to test"""
import numpy as np
from qiskit import QuantumCircuit

depth = 2 #completely arbitary idk what to set this to

# VQE
# Entangling Ansatz
from qiskit.circuit.library import EfficientSU2
# need to set all 1Q params to random values
def vqe_linear_lambda(q):
    # set np random seed
    np.random.seed(42)
    # apply the ansatz depth times
    vqe_circuit_linear = EfficientSU2(num_qubits=q, entanglement="linear", reps=depth*2).decompose()
    for param in vqe_circuit_linear.parameters:
        vqe_circuit_linear.assign_parameters({param: np.random.rand()}, inplace=1)
    return vqe_circuit_linear

def vqe_full_lambda(q):
    # set np random seed
    np.random.seed(42)
    vqe_circuit_full = EfficientSU2(num_qubits=q, entanglement="full").decompose()
    for param in vqe_circuit_full.parameters:
        vqe_circuit_full.assign_parameters({param: np.random.rand()}, inplace=1)
    return vqe_circuit_full


# Quantum Volume
from qiskit.circuit.library import QuantumVolume
def qv_lambda(q):
    qv_qc = QuantumVolume(num_qubits=q, depth=q).decompose()
    return qv_qc

# QFT
from qiskit.circuit.library.basis_change import QFT
def qft_lambda(q):
    qft_qc = QFT(q).decompose()
    return qft_qc

# QAOA
from qiskit.circuit.library import QAOAAnsatz
def qaoa_lambda(q):
    # set np random seed
    np.random.seed(42)
    qc_mix = QuantumCircuit(q)
    for i in range(0, q):
        qc_mix.rx(np.random.rand(), i)
    import networkx as nx
    # create a random Graph
    G = nx.gnp_random_graph(q, 0.5, seed=42)
    qc_p = QuantumCircuit(q)
    for pair in list(G.edges()):  # pairs of nodes
        qc_p.rzz(2 * np.random.rand(), pair[0], pair[1])
        qc_p.barrier()

    qaoa_qc = QAOAAnsatz(
        cost_operator=qc_p,
        reps=depth,
        initial_state=None,
        mixer_operator=qc_mix
    ).decompose()
    return qaoa_qc

# Adder
from qiskit.circuit.library.arithmetic.adders.cdkm_ripple_carry_adder import \
    CDKMRippleCarryAdder
def adder_lambda(q):
    if q % 2 != 0:
        raise ValueError("q must be even")
    add_qc = (
        QuantumCircuit(q)
        .compose(CDKMRippleCarryAdder(num_state_qubits=int((q - 1) / 2)), inplace=False)
        .decompose()
        .decompose()
        .decompose()
    )
    return add_qc

# Multiplier
from qiskit.circuit.library.arithmetic.multipliers import RGQFTMultiplier
def multiplier_lambda(q):
    if q % 4 != 0:
        raise ValueError("q must be divisible by 4")
    mul_qc = (
        QuantumCircuit(q)
        .compose(RGQFTMultiplier(num_state_qubits=int(q / 4)), inplace=False)
        .decompose()
        .decompose()
        .decompose()
    )
    return mul_qc

# GHZ 
# from qiskit.circuit.library import GHZState
def ghz_lambda(q):
    ghz_qc = QuantumCircuit(q)
    ghz_qc.h(0)
    for i in range(1, q):
        ghz_qc.cx(0, i)
    return ghz_qc

# Hidden Linear Function
from qiskit.circuit.library import HiddenLinearFunction
def hlf_lambda(q):
    # set np random seed
    np.random.seed(42)
    # create a random symmetric adjacency matrix
    adj_m = np.random.randint(2, size=(q, q))
    adj_m = adj_m + adj_m.T 
    adj_m = np.where(adj_m == 2, 1, adj_m)
    hlf_qc = HiddenLinearFunction(adjacency_matrix=adj_m).decompose()
    return hlf_qc

# # Grover
# from qiskit.algorithms import AmplificationProblem
# from qiskit.algorithms import Grover
# def grover_lambda(q):
#     q = int(q/2) # Grover's take so long because of the MCMT, do a smaller circuit
#     # set numpy seed
#     np.random.seed(42)
#     # integer iteration
#     oracle = QuantumCircuit(q)
#     # mark a random state 
#     oracle.cz(0, np.random.randint(1, q))
#     problem = AmplificationProblem(oracle)
#     grover = Grover(iterations=int(depth/2)) # takes too long to find SWAPs if too many iters
#     grover_qc = grover.construct_circuit(problem).decompose().decompose()
#     return problem

benchmark_lambdas = [qv_lambda, vqe_linear_lambda, vqe_full_lambda, qft_lambda, qaoa_lambda, adder_lambda, multiplier_lambda, ghz_lambda, hlf_lambda]
benchmark_lambdas_no_qv = [el for el in benchmark_lambdas if el != qv_lambda]