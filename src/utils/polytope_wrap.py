from __future__ import annotations

import logging
from typing import TYPE_CHECKING

#using this to avoid a circular import
if TYPE_CHECKING:
    from src.basis import CircuitTemplate
from src.utils.custom_gates import CustomCostGate

from fractions import Fraction
from sys import stdout

import numpy as np
from monodromy.coordinates import unitary_to_monodromy_coordinate
from monodromy.coverage import (CircuitPolytope, build_coverage_set,
                                deduce_qlr_consequences, print_coverage_set)
from monodromy.haar import expected_cost
from monodromy.polytopes import ConvexPolytope
from monodromy.static.examples import (everything_polytope, exactly,
                                       identity_polytope)
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from typing import List

MAX_ITERS = 10
"""Helper function for monodromy polytope package"""

# NOTE I'm not sure the best way to do this or if there is a more direct way already in the monodromy package somewhere
# references:
# https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/synthesis/xx_decompose/decomposer.py
# https://github.com/evmckinney9/monodromy/blob/main/scripts/single_circuit.py

def monodromy_range_from_target(basis:CircuitTemplate, target_u) -> range:
    #NOTE, depending on whether using precomputed coverages,
    # both return a range element, because this is value is used by the optimizer to call build
    # in order to bind the precomputed circuit polytope, we use a set method on basis

    if basis.n_qubits != 2:
        raise ValueError("monodromy only for 2Q templates")

    target_coords = unitary_to_monodromy_coordinate(target_u)
    #old method when polytopes not precomputed
    #NOTE possible deprecated not sure when this would ever be none
    if basis.coverage is None:
        iters = 0
        while (iters == 0 or not circuit_polytope.has_element(target_coords)) and iters < MAX_ITERS:
            iters+=1
            basis.build(iters)
            circuit_polytope = get_polytope_from_circuit(basis)
        
        if iters == MAX_ITERS and not circuit_polytope.has_element(target_coords):
            raise ValueError("Monodromy did not find a polytope containing U, may need better gate or adjust MAX_ITERS")
        logging.info(f"Monodromy found needs {iters} basis applications")
        return range(iters,iters+1)
    else: 
        # new method, now we can iterate over precomputed polytopes
        # we want to find the first polytoped (when sorted by cost) that contains target
        # this sorting might be redundant but we will do it just in case
        sorted_polytopes = sorted(basis.coverage, key=lambda k: k.cost)
        for i, circuit_polytope in enumerate(sorted_polytopes):
            if circuit_polytope.has_element(target_coords):
                #set polytope
                basis.set_polytope(circuit_polytope)
                #return a default range
                return range(i,i+1)
        raise ValueError("Monodromy did not find a polytope containing U")
    
def get_polytope_from_circuit(basis: CircuitTemplate) -> ConvexPolytope:
    from qiskit import QuantumCircuit
    if isinstance(basis, QuantumCircuit):
        if basis.num_qubits != 2:
            raise ValueError("monodromy only for 2Q templates")
        dag = circuit_to_dag(basis)
    else: #if isinstance(basis, CircuitTemplate): #XXX not imported for preventing circular import
        if basis.n_qubits != 2:
            raise ValueError("monodromy only for 2Q templates")
        dag = circuit_to_dag(basis.circuit)
        
    circuit_polytope = identity_polytope
    for gate in dag.two_qubit_ops():
        gd = Operator(gate.op).data
        b_polytope = exactly(
            *(Fraction(x).limit_denominator(10_000)
            for x in unitary_to_monodromy_coordinate(gd)[:-1])
        )
        circuit_polytope = deduce_qlr_consequences(
            target="c",
            a_polytope=circuit_polytope,
            b_polytope=b_polytope,
            c_polytope=everything_polytope
        )
    
    return circuit_polytope

#reference: monodromy/demo.py
def gate_set_to_haar_expectation(*basis_gates:List[CustomCostGate], chatty=True):
    coverage_set, basis_gate_hash_dict = gate_set_to_coverage(*basis_gates, chatty=chatty)
    return coverage_to_haar_expectation(coverage_set, chatty=chatty)

def gate_set_to_coverage(*basis_gates:List[CustomCostGate], chatty=True, cost_1q=0, bare_cost=False):
    #first converts all individal gates to circuitpolytope objeect
    operations = []
    basis_gate_hash_dict = {}
    for gate in basis_gates:

        #this is an ugly solution, but not sure a more direct way
        #when we see the circuit polytope later in the basis build method,
        #we need to reconstruct it as a variational circuit, and need a reference to the gates
        #the circuitpolytopes need a hashable object so we use string
        #this dict looks up string->gate object
        if str(gate) in basis_gate_hash_dict.keys():
            raise ValueError("need unique gate strings for hashing to work")
        basis_gate_hash_dict[str(gate)] = gate

        circuit_polytope = identity_polytope

        b_polytope = exactly(
            *(Fraction(x).limit_denominator(10_000)
            for x in unitary_to_monodromy_coordinate(np.matrix(gate, dtype=complex))[:-1])
        )
        circuit_polytope = deduce_qlr_consequences(
            target="c",
            a_polytope=circuit_polytope,
            b_polytope=b_polytope,
            c_polytope=everything_polytope
        )

        # use bare_cost to get cost in terms of number of gates - will be scaled by costs later
        # idea is we don't need to recompute everytime we change costs for speed limits or 2Q gates
        # this idea can't be used if using mixed basis gate sets because we need to know relative costs
        if bare_cost and len(basis_gates) != 1:
            raise ValueError("bare_cost only works for single 2Q gate sets")
        op_cost = gate.cost() + cost_1q
        if bare_cost:
            op_cost = 1

        operations.append(
            CircuitPolytope(
                operations= [str(gate)],
                cost = op_cost,
                convex_subpolytopes=circuit_polytope.convex_subpolytopes)
            )

    #second build coverage set which finds the necessary permutations to do a complete span
    if chatty:
        logging.info("==== Working to build a set of covering polytopes ====")
    coverage_set = build_coverage_set(operations, chatty=chatty)

    #TODO: add some warning or fail condition if the coverage set fails to coverage
    #one way, (but there may be a more direct way) is to check if expected haar == 0

    if chatty: 
        logging.info("==== Done. Here's what we found: ====")
        logging.info(print_coverage_set(coverage_set))
    # return circuit_polytope
    # return operations
    return coverage_set, basis_gate_hash_dict

def coverage_to_haar_expectation(coverage_set, chatty=True):
    #finally, return the expected haar coverage
    if chatty:
        logging.info("==== Haar volumes ====")
    cost = expected_cost(coverage_set, chatty=chatty)
    stdout.flush() #fix out of order logging
    if chatty:
        logging.info(f"Haar-expectation cost: {cost}")
    return cost

#In-house rendering, also see utils.visualize.py
"""
import matplotlib.pyplot as plt
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

from weylchamber import WeylChamber
w = WeylChamber();

total_coord_list = []
for subpoly in reduced_vertices:
    subpoly_coords = [[float(x) for x in coord] for coord in subpoly]
    total_coord_list += subpoly_coords
    w.scatter(*zip(*subpoly_coords))

from scipy.spatial import ConvexHull
pts = np.array(total_coord_list)
hull = ConvexHull(pts)
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

w.render(ax)
"""