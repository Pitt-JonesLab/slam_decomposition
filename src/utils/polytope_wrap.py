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

MAX_ITERS = 10
"""Helper function for monodromy polytope package"""

# NOTE I'm not sure the best way to do this or if there is a more direct way already in the monodromy package somewhere
# references:
# https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/synthesis/xx_decompose/decomposer.py
# https://github.com/evmckinney9/monodromy/blob/main/scripts/single_circuit.py

def monodromy_range_from_target(basis:CircuitTemplate, target_u) -> range:
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
        print("here")
        pass
    
        return None

def get_polytope_from_circuit(basis: CircuitTemplate) -> ConvexPolytope:
    if basis.n_qubits != 2:
        raise ValueError("monodromy only for 2Q templates")
    
    circuit_polytope = identity_polytope

    dag = circuit_to_dag(basis.circuit)
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
def gate_set_to_haar_expectation(*basis_gates:list[CustomCostGate], chatty=True):
    coverage = gate_set_to_coverage(*basis_gates, chatty=chatty)
    return coverage_to_haar_expectation(coverage, chatty=chatty)

def gate_set_to_coverage(*basis_gates:list[CustomCostGate], chatty=True):

    #first converts all individal gates to circuitpolytope objeect
    operations = []

    for gate in basis_gates:
        circuit_polytope = identity_polytope

        b_polytope = exactly(
            *(Fraction(x).limit_denominator(10_000)
            for x in unitary_to_monodromy_coordinate(np.matrix(gate))[:-1])
        )
        circuit_polytope = deduce_qlr_consequences(
            target="c",
            a_polytope=circuit_polytope,
            b_polytope=b_polytope,
            c_polytope=everything_polytope
        )

        operations.append(
            CircuitPolytope(
                operations=[str(gate)],
                #cost=1,
                #cost=1 - (1- base_iswap_fidelity)*basis_gate.params[0],
                cost= 1, #gate.cost, #if isinstance(gate, CustomCostGate) else 1, #XXX danger
                convex_subpolytopes=circuit_polytope.convex_subpolytopes)
            )

    #second build coverage set which finds the necessary permutations to do a complete span
    logging.info("==== Working to build a set of covering polytopes ====")
    coverage_set = build_coverage_set(operations, chatty=chatty)

    #TODO: add some warning or fail condition if the coverage set fails to coverage
    #one way, (but there may be a more direct way) is to check if expected haar == 0

    if chatty: 
        logging.info("==== Done. Here's what we found: ====")
        logging.info(print_coverage_set(coverage_set))
    return coverage_set

def coverage_to_haar_expectation(coverage_set, chatty=True):
    #finally, return the expected haar coverage
    logging.info("==== Haar volumes ====")
    cost = expected_cost(coverage_set, chatty=chatty)
    stdout.flush() #fix out of order logging
    logging.info(f"Haar-expectation cost: {cost}")
    return cost
