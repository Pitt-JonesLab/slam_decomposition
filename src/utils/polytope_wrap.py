from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.basis import VariationalTemplate

from monodromy import *

"""Helper function for monodromy polytope package"""

#NOTE I'm not sure the best way to do this or if there is a more direct way already in the monodromy package somewhere
#reference:https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/synthesis/xx_decompose/decomposer.py

def get_monodromy_span(basis: VariationalTemplate, target_u):
    return range(2,3)
