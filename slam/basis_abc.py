from abc import ABC
from dataclasses import dataclass
from random import uniform
from qiskit.circuit.library.standard_gates import *
from scipy.spatial import KDTree
from weylchamber import c1c2c3

from slam.utils.gates.custom_gates import *
from slam.utils.data_utils import pickle_load, pickle_save
from slam.utils.polytopes.polytope_wrap import monodromy_range_from_target
"""
Defines the variational object passed to the optimizer
#TODO: this should extend Qiskit's NLocal class
"""

class VariationalTemplate(ABC):
    """NOTE: most of the functionality in the abstract class is deprecated
    was originally designed for preseeding tasks
    the more updated approaches are in the various subclasses circuittemplatev2 and mixedbasistemplate"""

    def __init__(self, preseed:bool, use_polytopes:bool):
        if self.filename is None:
            raise NotImplementedError
        self.data_dict = pickle_load(self.filename)

        self._construct_tree()

        #messy bit of logic here
        # I want spanning rule to refer to either a function using polytopes given a targer
        # or a constant set range
        # only valid if range is 1 otherwise nuop fails when on boundary
        self.use_polytopes = use_polytopes
        if not self.use_polytopes and self.spanning_range is None:
            raise NotImplementedError
        
        #NOTE: preseeding without polytopes can work if checking that nuop matches spanning range
        #rather than implementing this I'll just force using polytopes so spanning range always matches
        self.preseeded = preseed and self.use_polytopes #(self.use_polytopes or len(self.spanning_range))
        self.seed = None
    
    def eval(self, Xk):
        #evaluate on vector of parameters
        raise NotImplementedError
    
    def parameter_guess(self, temperature=0):
        #return a random vector of parameters
        if self.preseeded and self.seed is not None:
            #add a dash of randomization here, ie +- 5% on each value
            return [el*uniform(1-.05*temperature, 1+ .05*temperature) for el in self.seed]
        return None

    def assign_seed(self, Xk):
        self.seed = Xk
    
    def clear_all_save_data(self):
        self.data_dict = {}
        self._construct_tree()
        self.save_data()

    def save_data(self):
        pickle_save(self.filename, self.data_dict)

    def _construct_tree(self):
        if len(self.data_dict) > 0:
                #for preseeding, a good data structure to find the closest already known coordinate
                self.coordinate_tree = KDTree(list(self.data_dict.keys()))
        else:
            #no data yet
            self.coordinate_tree = None

    #XXX below will fail for 3Q+
    def target_invariant(self, target_U):
        if not (4,4) == target_U.shape:
            return (-1,-1,-1,-1)
            raise NotImplementedError
        return c1c2c3(target_U)

    def undo_invariant_transform(self, target_U):
        # at this point state of self is locally equivalent to target, 
        # what transformation is needed to move state of template to target basis?
        raise NotImplementedError
        #we need this in a transpiler toolflow, but not for now

@dataclass
class DataDictEntry():
    success_label: int
    loss_result: float
    Xk: list
    cycles: int
