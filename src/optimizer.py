import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from weylchamber import c1c2c3

from .basis import CircuitTemplate, DataDictEntry, VariationalTemplate
from .cost_function import UnitaryCostFunction
from .sampler import SampleFunction

SUCCESS_THRESHOLD = 1e-9
TRAINING_RESTARTS = 5
"""
Given a gate basis objects finds parameters which minimize cost function
"""
class TemplateOptimizer:
    def __init__(self, basis:VariationalTemplate, objective:UnitaryCostFunction, use_callback=False):
        self.basis = basis
        self.objective = objective
        self.preseeding = self.basis.preseeded

        self.use_callback = use_callback
        self.sample_iter = 0
        self.training_loss = [] #2d list sample_iter -> [training iter -> loss]
        self.training_reps = [] #1d list sample_iter -> best result cycle length
        self.coordinate_list = [] #2d list sample_iter -> [training iter -> (coordinate)]

    #TODO, investigate, when does basis data_dict get updated, when do I need to call save to file
    def approximate_target_U(self, target_U):

        target_coordinates = self.basis.target_invariant(target_U)

        if self.preseeding and self.basis.coordinate_tree is not None:
            #check if coordinate already exists in loaded_data
            distance, index = self.basis.coordinate_tree.query([target_coordinates])
            close_coords = self.basis.coordinate_tree.data[index]
            found_saved = self.basis.data_dict[close_coords]

            #TODO rewrite needs to check over k nearest neighbors to find first valid
            #check if valid for given template means success and correct template length
            if found_saved.success_label and found_saved.cycles == self.basis.get_spanning_range()[0]:
                pass
                
                if distance == 0:
                    logging.info(f"Found saved: {target_coordinates}")
                    return found_saved
                
                else:
                    logging.info(f"Preseed from neighbor: {close_coords}")
                    self.basis.assign_seed(found_saved.Xk)

        logging.info(f"Begin search: {target_coordinates}")
        best_result, best_Xk, best_cycles = self.run(target_U)

        if best_result <= SUCCESS_THRESHOLD:
            # label target coordinate as success
            success_label = 1
            logging.info(f"Success: {target_coordinates}")
        else:
            # label target coordinate as fail, label alternative coordinate as succss
            success_label = 0

            #reset back to best size
            if isinstance(self.basis, CircuitTemplate):
                self.basis.build(n_repetitions=best_cycles)
            alternative_coordinate = c1c2c3(self.basis.eval(best_Xk))
    
            logging.info(f"Fail: {target_coordinates}, Alternative: {alternative_coordinate}")
            self.basis.data_dict[alternative_coordinate] = DataDictEntry(1, 0, best_Xk, best_cycles)

        #save target, update tree
        target_data = DataDictEntry(success_label, best_result, best_Xk, best_cycles)
        self.basis.data_dict[target_coordinates] = target_data
        self.basis._construct_tree()
        self.basis.save_data()
        return target_data

    def approximate_from_distribution(self, sampler:SampleFunction):
        for target in sampler:
            logging.info(f"Starting sample iter {self.sample_iter}")
            self.approximate_target_U(target_U=target)
            self.sample_iter += 1
        return self.training_loss, self.training_reps, self.coordinate_list
            
    def run(self, target_u):
        
        def objective_func(xk):
            current_u = self.basis.eval(xk)
            return self.objective.unitary_fidelity(current_u, target_u)
        
        # callback used to save current loss and coordiante after each iteration
        def callbackF(xk):
            current_state = self.basis.eval(xk)
            loss = objective_func(xk)
            temp_training_loss.append(loss)
            temp_coordinate_list.append(c1c2c3(current_state))

        best_result = None
        best_Xk = None
        best_cycles = -1

        # each iter creates fresh template with new repetition param
        for spanning_iter in self.basis.get_spanning_range(target_u):
            logging.info(f"Starting opt on template size {spanning_iter}")
            temp_training_loss = []
            temp_coordinate_list = []
        
            if isinstance(self.basis, CircuitTemplate):
                self.basis.build(n_repetitions=spanning_iter)

            #TODO can do this in threads?
            for _ in range(TRAINING_RESTARTS):
           
                result = opt.minimize(
                    fun=objective_func,
                    method='BFGS',
                    x0=self.basis.paramter_guess(),
                    callback=callbackF if self.use_callback else None,
                    options={"maxiter": 200},
                )

                # result is good, update temp vars
                if best_result is None or result.fun < best_result:
                    best_result = result.fun
                    best_Xk = result.x
                    best_cycles = spanning_iter
                
                #break over starting attempts
                if best_result < SUCCESS_THRESHOLD:
                    if self.use_callback:
                        self.training_loss.append(temp_training_loss)
                        self.coordinate_list.append(temp_coordinate_list)
                    break
            
            #break over template extensions
            # already good enough, save time by stopping here
            if best_result < SUCCESS_THRESHOLD:
                logging.info(f"Break on cycle {spanning_iter}")
                break
        
        if self.use_callback:
            self.training_reps.append(best_cycles)

        logging.info(f"Loss={best_result}")
        return best_result, best_Xk, best_cycles
