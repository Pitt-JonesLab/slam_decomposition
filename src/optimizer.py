import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from weylchamber import c1c2c3

from .basis import CircuitTemplate, CustomCostCircuitTemplate, DataDictEntry, VariationalTemplate
from .cost_function import UnitaryCostFunction
from .sampler import SampleFunction

SUCCESS_THRESHOLD = 2e-9
TRAINING_RESTARTS = 10
"""
Given a gate basis objects finds parameters which minimize cost function
"""
class TemplateOptimizer:
    def __init__(self, basis:VariationalTemplate, objective:UnitaryCostFunction, use_callback=False):
        self.basis = basis
        self.objective = objective
        self.preseeding = self.basis.preseeded

        self.use_callback = use_callback
        self.training_loss = [] #2d list sample_iter -> [training iter -> loss]
        self.coordinate_list = [] #2d list sample_iter -> [training iter -> (coordinate)]

    def approximate_target_U(self, target_U):

        target_coordinates = self.basis.target_invariant(target_U)
        target_spanning_range = self._initialize_run(target_U, target_coordinates)

        # bad code :(, but here we are checking if init_run is returning either a range 
        # or if preseeding already had the exact target already saved
        if isinstance(target_spanning_range, DataDictEntry):
            return target_spanning_range

        logging.info(f"Begin search: {target_coordinates}")
        best_result, best_Xk, best_cycles = self._run(target_U, target_spanning_range)

        if best_result <= SUCCESS_THRESHOLD:
            # label target coordinate as success
            success_label = 1
            logging.info(f"Success: {target_coordinates}")
        else:
            raise ValueError("Failed to converge. Try increasing restart attempts or increasing temperature scaling on preseed.")
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

    def _initialize_run(self, target_U, target_coordinates=None):

        if target_coordinates is None:
            target_coordinates = self.basis.target_invariant(target_U)

        # target_spanning_range = self.basis.get_spanning_range(target_U)
        if self.preseeding and self.basis.coordinate_tree is not None:
            #TODO rewrite needs to check over k nearest neighbors to find first valid

            #check if coordinate already exists in loaded_data
            distance, index = self.basis.coordinate_tree.query([target_coordinates])
            close_coords = tuple(self.basis.coordinate_tree.data[index[0]])
            found_saved = self.basis.data_dict[close_coords]
            
            #check if valid for given template means success and correct template length
            #XXX what if closest value requires a different number of applications, parameters would be misaligned
            #this if structure means don't need to check spanning range if d=0 break early,
            #then preseed only if spanning range matches the pressed
            #otherwise set spanning range like normal
            if found_saved.success_label: #and found_saved.cycles == target_spanning_range[0]:
        
                if distance == 0:
                    logging.info(f"Found saved: {target_coordinates}")
                    return found_saved

                target_spanning_range = self.basis.get_spanning_range(target_U)
                if found_saved.cycles == target_spanning_range[0]:
                    logging.info(f"Preseed from neighbor: {close_coords}")
                    self.basis.assign_seed(found_saved.Xk)
        else:
            self.basis.assign_seed(None)
            target_spanning_range = self.basis.get_spanning_range(target_U)

        return target_spanning_range

    def cost_from_distribution(self, sampler:SampleFunction):
        if not isinstance(self.basis, CustomCostCircuitTemplate):
            raise ValueError("use customcosttemplate to have defined costs")
        total_cost = 0
        for index, target in enumerate(sampler):
            logging.info(f"Starting sample iter {index}")
            init_info = self._initialize_run(target_U=target)
            #bad code :)
            #here we check if init function either returned an exact result
            if isinstance(init_info, DataDictEntry):
                init_info = init_info.cycles
            #or if it returned a range, convert to a single value
            else:
                init_info = max(init_info)
            total_cost += self.basis.unit_cost(init_info)
        logging.info(f"Total circuit pulse cost: {total_cost}")
        logging.info(f"Average gate pulse cost: {total_cost/(index+1)}")
        return total_cost

    def approximate_from_distribution(self, sampler:SampleFunction):
        for index, target in enumerate(sampler):
            logging.info(f"Starting sample iter {index}")
            self.approximate_target_U(target_U=target)
        return self.training_loss, self.coordinate_list
            
    def _run(self, target_u, target_spanning_range):
        
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

        temp_training_loss = []

        # each iter creates fresh template with new repetition param
        for spanning_iter in target_spanning_range:
            logging.info(f"Starting opt on template size {spanning_iter}")
            temp_coordinate_list = []
            #flags for plotting function
            #sort of convoluted but doing this so can keep data as just a sample list
            temp_training_loss.extend([-1, spanning_iter])
            
        
            if isinstance(self.basis, CircuitTemplate):
                self.basis.build(n_repetitions=spanning_iter)
            else:
                #XXX module reload might cause this to fail if basis.py has been changed and not reloaded
                logging.warning("CircuitTemplate type checking failed")

            #TODO if spanning range == 1, don't pass to optimizer
            #implement a mathematica cloud call?
            
            for r_i in range(TRAINING_RESTARTS):
           
                result = opt.minimize(
                    fun=objective_func,
                    method='BFGS',
                    x0=self.basis.parameter_guess(t=r_i),
                    callback=callbackF if self.use_callback else None,
                    options={"maxiter": 400},
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

        logging.info(f"Loss={best_result}")
        return best_result, best_Xk, best_cycles
