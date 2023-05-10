import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from weylchamber import c1c2c3

from slam.basisv2 import CircuitTemplateV2

from slam.basis_abc import DataDictEntry, VariationalTemplate
from slam.basis import CircuitTemplate, MixedOrderBasisCircuitTemplate
from slam.cost_function import UnitaryCostFunction, EntanglementCostFunction, BasicCostInverse, LineSegmentDistanceCost
from slam.sampler import SampleFunction

from tqdm import tqdm

SUCCESS_THRESHOLD = 1e-10
TRAINING_RESTARTS = 5
"""
Given a gate basis objects finds parameters which minimize cost function
"""
class TemplateOptimizer:
    def __init__(self, basis:VariationalTemplate, objective:UnitaryCostFunction, use_callback=False, override_fail=False, success_threshold=None, training_restarts=None, override_method=None):
        self.basis = basis
        self.objective = objective
        self.preseeding = self.basis.preseeded

        self.use_callback = use_callback
        self.training_loss = [] #2d list sample_iter -> [training iter -> loss]
        self.coordinate_list = [] #2d list sample_iter -> [training iter -> (coordinate)]
        #used for counting haar length
        self.best_cycle_list = []
        self.override_fail = override_fail
        self.override_method = override_method

        if success_threshold is not None:
            self.success_threshold = success_threshold
        else:
            self.success_threshold = SUCCESS_THRESHOLD

        if training_restarts is not None:
            self.training_restarts = training_restarts
        else:
            self.training_restarts = TRAINING_RESTARTS

        assert not (self.preseeding and self.override_fail) #don't want to save failed data
        assert not (self.preseeding and self.basis.n_qubits !=2) #currently preseeding based on weyl coordinates

    def approximate_target_U(self, target_U):
        """Atomic training function"""
        target_coordinates = self.basis.target_invariant(target_U)
        init_run_results = self._initialize_run(target_U, target_coordinates)

        # bad code :(, but here we are checking if init_run is returning either a range 
        # or if preseeding already had the exact target already saved
        if isinstance(init_run_results, DataDictEntry):
            return target_spanning_range
        else:
            target_spanning_range = init_run_results

        logging.info(f"Begin search: {target_coordinates}")
        best_result, best_Xk, best_cycles = self._run(target_U, target_spanning_range)

        if best_result <= self.success_threshold:
            # label target coordinate as success
            success_label = 1
            #logging.info(f"Success: {target_coordinates}")
            if self.basis.n_qubits == 2:
                alternative_coordinate = c1c2c3(self.basis.eval(best_Xk))
                logging.info(f"Success: {target_coordinates}, Found: {alternative_coordinate}")
        else:
            if not self.override_fail:
                raise ValueError("Failed to converge within error threshold. Try increasing restart attempts or increasing temperature scaling on preseed.")
            # label target coordinate as fail, label alternative coordinate as succss
            success_label = 0

            #reset back to best size
            if isinstance(self.basis, CircuitTemplate) or isinstance(self.basis, CircuitTemplateV2):
                self.basis.build(n_repetitions=best_cycles)
            if self.basis.n_qubits == 2:
                alternative_coordinate = c1c2c3(self.basis.eval(best_Xk))
                logging.info(f"Fail: {target_coordinates}, Found: {alternative_coordinate}")
            if self.preseeding:
                self.basis.data_dict[alternative_coordinate] = DataDictEntry(1, 0, best_Xk, best_cycles)

        #save target, update tree
        target_data = DataDictEntry(success_label, best_result, best_Xk, best_cycles)
    
        if self.preseeding:
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

    def cost_target_U(self, target):
        """Atomic cost function - doesn't actually fit 1Q parameters"""
        #logging.info(f"Starting sample iter {index}")
        init_info = self._initialize_run(target_U=target)
        #bad code :)
        #here we check if init function either returned an exact result
        if isinstance(init_info, DataDictEntry):
            init_info = init_info.cycles
        #or if it returned a range, convert to a single value
        else:
            init_info = max(init_info)
        return self.basis.unit_cost(init_info)

    def cost_from_distribution(self, sampler:SampleFunction):
        #use this function if you want to calculate cost over a distribution, but not the entire decomposition fitting
        if not isinstance(self.basis, MixedOrderBasisCircuitTemplate):
            raise ValueError("use customcosttemplate to have defined costs")
        total_cost = 0
        for index, target in enumerate(sampler):
            total_cost += self.cost_target_U(target)
        logging.info(f"Total circuit pulse cost: {total_cost}")
        logging.info(f"Average gate pulse cost: {total_cost/(index+1)}")
        return total_cost

    def approximate_from_distribution(self, sampler:SampleFunction):
        target_data = []
        for index, target in enumerate(sampler):
            logging.info(f"Starting sample iter {index}")
            td = self.approximate_target_U(target_U=target)
            target_data.append(td)
        return self.training_loss, self.coordinate_list, target_data
            
    def _run(self, target_u, target_spanning_range):
        self.ii=0
        def objective_func(xk):

            if isinstance(self.objective, UnitaryCostFunction):
                current_u = self.basis.eval(xk)
                objf_val = self.objective.unitary_fidelity(current_u, target_u)/self.objective.normalization
                
                #optionally, multiply decomposition fidelity objf_val by circuit fidelity
                if isinstance(self.objective, BasicCostInverse):
                    objf_val = 1 - (objf_val * self.basis.circuit_fidelity(xk))

            elif isinstance(self.objective, EntanglementCostFunction):
                current_qc = self.basis.assign_Xk(xk)
                objf_val = self.objective.entanglement_monotone(current_qc)

            elif isinstance(self.objective, LineSegmentDistanceCost):
                current_qc = self.basis.assign_Xk(xk)
                objf_val = self.objective.distance(current_qc)
            else:
                raise ValueError("Unrecognized Cost Function")

            self.objf_val_cache = objf_val
            return objf_val
        
        # callback used to save current loss and coordiante after each iteration
        def callbackF(xk):
            loss = self.objf_val_cache
            temp_training_loss.append(loss)
            if self.basis.n_qubits == 2: #will break for hamiltonianvariationalobjects?
                current_state = self.basis.eval(xk)
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
            
        
            if isinstance(self.basis, CircuitTemplate) or isinstance(self.basis, CircuitTemplateV2):
                self.basis.build(n_repetitions=spanning_iter)
            else:
                #XXX module reload might cause this to fail if basis.py has been changed and not reloaded
                logging.warning("CircuitTemplate type checking failed OR is HamiltonianTemplate(?)")

            #TODO if spanning range == 1, don't pass to optimizer
            #implement a mathematica cloud call?
            
            for r_i in tqdm(range(self.training_restarts)):
                #Constraints definition (only for COBYLA and SLSQP)
                method_str = "BFGS"
                if self.basis.using_bounds:
                    method_str = "L-BFGS-B" 
                    #method_str = "Nelder-Mead" #trying this out to debug
                if self.basis.using_constraints:
                    #method_str = "COBYLA" 
                    # #NOTE cobyla does not support bounds
                    #this is probably fine, but we need to convert the bounds to constraints
                    #just use SLSQP instead
                    method_str = "SLSQP"
                #method_str = "Nelder-Mead" #trying this out to debug
                # if provided a custom optimizer, use that instead
                if self.override_method is not None:
                    method_str = self.override_method

                result = opt.minimize(
                    fun=objective_func,
                    method=method_str,
                    x0=self.basis.parameter_guess(t=r_i),
                    callback=callbackF if self.use_callback else None,
                    options={"maxiter": 2500},
                    bounds=self.basis.bounds_list, #grab bounds list
                    constraints=self.basis.constraint_func #grab constraint function
                )

                # result is good, update temp vars
                if best_result is None or result.fun < best_result:
                    best_result = result.fun
                    best_Xk = result.x
                    best_cycles = spanning_iter
                
                #break over starting attempts
                if best_result < self.success_threshold or (self.override_fail and r_i == self.training_restarts-1):
                    if self.use_callback:
                        self.training_loss.append(temp_training_loss)
                        self.coordinate_list.append(temp_coordinate_list)

                    if best_result < self.success_threshold:
                        break

            logging.info(f"Cycle (k ={spanning_iter}), Best Loss={best_result}")
            
            #break over template extensions
            # already good enough, save time by stopping here
            if best_result < self.success_threshold:
                logging.info(f"Break on cycle {spanning_iter}")
                break

        logging.info(f"Overall Best Loss={best_result}")

        if not self.use_callback:
            #if not saving into a list, just save the last value pass or fail
            self.training_loss.append(best_result)
        
        self.best_cycle_list.append(best_cycles)

        return best_result, best_Xk, best_cycles
