import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from weylchamber import c1c2c3

from basis import CircuitTemplate, DataDictEntry, VariationalTemplate
from cost_function import UnitaryCostFunction
from sampler import SampleFunction

SUCCESS_THRESHOLD = 1e-9
TRAINING_RESTARTS = 5
"""
Given a gate basis objects finds parameters which minimize cost function
"""
class TemplateOptimizer:
    def __init__(self, basis:VariationalTemplate, objective:UnitaryCostFunction, use_callback=False):
        self.basis = basis
        self.objective = objective

        self.use_callback = use_callback
        self.sample_iter = 0
        self.training_loss = [] #2d list sample_iter -> [training iter -> loss]
        self.training_reps = [] #1d list sample_iter -> best result cycle length
        self.coordinate_list = [] #2d list sample_iter -> [training iter -> (coordinate)]

    #TODO, investigate, when does basis data_dict get updated, when do I need to call save to file
    def approximate_target_U(self, target_U):

        target_coordinates = self.basis.target_invariant(target_U)

        if self.preseeding:
            #check if coordinate already exists in loaded_data
            distance, index = self.basis.coordinate_tree.query([target_coordinates])
            found_saved = self.basis.coordinate_tree.data[index]
            
            #if target_coordinates in self.data_dict.keys():
            if distance == 0 and found_saved.success_label:
                logging.info(f"Found saved: {target_coordinates}")
                return found_saved

            #convert key to guess vector
            starting_guess = found_saved.Xk
        else:
            starting_guess = None
        
        logging.info(f"SEARCHING: {target_coordinates}")
        best_result, best_Xk, best_cycles = self.run(target_U, seed_Xk=starting_guess)

        if best_result <= SUCCESS_THRESHOLD:
            # label target coordinate as success
            success_label = 1
            logging.info(f"Success: {target_coordinates}")
        else:
            # label target coordinate as fail, label alternative coordinate as succss
            success_label = 0

            if isinstance(self.basis, CircuitTemplate):
                self.basis.build(n_repetitions=best_cycles)
            alternative_coordinate = c1c2c3(self.basis.eval(best_Xk))
    
            logging.info(f"Fail: {target_coordinates}, Alternative: {alternative_coordinate}")
            self.basis.data_dict[alternative_coordinate] = DataDictEntry(1, 0, best_Xk, best_cycles)

        #save target
        target_data = DataDictEntry(success_label, best_result, best_Xk, best_cycles)
        self.basis.data_dict[target_coordinates] = target_data
        return target_data

    def approximate_from_distribution(self, sampler:SampleFunction):
        for target in sampler:
            logging.info(f"Starting sample iter {self.sample_iter}")
            self.approximate_target_U(target_U=target)
            self.sample_iter += 1
            
    def run(self, target_u, seed_Xk):
        
        objective_func = self.objective.fidelity_lambda(target_u)
        
        # callback used to save current loss and coordiante after each iteration
        def callbackF(xk):
            current_state = self.basis.eval(xk)
            loss = objective_func(current_state)
            temp_training_loss.append(loss)
            temp_coordinate_list.append(c1c2c3(current_state))

        best_result = None
        best_Xk = None
        best_cycles = -1

        # each t creates fresh template with new repetition param
        for spanning_iter in self.basis.get_spanning_range():
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
                    if self.use_callback:
                        self.training_loss.append(temp_training_loss)
                        self.coordinate_list.append(temp_coordinate_list)
                
                #break over starting attempts
                if best_result < SUCCESS_THRESHOLD:
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


    #TODO
    @staticmethod
    def plot(fig_title, *optimizers):
        #NOTE: previous version used different colors to signal different template lengths
        #now I am changing so different colors correspond to different Haar samples, assuming now template size is always fixed
        c = ["black", "tab:red", "tab:blue", "tab:orange", "tab:green"]

        # each optimizer is its own column subplot
        fig, axs = plt.subplots(1, len(optimizers), sharey=True, squeeze=False)
        for ax_index, optimizer in enumerate(optimizers):

            # each sample gets plotted as a faint line
            for i in range(optimizer.n_samples):
                axs[0][ax_index].plot(
                    optimizer.training_loss[i],
                    alpha=0.2,
                    color=c[i %len(c)],
                    #color=c[optimizer.training_reps[i] % len(c)],
                    linestyle="-",
                )

            # plot horizontal line to show average of final converged value
            converged_averaged = np.mean([min(el) for el in optimizer.training_loss])
            axs[0][ax_index].axhline(
                converged_averaged, alpha=0.8, color="tab:gray", linestyle="--"
            )
            axs[0][ax_index].text(
                0.5,
                converged_averaged * 1.01,
                "Avg: " + "{:.2E}".format(converged_averaged),
                {"size": 5},
            )

            # custom average for ragged array
            # XXX there must be a smart way to do this I couldn't think of it at the time I wrote this :(
            # for reps in set(optimizer.training_reps):
            #     # filter training data for each rep value
            #     temp = [
            #         optimizer.training_loss[i]
            #         for i in range(len(optimizer.training_reps))
            #         if optimizer.training_reps[i] == reps
            #     ]

            #     # construct average over points where data exists
            #     # uses row-col outer-inner loop over points, increments counter k for norming
            #     temp_average = []
            #     for i in range(max([len(el) for el in temp])):
            #         temp_average.append(0)
            #         k = 0
            #         for j in range(len(temp)):
            #             if i < len(temp[j]):
            #                 temp_average[i] += temp[j][i]
            #                 k += 1
            #         temp_average[i] /= k

            #     # plot average with full color
            #     axs[0][ax_index].plot(
            #         temp_average,
            #         color='black',
            #         #color=c[reps % len(c)],
            #         label=f"L{reps}",
            #         linestyle="dashed",
            #     )

            axs[0][ax_index].set_yscale("log")
            axs[0][ax_index].set_xlabel("Training Steps")
            axs[0][ax_index].set_title(f"{optimizer.plot_title}")
            axs[0][ax_index].legend()

        fig.suptitle(f"{fig_title}, (N={optimizers[0].n_samples})", y=0.92)
        axs[0][0].set_ylabel("Training Loss")
        fig.tight_layout()
        fig.show()


# test objective function
# n = TemplateCircuit(n_qubits=3)
# a = TemplateOptimizer(n)._objective_function(name="nuop", target=Operator(CCXGate()).data)
# a(Operator(CCXGate()).data)
