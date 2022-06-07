import logging
import h5py
SUCCESS_THRESHOLD = 1e-9


#TODO: rewrite how sampling works
# we don't need to save training data/weyl walks to files, but keep it in a variable for plotting?

"""
Given a gate basis objects finds parameters which minimize cost function
"""

class TemplateOptimizer:
    def __init__(
        self,
        objective_function_name="basic",
        unitary_sample_function="Haar",
        n_samples=1,
        template_iter_range=range(2, 4),
        no_save=True
    ):
        """Args:
        template: TemplateCircuit object
        objective_function_name: "basic" or "square"
        unitary_sample_function: "Haar" or "Clifford for random sampling, "SWAP", "CNOT", "iSWAP" for single gates
        n_samples: the number of times to sample a gate and minimize template on
        template_iter_range: a range() object that whos values are passed to template.build()
        """
        self.sampler = self._sample_function(unitary_sample_function)
        self.n_samples = n_samples
        self.obj_f_name = objective_function_name
        self.template_iter_range = template_iter_range
        self.filekey = hashlib.sha1(
            (
                self.template.hash
                + str(objective_function_name)
                + str(unitary_sample_function)
                + str(template_iter_range)
            ).encode()
        ).hexdigest()
        self.plot_title = None
        self.training_loss = []
        self.training_reps = []
        self.no_save=no_save

    from basis import VariationalObject
    def approximate_target_U(self, basis:VariationalObject, target_U):

        target_coordinates = basis.target_invariant(target_U)

        if self.preseeding:
            #check if coordinate already exists in loaded_data
            distance, index = basis.coordinate_tree.query([target_coordinates])

            #if target_coordinates in self.data_dict.keys():
            if distance == 0:
                logging.info(f"Found saved: {target_coordinates}")
                return basis.data_dict.get(target_coordinates)
        else:
            starting_guess = None
        logging.info(f"SEARCHING: {target_coordinates}")
        starting_guess = basis.data_dict[index]
        best_result, best_Xk, best_cycles = self.run(target_U, seed_Xk=starting_guess)

        if best_result <= SUCCESS_THRESHOLD:
            # label target coordinate as success
            label = 1
            logging.info(f"Success: {target_coordinates}")
        else:
            # label target coordinate as fail, label alternative coordinate as succss
            label = 0
            alternative_coordinate = weylchamber.c1c2c3(basis.eval(best_Xk))
            logging.info(f"Fail: {target_coordinates}, Alternative: {alternative_coordinate}")
            basis.data_dict[alternative_coordinate] = (1, 0) #, best_Xk, best_cycles)
            
    def approximate_from_distribution(self):
        for i in range(offset, self.n_samples):
            logging.info(f"Starting sample iter {i}")
            self.training_loss.append([])
            target_unitary = self.sampler()
            obj = self._objective_function(self.obj_f_name, target_unitary)
            best_result, best_Xk, best_cycles = self.minimize(
                obj=obj, iter=i, t_range=self.template_iter_range
            )

    def minimize(self, obj, iter, t_range):
        # NOTE: potential for speedup?
        # you can calculate ahead of time the number of repetitions needed using traces??

        # callback used to save current loss after each iteration
        # can also be used to save current coordinate
        def callbackF(xk):
            loss = obj(xk)
            temp_training_loss.append(loss)
            gate = self.template.eval(xk)
            c1, c2, c3 = weylchamber.c1c2c3(gate)
            self.coordinate_list.append((c1,c2,c3))

        best_result = None
        best_Xk = None
        best_cycles = -1

        # each t creates fresh template with new repetition param
        for t in t_range:
            logging.info(f"Starting cycle length {t}")
            temp_training_loss = []
            self.template.build(n_repetitions=t)

            #TODO can do this in threads?
            starting_attempts = 5
            for _ in range(starting_attempts):

                else: #self.obj_f_name == "basic" or self.obj_f_name == "square"
                    result = opt.minimize(
                        fun=obj,
                        method='BFGS',#'nelder-mead',
                        x0=self.template.initial_guess(),
                        callback=callbackF,
                        options={"maxiter": 200},
                    )
                    # result = opt.minimize(
                    #     fun=obj,
                    #     method='BFGS',
                    #     x0=result.x,
                    #     callback=callbackF,
                    #     options={"maxiter": 1000},
                    # )
                

                # result is good, update temp vars
                if best_result is None or result.fun < best_result:
                    best_result = result.fun
                    best_Xk = result.x
                    best_cycles = self.template.cycles
                    self.training_loss[iter] = temp_training_loss
                
                #break over starting attempts
                if best_result < 1e-9:
                    break
            
            #break over template extensions
            # already good enough, save time by stopping here
            if best_result < 1e-9:
                logging.info(f"Break on cycle {t}")
                break

        logging.info(f"loss= {best_result}")
        self.training_reps.append(best_cycles)
        return best_result, best_Xk, best_cycles

    def _sample_function(self, name):
        #XXX refactored

    def _objective_function(self, name, target):
        #XXX refactored

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
