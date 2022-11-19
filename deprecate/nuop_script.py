# %%
from custom_gates import *
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from itertools import cycle
from weylchamber import c1c2c3
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info import Operator, random_unitary, random_clifford
import scipy.optimize as opt
import hashlib


# %%
import matplotlib.pyplot as plt

# plt.style.use(["science", "ieee"])
# plt.plot([0,0], [1,1]);
# plt.style.use(["science", "ieee"])
# I'm not sure why but the styles don't get updated until after running twice, so monkey fix like this??

# %%
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# %%
import h5py


def h5py_load(filekey, *args):
    filename = f"data/{filekey}.h5"
    results = {}
    try:
        with h5py.File(filename, "r") as h5f:
            for arg in args:
                results[arg] = h5f[arg][:]
        return results
    except FileNotFoundError:
        logging.debug(f"Failed to load {filename}")
        return None


def h5py_save(filekey, **kwargs):
    filename = f"data/{filekey}.h5"
    with h5py.File(filename, "a") as h5f:
        for key, value in kwargs.items():
            try:
                del h5f[key]
            except Exception:
                pass
            h5f.create_dataset(key, data=value)
    logging.debug(f"Successfully saved to {filename}")


# we need some helper method to fix ragged arrays from training loss data
# previously, I was padding values with -1, so data points could easily be deleted later,
# I think it makes more sense to pad with the last remaining value, since training converges it just sits at that value
# TODO: could rewrite for so functional handles deepcopying rather than caller like as is currently
def rag_to_pad(arr):
    max_len = max(len(arr[i]) for i in range(len(arr)))
    for i in range(len(arr)):
        temp_len = len(arr[i])
        for j in range(max_len):
            if j >= temp_len:
                if j == 0:
                    raise ValueError("cant extend blank row")
                arr[i].append(arr[i][j - 1])
                # arr[i].append(-1)
    return np.array(arr)


# rewrite convert to ragged array by detecting when row is being extended
def pad_to_rag(arr):
    arr = arr.tolist()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            # if arr[i][j] == -1:
            if arr[i][j] == arr[i][j - 1]:
                arr[i] = arr[i][0:j]
                break
    return arr


# test
# arr = [[1, 2, 3, 4], [1, 2], [1, 2, 3]]
# arr = rag_to_pad(arr)
# print(arr)
# arr = pad_to_rag(arr)
# print(arr)

# %%
class TemplateCircuit:
    def __init__(
        self,
        n_qubits=2,
        base_gate_class=[RiSwapGate],
        gate_2q_params=[1 / 2],
        edge_params=[(0, 1)],
        trotter=False,
    ):
        """Initalizes a qiskit.quantumCircuit object with unbound 1Q gate parameters
        Args:
            n_qubits: size of target unitary,
            base_gate_class: Gate class of 2Q gate,
            gate_2q_params: List of params to define template gate cycle sequence
            edge_params: List of edges to define topology cycle sequence
            trotter: if true, only use gate_2q_params[0], override cycle length and edge_params, each 1Q gate share parameters per qubit row
        """
        self.hash = (
            str(n_qubits)
            + str(base_gate_class)
            + str(gate_2q_params)
            + str(edge_params)
            + str(trotter)
        )

        if n_qubits != 2 and trotter:
            raise NotImplementedError
        self.n_qubits = n_qubits
        self.trotter = trotter
        self.circuit = QuantumCircuit(n_qubits)
        self.gate_2q_base = base_gate_class

        self.cycles = 0

        if self.trotter:
            # raise NotImplementedError
            logging.warning("Trotter may not work as intended")

        # else:
        self.gate_2q_base = cycle(base_gate_class)
        self.gate_2q_params = cycle(gate_2q_params)
        self.gate_2q_edges = cycle(edge_params)
        self.cycle_length = max(len(gate_2q_params), len(edge_params))

        self.gen_1q_params = self._param_iter()

    # def __str__(self):
    #     s = ""
    #     for param in self.gate_2q_params:
    #         s += self.gate_2q_base.latex_string(param)
    #     return s

    def build(self, n_repetitions):
        self._reset()
        if self.trotter:
            pass
            # n_repetitions = int(1 / next(self.gate_2q_params))
        for _ in range(n_repetitions - 1):
            self._build_cycle()

    def _reset(self):
        """Return template to a single cycle"""
        self.cycles = 0
        self.circuit = QuantumCircuit(self.n_qubits)
        self._build_cycle(initial=True)

    def initial_guess(self):
        """returns a np array of random values for each parameter"""
        return np.random.random(len(self.circuit.parameters)) * 2 * np.pi

    def assign_Xk(self, Xk):

        return self.circuit.assign_parameters(
            {parameter: i for parameter, i in zip(self.circuit.parameters, Xk)}
        )

    def eval(self, Xk):
        """returns an Operator after binding parameter array to template"""
        return Operator(self.assign_Xk(Xk)).data

    def _param_iter(self):
        index = 0
        while True:
            # Check if Parameter already created, then return reference to that variable
            def _filter_param(param):
                return param.name == f"P{index}"

            res = list(filter(_filter_param, self.circuit.parameters))
            if len(res) == 0:
                yield Parameter(f"P{index}")
            else:
                yield res[0]
            index += 1
            if self.trotter:
                index %= 3 * self.n_qubits

    def _build_cycle(self, initial=False):
        """Extends tempalte by one full cycle"""
        if initial:
            # before build by extend, add first pair of 1Qs
            for qubit in range(self.n_qubits):
                self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
        for _ in range(self.cycle_length):
            edge = next(self.gate_2q_edges)
            self.circuit.append(
                next(self.gate_2q_base)(next(self.gate_2q_params)), edge
            )
            for qubit in edge:
                self.circuit.u(*[next(self.gen_1q_params) for _ in range(3)], qubit)
        self.cycles += 1


# %%
# a = TemplateCircuit(
#     gate_2q_params=[1 /3, 1 / 3], n_qubits=2, edge_params=[(0, 1), (0, 2), (1, 2)], trotter=True
# )
# a.build(2)
# a.circuit.draw(output="mpl")


# %%
class TemplateOptimizer:
    def __init__(
        self,
        template,
        objective_function_name="basic",
        unitary_sample_function="Haar",
        n_samples=1,
        template_iter_range=range(2, 4),
        thread_id=0,
        no_save=False,
    ):
        """Args:
        template: TemplateCircuit object
        objective_function_name: "nuop|basic" or "weyl"
        unitary_sample_function: "Haar" or "Clifford for random sampling, "SWAP", "CNOT", "iSWAP" for single gates
        n_samples: the number of times to sample a gate and minimize template on
        template_iter_range: a range() object that whos values are passed to template.build()
        """
        self.template = template
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
        self.thread_id = thread_id
        self.no_save = no_save

    def run(self, override_saved=False):
        # first attempt to load n_samples from data
        offset = 0
        results = None
        if not override_saved and not self.no_save:
            results = h5py_load(self.filekey, "training_loss", "training_reps")
            if results is not None:
                self.training_loss = pad_to_rag(results["training_loss"])
                self.training_reps = results["training_reps"].tolist()
                offset = len(self.training_loss)
                logging.info(f"Thread{self.thread_id}: Loaded {offset} samples")

        # run minimize on the remaining samples
        # use try finally, so if end early stills writes back what it has thus far
        # XXX I'm not sure this actually works, dont rely on it
        best_result, best_Xk, best_cycles = None, None, None
        try:
            for i in range(offset, self.n_samples):
                logging.info(f"Thread{self.thread_id}: Starting sample iter {i}")
                self.training_loss.append([])
                target_unitary = self.sampler()
                obj = self._objective_function(self.obj_f_name, target_unitary)
                best_result, best_Xk, best_cycles = self.minimize(
                    obj=obj, iter=i, t_range=self.template_iter_range
                )
        finally:
            # finally, save again
            if self.n_samples > offset and not self.no_save:
                # use list comprehension to pass by value (deepcopy)
                h5py_save(
                    self.filekey,
                    training_loss=rag_to_pad([row[:] for row in self.training_loss]),
                    training_reps=self.training_reps,
                )
        return best_result, best_Xk, best_cycles

    def minimize(self, obj, iter, t_range):
        # NOTE: potential for speedup?
        # you can calculate ahead of time the number of repetitions needed using traces??

        # callback used to save current loss after each iteration
        def callbackF(xk):
            loss = obj(xk)
            temp_training_loss.append(loss)

        best_result = None
        best_Xk = None
        best_cycles = -1

        # each t creates fresh template with new repetition param
        for t in t_range:
            logging.info(f"Thread{self.thread_id}: Starting cycle length {t}")
            temp_training_loss = []
            self.template.build(n_repetitions=t)

            result = opt.minimize(
                fun=obj,
                x0=self.template.initial_guess(),
                callback=None if self.no_save else callbackF,
                options={"maxiter": 200},
            )

            # result is good, update temp vars
            if best_result is None or result.fun < best_result:
                best_result = result.fun
                best_Xk = result.x
                best_cycles = self.template.cycles
                self.training_loss[iter] = temp_training_loss

            # already good enough, save time by stopping here
            if best_result < 1e-9:
                logging.info(f"Thread{self.thread_id}: Break on cycle {t}")
                break

        logging.info(f"Thread{self.thread_id}: loss= {best_result}")
        self.training_reps.append(best_cycles)
        return best_result, best_Xk, best_cycles

    def _sample_function(self, name):
        if name == "CParitySwap":
            return lambda: CParitySwap()
        if name == "CiSWAP":
            return lambda: np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1j, 0],
                    [0, 0, 0, 0, 0, 1j, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        if name == "CCiX":
            return lambda: np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1j],
                    [0, 0, 0, 0, 0, 0, 1j, 0],
                ]
            )
        if name == "Peres":
            return lambda: Peres()
        if name == "Toffoli":
            return lambda: CCXGate()
        if name == "CSWAP":
            return lambda: CSwapGate()
        if name == "RCCXGate":
            return lambda: RCCXGate()
        if name == "SWAP":
            return lambda: SwapGate()
        if name == "CNOT":
            return lambda: CXGate()
        if name == "iSWAP":
            return lambda: iSwapGate()
        if name == "Haar":
            return lambda: random_unitary(dims=2**self.template.n_qubits).data
        if name == "Clifford":
            return lambda: Operator(
                random_clifford(num_qubits=self.template.n_qubits)
            ).data
        else:
            raise ValueError(f"No sample function named {name}")

    def _objective_function(self, name, target):
        if name == "basic":
            return (
                lambda X: 1
                - np.abs(
                    np.trace(np.matmul(np.matrix(target).getH(), self.template.eval(X)))
                )
                / np.array(target).shape[0]
            )
        if name == "nuop":
            return lambda X: 1 - np.abs(
                np.sum(np.multiply(self.template.eval(X), np.conj(target)))
            ) / (2 * self.template.n_qubits)
        if name == "weyl":
            if self.template.n_qubits != 2:
                raise ValueError("Weyl chamber only for 2Q gates")
            return lambda X: np.linalg.norm(
                np.array(c1c2c3(target)) - np.array(c1c2c3(self.template.eval(X)))
            )
        else:
            raise ValueError(f"No objective function named {name}")

    @staticmethod
    def plot(fig_title, *optimizers):
        c = ["black", "tab:red", "tab:blue", "tab:orange", "tab:green"]

        # each optimizer is its own column subplot
        fig, axs = plt.subplots(1, len(optimizers), sharey=True, squeeze=False)
        for ax_index, optimizer in enumerate(optimizers):

            # each sample gets plotted as a faint line
            for i in range(optimizer.n_samples):
                axs[0][ax_index].plot(
                    optimizer.training_loss[i],
                    alpha=0.2,
                    color=c[optimizer.training_reps[i] % len(c)],
                    linestyle="-",
                )

            # plot horizontal line to show average of final converged value
            converged_averaged = np.mean([el[-1] for el in optimizer.training_loss])
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
            for reps in set(optimizer.training_reps):
                # filter training data for each rep value
                temp = [
                    optimizer.training_loss[i]
                    for i in range(len(optimizer.training_reps))
                    if optimizer.training_reps[i] == reps
                ]

                # construct average over points where data exists
                # uses row-col outer-inner loop over points, increments counter k for norming
                temp_average = []
                for i in range(max([len(el) for el in temp])):
                    temp_average.append(0)
                    k = 0
                    for j in range(len(temp)):
                        if i < len(temp[j]):
                            temp_average[i] += temp[j][i]
                            k += 1
                    temp_average[i] /= k

                # plot average with full color
                axs[0][ax_index].plot(
                    temp_average,
                    color=c[reps % len(c)],
                    label=f"L{reps}",
                    linestyle="-",
                )

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


# %%
# TODO: it would be nice to improve the random selection, perhaps just structure into all permutations instead or is random preferred?
import random
import multiprocessing

# gate_list = [CSwapGate, CCXGate, CXGate, SwapGate, iSwapGate, CSwapGate, Peres]
gate_list = [CParitySwap, SwapGate, iSwapGate, CXGate]
_3qedge = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
_2qedge = [(0, 1), (0, 2), (1, 2)]
edge_dict = {
    CParitySwap: _3qedge,
    CCXGate: _3qedge,
    CSwapGate: _3qedge,
    CXGate: _2qedge,
    SwapGate: _2qedge,
    iSwapGate: _2qedge,
}

best_result = None
best_template = None
best_Xk = None


def random_template_test(N):
    X = random.choice(range(5))
    base_gate_class = [random.choice(gate_list) for _ in range(X)]
    # base_gate_class = [CSwapGate, CSwapGate]
    edge_params = [random.choice(edge_dict[gate]) for gate in base_gate_class]
    template = TemplateCircuit(
        n_qubits=3,
        base_gate_class=base_gate_class,
        gate_2q_params=[None],
        edge_params=edge_params,
    )
    optimizer = TemplateOptimizer(
        template,
        n_samples=1,
        unitary_sample_function="Toffoli",
        template_iter_range=range(1, 3),
        thread_id=N,
        no_save=True,
    )
    # unitary_sample_function="CParitySwap"

    result, Xk, cycles = optimizer.run(override_saved=True)
    # print(f"thread {N}: {result}")
    return (result, base_gate_class, edge_params, Xk, cycles)


if __name__ == "__main__":
    multi_thread_repeat_count = 8  # 256
    # pool_obj = multiprocessing.Pool()
    with multiprocessing.Pool() as p:
        answer = p.map(random_template_test, range(0, multi_thread_repeat_count))

    s = sorted(answer, key=lambda x: 1 if (x is None or x[0] is None) else x[0])
    from sys import stdout

    for i in range(len(s)):
        if s[i] is None:
            continue
        best_result, best_base_gate_class, best_edge_params, best_Xk, best_cycles = s[i]
        if best_result < 1e-9:
            print(best_result)
            best_template = TemplateCircuit(
                n_qubits=3,
                base_gate_class=best_base_gate_class,
                gate_2q_params=[None],
                edge_params=best_edge_params,
            )
            result_circuit = best_template.build(best_cycles)
            result_circuit = best_template.assign_Xk(best_Xk)

            print(result_circuit.draw(output="text"))
            stdout.flush()
            print("\n")

# 4.6132964115486175e-10
#      ┌─────────────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐   »
# q_0: ┤ U(3.3844933176244e-5,4.99847687552002,2.32226265525574) ├─■─┤ U(-0.236649025804495,4.85080931843722,0.496397548490396) ├─X──┤ U(6.44048454723241,4.0519759434474,5.24972799609676) ├─X─»
#      └┬────────────────────────────────────────────────────────┤ │ └┬───────────────────────────────────────────────────────┬─┘ │ ┌┴──────────────────────────────────────────────────────┤ │ »
# q_1: ─┤ U(4.5203206947232,-0.522854661056527,2.32362597424714) ├─X──┤ U(3.50423341864694,3.33043628836768,5.10392352915538) ├───X─┤ U(4.14157575779863,1.68921366580331,4.43761289929974) ├─X─»
#       ├───────────────────────────────────────────────────────┬┘ │  └┬──────────────────────────────────────────────────────┤     └───────────────────────────────────────────────────────┘ │ »
# q_2: ─┤ U(4.90444009678939,2.61872853629591,4.80631456307294) ├──X───┤ U(4.90445709272882,1.8779730062686,6.80601822515001) ├───────────────────────────────────────────────────────────────■─»
#       └───────────────────────────────────────────────────────┘      └──────────────────────────────────────────────────────┘                                                                 »
# «      ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────────┐      ┌───────────────────────────────────────────────────────┐
# «q_0: ─┤ U(4.3762952179078,6.54446607670712,1.4186393042608) ├───■─┤ U(0.669920777050112,4.84148094495864,0.765452677249713) ├─X────┤ U(2.47168562598555,4.73401437016179,7.72481031885594) ├───
# «     ┌┴─────────────────────────────────────────────────────┴─┐ │ └┬────────────────────────────────────────────────────────┤ │   ┌┴───────────────────────────────────────────────────────┴─┐
# «q_1: ┤ U(5.26866415590355,0.173658605067415,5.05175922804841) ├─X──┤ U(1.04497157241386,-1.00879649263508,2.32349255015102) ├─X───┤ U(-0.669924803759966,0.702408635644935,1.44166739927367) ├─
# «     ├───────────────────────────────────────────────────────┬┘ │  ├───────────────────────────────────────────────────────┬┘ │ ┌─┴──────────────────────────────────────────────────────────┴┐
# «q_2: ┤ U(6.77611253710733,1.40092229376439,1.16442814103968) ├──X──┤ U(3.63452417110915,2.99150402857734,1.74066115767031) ├──■─┤ U(-1.73954442248929e-6,4.34363215680226,-0.333178980957175) ├
# «     └───────────────────────────────────────────────────────┘     └───────────────────────────────────────────────────────┘    └─────────────────────────────────────────────────────────────┘
