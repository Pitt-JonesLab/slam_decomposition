import matplotlib.pyplot as plt
import numpy as np
from weylchamber import WeylChamber, c1c2c3

"""Helper functions for plotting"""

# pretty print matrix from Chao
#FIXME use figures and return
def plotMatrix(matrix, rounder=2, vmin=0, vmax=1):
    matrix = np.array(matrix)
    dim = len(matrix)
    nBits = int(np.log2(dim))
    number_label = np.arange(0, dim, 1)
    fig = plt.figure(figsize=(7, 7))
    plt.subplot(1, 1, 1)
    pm = plt.imshow(np.abs(matrix), interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.xticks(number_label, [f"{n:0{nBits}b}" for n in number_label])
    plt.yticks(number_label, [f"{n:0{nBits}b}" for n in number_label])
    for (j, i), label in np.ndenumerate(matrix):
        label = np.round(label, rounder)
        plt.text(i, j, label, ha="center", va="center")
    plt.colorbar()
    plt.show()
    return pm

"""Optimizer plot"""
 # self.training_loss, self.coordinate_list
 # this are treated like 2d list over set of sampled targets
#FIXME adaptive figure size
def optimizer_training_plot(training_loss, coordinate_list):
    """Plot to show convergence of loss and movement in chamber"""
    plt.close()
    n_samples = len(training_loss)
    fig = plt.figure(figsize=(6,4*n_samples))
    for index, (sample_loss, sample_coords) in enumerate(zip(training_loss, coordinate_list)):
        axs = fig.add_subplot(n_samples,2,2*index+1)
        training_loss_plot(axs, sample_loss)

        axs= fig.add_subplot(n_samples,2,2*index+2, projection="3d")
        weyl_training_plot(axs, sample_coords)
    fig.suptitle(f"Convergence Data (N={len(training_loss)})")
    return fig
  

def training_loss_plot(axs, training_loss):
    c = ["black", "tab:red", "tab:blue", "tab:orange", "tab:green"]
    #deliminate using training loss flags (-1)
    current_index = 0
    x_loss = range(0)
    while True:
        assert training_loss[current_index] == -1
        reps = training_loss[1+current_index]
        try:
            loss = training_loss[2+current_index:training_loss[2+current_index:].index(-1)]
        except ValueError:
            loss = training_loss[2+current_index:] 

        x_loss = range(x_loss.stop, x_loss.stop + len(loss))
        axs.plot(
            x_loss,
            loss,
            alpha=0.8,
            #color=c[i %len(c)],
            color=c[reps% len(c)],
            linestyle="-",
            label=reps
        )
        if -1 not in training_loss[2+current_index:]:
            break
        current_index += training_loss[2+current_index:].index(-1) + 2

    # plot horizontal line to show average of final converged value
    #converged_average = np.mean([min(el) for el in training_loss])
    #filter flags (-1)
    converged_average = min([el for el in training_loss if el >= 0])
    axs.axhline(
        converged_average, alpha=0.8, color="tab:gray", linestyle="--"
    )
    axs.text(
        0.5,
        converged_average * 1.01,
        "Avg: " + "{:.2E}".format(converged_average),
        {"size": 5},
    )

    axs.set_yscale("log")
    axs.set_xlabel("Training Steps")
    axs.set_ylabel("Training Loss")
    axs.legend()

"""Generic Weyl Chamber plots"""

# I have a lot of versions of this code plotting around
# TODO rewrite to be usable generically

def weyl_training_plot(axs, coordinate_list):
    w = WeylChamber();
    col = np.arange(len(coordinate_list))
    w.scatter(*zip(*coordinate_list), c=col)
    w.render(axs)


# from weylchamber import WeylChamber
# def weyl_plot():
#     w = WeylChamber()
#     _, coordinate_list = build_time_dependent_U(N, result.x)
#     col = np.arange(len(coordinate_list))
#     w.scatter(*zip(*coordinate_list), c=col, marker="o")
#     w.scatter(*c1c2c3(swap_gate_target), marker="x")
#     w.plot()

##############################################################

# TODO wrap in a function, can be integrated with above
"""monoromy rendering"""
# %matplotlib widget

# import matplotlib.pyplot as plt
# plt.close()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# from weylchamber import WeylChamber
# w = WeylChamber();

# total_coord_list = []
# for subpoly in reduced_vertices:
#     subpoly_coords = [[float(x) for x in coord] for coord in subpoly]
#     total_coord_list += subpoly_coords
#     w.scatter(*zip(*subpoly_coords))

# from scipy.spatial import ConvexHull
# pts = np.array(total_coord_list)
# hull = ConvexHull(pts)
# for s in hull.simplices:
#     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
#     ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

# w.render(ax)
