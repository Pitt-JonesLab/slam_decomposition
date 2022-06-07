import matplotlib.pyplot as plt
import numpy as np
from weylchamber import WeylChamber, c1c2c3

"""Helper functions for plotting"""

# pretty print matrix from Chao
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
 #TODO rewrite
 # self.training_loss, self.training_reps, self.coordinate_list
 # this are treated like 2d list over set of sampled targets

def optimizer_training_plot(training_loss, training_reps, coordinate_list):
    """Plot to show convergence of loss and movement in chamber"""
    plt.close()
    fig = plt.figure()
    weyl_training_plot(fig, coordinate_list)
    training_loss_plot(fig, training_loss, training_reps)
    fig.suptitle(f"Convergence Data (N={len(training_loss)})")
    return fig
  

def training_loss_plot(fig, training_loss, training_reps):
    axs = fig.add_subplot(121)
    c = ["black", "tab:red", "tab:blue", "tab:orange", "tab:green"]

    # each sample gets plotted as a faint line
    for sample_loss,sample_reps in zip(training_loss, training_reps):
        axs.plot(
            sample_loss,
            alpha=0.8,
            #color=c[i %len(c)],
            color=c[sample_reps% len(c)],
            linestyle="-",
            label=sample_reps
        )

    # plot horizontal line to show average of final converged value
    converged_averaged = np.mean([min(el) for el in training_loss])
    axs.axhline(
        converged_averaged, alpha=0.8, color="tab:gray", linestyle="--"
    )
    axs.text(
        0.5,
        converged_averaged * 1.01,
        "Avg: " + "{:.2E}".format(converged_averaged),
        {"size": 5},
    )

    axs.set_yscale("log")
    axs.set_xlabel("Training Steps")
    axs.set_ylabel("Training Loss")
    axs.legend()

"""Generic Weyl Chamber plots"""

# I have a lot of versions of this code plotting around
# TODO rewrite to be usable generically

def weyl_training_plot(fig, coordinate_list):
    ax = fig.add_subplot(122, projection="3d")
    w = WeylChamber();
    for sample in coordinate_list:
        col = np.arange(len(sample))
        w.scatter(*zip(*sample), c=col)
    w.render(ax)


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
