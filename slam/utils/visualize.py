"""Helper functions for plotting"""
import matplotlib.pyplot as plt
import numpy as np
from weylchamber import WeylChamber, c1c2c3
import scipy.spatial as ss
from tqdm import tqdm
from monodromy.coordinates import monodromy_to_positive_canonical_polytope

from config import srcpath
fpath_images = srcpath + "/images"


# pretty print matrix from Chao
def plotMatrix(matrix, rounder=2, vmin=0, vmax=1):
    matrix = np.array(matrix)
    dim = len(matrix)
    nBits = int(np.log2(dim))
    number_label = np.arange(0, dim, 1)
    fig = plt.figure(figsize=(7, 7))
    axs = fig.add_subplot(1, 1, 1)
    pm = plt.imshow(np.abs(matrix), interpolation="nearest", vmin=vmin, vmax=vmax)
    axs.set_xticks(number_label, [f"{n:0{nBits}b}" for n in number_label])
    axs.set_yticks(number_label, [f"{n:0{nBits}b}" for n in number_label])
    for (j, i), label in np.ndenumerate(matrix):
        label = np.round(label, rounder)
        axs.text(i, j, label, ha="center", va="center")
    plt.colorbar(pm, ax=axs)
    # fig.show()
    return fig


def plotHamiltonianSweep(
    matrix, title="Hamiltonian Sweep", labels=None, rounder=2, vmin=0, vmax=1
):
    matrix = np.array(matrix)
    dim1 = len(matrix)
    dim2 = len(matrix[0])
    fig = plt.figure(figsize=(7, 7))
    axs = fig.add_subplot(1, 1, 1)
    pm = axs.imshow(np.abs(matrix), interpolation="nearest", vmin=vmin, vmax=vmax)
    axs.set_title(title)
    n_labels = np.arange(0, dim1, 1)
    m_labels = np.arange(0, dim2, 1)
    if labels is not None:
        axs.set_xticks(m_labels, [labels[0][n][1] for n in m_labels])
        axs.set_yticks(n_labels, [labels[n][0][0] for n in n_labels])
    for (j, i), label in np.ndenumerate(matrix):
        label = np.round(label, rounder)
        axs.text(i, j, label, ha="center", va="center")
    plt.colorbar(pm, ax=axs)
    # fig.show()
    return fig


"""Optimizer plot"""
# self.training_loss, self.coordinate_list
# this are treated like 2d list over set of sampled targets
def optimizer_training_plot(training_loss, coordinate_list, target_str=None, gate_str=None):
    """Plot to show convergence of loss and movement in chamber"""
    plt.close()
    n_samples = len(training_loss)
    # with science, ieee style
    with plt.style.context(['science', 'ieee']):
        fig = plt.figure(figsize=(4, 1.8 * n_samples))
        for index, (sample_loss, sample_coords) in enumerate(
            zip(training_loss, coordinate_list)
        ):
            axs = fig.add_subplot(n_samples, 2, 2 * index + 1)
            training_loss_plot(axs, sample_loss)

            axs = fig.add_subplot(n_samples, 2, 2 * index + 2, projection="3d")
            weyl_training_plot(axs, sample_coords)
            # set the numbber of ticks
            axs.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
            axs.xaxis.set_ticklabels(['0', '', r'$\pi/2$', '', r'$\pi$'])
            axs.yaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            axs.yaxis.set_ticklabels(['0', '', '', '', '', r'$\pi/2$'])
            axs.zaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            axs.zaxis.set_ticklabels(['0', '', '', '', '', r'$\pi/2$'])
        # if target_str is not None and gate_str is not None:
        #     fig.suptitle(f"{gate_str} Training Data, Target: {target_str}")
        # else:
        #     fig.suptitle(f"Training Data (N={len(training_loss)})")
        fig.tight_layout()
        return fig


def training_loss_plot(axs, training_loss):
    c = ["black", "tab:red", "tab:blue", "tab:orange", "tab:green"]
    # deliminate using training loss flags (-1)
    current_index = 0
    x_loss = range(0)
    while True:
        assert training_loss[current_index] == -1
        reps = training_loss[1 + current_index]
        try:
            loss = training_loss[
                2 + current_index : training_loss[2 + current_index :].index(-1)
            ]
        except ValueError:
            loss = training_loss[2 + current_index :]

        x_loss = range(x_loss.stop, x_loss.stop + len(loss))
        axs.plot(
            x_loss,
            loss,
            alpha=0.8,
            # color=c[i %len(c)],
            color=c[reps % len(c)],
            linestyle="-",
            label=reps,
        )
        if -1 not in training_loss[2 + current_index :]:
            break
        current_index += training_loss[2 + current_index :].index(-1) + 2

    # plot horizontal line to show average of final converged value
    # converged_average = np.mean([min(el) for el in training_loss])
    # filter flags (-1)
    converged_average = min([el for el in training_loss if el > 0])
    axs.axhline(converged_average, alpha=0.8, color="tab:gray", linestyle="--")
    axs.text(
        0.5,
        converged_average * 1.01,
        "Best: " + "{:.2E}".format(converged_average),
        {"size": 6},
    )

    axs.set_yscale("log")
    axs.set_xlabel("Training Steps")
    axs.set_ylabel("Training Loss")
    axs.legend(title="Gate Applications")


"""Generic Weyl Chamber plots"""


def weyl_training_plot(axs, coordinate_list, **kwargs):
    w = WeylChamber()
    w.labels = {}
    col = np.arange(len(coordinate_list))
    w.scatter(*zip(*coordinate_list), c=col, **kwargs)
    w.render(axs)


def unitary_2dlist_weyl(*unitary_list, no_bar=0, **kwargs):
    plt.close()
    fig = plt.figure()
    w = WeylChamber()
    w.labels = {}
    axs = fig.add_subplot(111, projection="3d")
    for i, inner_list in enumerate(unitary_list):
        coordinate_list = [c1c2c3(np.array(u)) for u in inner_list]
        if "c" not in kwargs:
            col = ["c", "m", "y", "k", "w"][i % 5]
            sp = w.scatter(*zip(*coordinate_list), c=col, **kwargs)
        else:
            sp = w.scatter(*zip(*coordinate_list), **kwargs)
    w.render(axs)
    if "c" in kwargs and not no_bar:
        fig.colorbar(sp)
    return fig


def coordinate_2dlist_weyl(*coordinate_list, no_bar=0, elev=20, azim=-50, **kwargs):
    plt.close()
    fig = plt.figure()
    w = WeylChamber()
    w.elev = elev
    w.azim = azim
    if (elev, azim) == (90, -90):
        w.show_c3_label = False
    w.labels = {}
    axs = fig.add_subplot(111, projection="3d")
    for i, inner_list in enumerate(coordinate_list):
        if "c" not in kwargs:
            col = ["c", "m", "y", "k", "r"][i % 5]
            sp = w.scatter(*zip(*inner_list), c=col, **kwargs)
        else:
            sp = axs.scatter3D(*zip(*inner_list), **kwargs)

    w.render(axs)
    if "c" in kwargs and not no_bar:
        # make colorbar smaller
        fig.colorbar(sp, shrink=0.4)
    return fig


def unitary_to_weyl(*unitary):
    plt.close()
    fig = plt.figure()
    w = WeylChamber()
    axs = fig.add_subplot(111, projection="3d")
    for u in unitary:
        w.add_point(*c1c2c3(np.array(u)))
    w.render(axs)
    return fig


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
