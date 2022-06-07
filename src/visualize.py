import matplotlib.pyplot as plt
import numpy as np

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


"""Generic Weyl Chamber plots"""

# I have a lot of versions of this code plotting around
# TODO rewrite to be usable generically

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
