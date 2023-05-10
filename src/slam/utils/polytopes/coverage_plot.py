"""Helper functions for plotting"""
import matplotlib.pyplot as plt
import numpy as np
from weylchamber import WeylChamber, c1c2c3
import scipy.spatial as ss
from tqdm import tqdm
from monodromy.coordinates import monodromy_to_positive_canonical_polytope

from config import srcpath
fpath_images = f"{srcpath}/images"


def plot_coverage_set(coverage_set, save=False, filename=None, **kwargs):
    """NOTE Forcing to print subpoly hulls for now until I figure out convex hull intersection
    this is nontrivial because the intersection creates concave polyhedra, i.e. the gap between 2 intersecting pyramids"""
    plt.close()
    fig = None
    # skip the first convex polytope in the set, which is the identity
    for k, circuit_polytope in enumerate(
        coverage_set[1:][::-1]
    ):  # reversed so inner polytopes are on top
        # recalc the index becuase of the reversed order
        index = len(coverage_set[1:]) - k - 1
        fig = _plot_circuit_polytope(
            circuit_polytope, fig=fig, index=index, save=False
        )  # , override_connect=index in kwargs.get("override_connect_indices", []))
    if save and filename is not None:
        plt.savefig(f"{fpath_images}/{filename}.pdf", format="pdf")
    plt.show()
    return fig


def _plot_circuit_polytope(
    circuit_poly, fig=None, index=0, save=False, filename=None, override_connect=True
):
    """If override_connect is True, then use the subpoly lists to build shapes rather than total_coords"""

    def _make_shape(total_coords, ax, override=False):
        hull_flag = 0
        total_coords = np.array(total_coords)
        if len(total_coords) >= 4:
            # try to draw the convex hull
            try:
                if not override:
                    hull = ss.ConvexHull(total_coords)
                    # get a list of the vertices
                    vertices = [total_coords[v] for v in hull.vertices]
                else:
                    vertices = (
                        total_coords  # we've already found the hulls in previous calls
                    )
                # get a list of the edges
                edges = _pycddlib_helper(vertices)
                # casting edge indices to int for indexing vertices
                start_indices = [int(el) for el in edges[:, 0]]
                start = np.array(vertices)[start_indices]
                end_indices = [int(el) for el in edges[:, 1]]
                end = np.array(vertices)[end_indices]

                # rather than plotting, return edge list
                # warning messy as Im still figuring this out
                # for the subpolys, we want to create find additional vertices from the concat edge lists
                # then use that new set of vertices to find the convex hull
                # this should avoid the problem from before, extending lines where a gap should exist
                hull_flag = 1
                if not override:
                    return 1, [(s, t) for s, t in zip(start, end)]
                else:
                    for i in range(len(edges)):
                        ax.plot(
                            [start[i, 0], end[i, 0]],
                            [start[i, 1], end[i, 1]],
                            [start[i, 2], end[i, 2]],
                            color=color,
                            linestyle="-",
                            marker="",
                        )

                # on success, raise hull_flag
            except ss.QhullError:
                hull_flag = 0
        # flag raised if convex hull fails because is a flat plane
        if len(total_coords) >= 3 and not hull_flag:  # draw the line manually
            for si in range(len(total_coords)):
                dv = np.transpose(
                    [total_coords[si], total_coords[(si + 1) % len(total_coords)]]
                )  # use modulus to wrap around
                ax.plot(dv[0], dv[1], dv[2], color=color)
        elif len(total_coords) < 3:  # draw the points manually:
            # plot the point
            ax.scatter3D(*zip(*total_coords), color=color)
        return 0, ax

    if fig is None:
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        w = WeylChamber()
        w.labels = {}
        w.render(ax)
    else:
        # ax = fig.gca(projection="3d") # deprecated
        ax = fig.gca()

    color = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "cyan",
        "black",
        "pink",
        "brown",
        "grey",
    ][index % 10]
    reduced_vertices = (
        monodromy_to_positive_canonical_polytope(circuit_poly).reduce().vertices
    )
    total_coords = []
    total_edges = []
    for subpoly in reduced_vertices:
        subpoly_coords = [
            [float(x) for x in coord] for coord in subpoly
        ]  # convert from fraction to floats
        total_coords += subpoly_coords
        if override_connect or 1:
            f, ret = _make_shape(subpoly_coords, ax, override=True) #setting override to go back to old method
            if f:  # means ax is an edgelist
                total_edges += ret
            else:
                ax = ret
        # ax.scatter3D(*zip(*subpoly_coords), color="red")

        # # old method
        # pts = np.array(subpoly_coords)
        # hull = ss.ConvexHull(pts)
        # for s in hull.simplices:
        #     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        #     ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], color=color)

    # brute force :)
    """Strategy: I know there is some complicated math to find intersection between 3d lines
    OR we could add a bunch of points along the line and report if any line has a point in common"""
    if False and len(total_edges) > 0:
        # for each edge, add a bunch of points along the line
        temp_coords = []
        for debug, edge in enumerate(total_edges):
            # get the start and end points
            start = edge[0]
            end = edge[1]
            # get the vector between the points
            vec = end - start
            # get the length of the vector
            vec_len = np.linalg.norm(vec)
            # get the unit vector
            unit_vec = vec / vec_len
            # get the number of points to add
            num_points = int(vec_len * 1000)
            # get the step size
            step_size = vec_len / num_points
            # get the points
            points = [start + unit_vec * step_size * i for i in range(num_points)]
            # ax.scatter3D(*zip(*points), color=['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'black', 'pink', 'brown', 'grey'][debug%10])
            # save the points
            temp_coords.append(points)
        # now check if any of the points are in common
        for i in tqdm(range(len(temp_coords))):
            for j in range(i + 1, len(temp_coords)):
                for edge_1_points in temp_coords[i]:
                    edge_2_points = temp_coords[j]
                    # close_points = [point for point in edge_2_points if np.linalg.norm(point - edge_1_points) < 0.0001]
                    close_indices = np.where(
                        np.all(np.isclose(edge_1_points, edge_2_points), axis=1)
                    )[0]
                    # if 0 in close_indices remove it
                    if 0 in close_indices:
                        close_indices = close_indices[1:]  # remove trivial case
                    if len(close_indices) > 5:
                        continue  # too many points, probably colinear lines
                    elif len(close_indices) > 0:
                        close_points = [edge_2_points[index] for index in close_indices]
                        # # plot the points
                        # ax.scatter3D(*zip(*close_points))
                        total_coords += close_points

        # now use the new total_coords to make the shape
        _make_shape(total_coords, ax, override=True)
        ax.scatter3D(*zip(*total_coords))
        # phew !

    # deprecate, always use subpoly lists to construct a new set of vertices
    # if not override_connect:
    #    f, ax = _make_shape(total_coords, ax)

    if save and filename is not None:
        plt.savefig("{fpath_images}/{filename}.svg", format="svg")
    return fig


def _pycddlib_helper(points):
    import cdd as pcdd

    """points to edge only convex hull
    https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud"""
    # to get the convex hull with cdd, one has to prepend a column of ones
    vertices = np.hstack((np.ones((len(points), 1)), points))

    # do the polyhedron
    mat = pcdd.Matrix(vertices, linear=False, number_type="fraction")
    mat.rep_type = pcdd.RepType.GENERATOR
    poly = pcdd.Polyhedron(mat)

    # get the adjacent vertices of each vertex
    adjacencies = [list(x) for x in poly.get_input_adjacency()]

    # store the edges in a matrix (giving the indices of the points)
    edges = [None] * (len(points) - 1)
    for i, indices in enumerate(adjacencies[:-1]):
        indices = list(filter(lambda x: x > i, indices))
        l = len(indices)
        col1 = np.full((l, 1), i)
        indices = np.reshape(indices, (l, 1))
        edges[i] = np.hstack((col1, indices))
    Edges = np.vstack(tuple(edges))
    return Edges
