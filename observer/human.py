import pylab as plt
from matplotlib import collections as mc
import numpy as np


def render(*gg):
    """
    function to plot the graph g 
    """
    try:
        g = gg[0]
    except:
        print('At least one argument is needed!')
        return
    vertices = g.get_vertices()
    locations = []
    deposits = []

    for vert in vertices:
        location = g.get_vertex(vert).get_location()
        if vert.startswith('l'):
            locations.append(location)
            # s.append(self.load_per_point[vert])
        elif vert.startswith('d'):
            deposits.append(location)
    locations = np.array(locations)
    deposits = np.array(deposits)
    fig, ax = plt.subplots()

    ax.scatter(
        locations[:, 0], locations[:, 1],
        color='blue')  # , s=5*s)
    ax.scatter(
        deposits[:, 0], deposits[:, 1],
        color='red'
    )
    color_list = ['green', 'orange', 'purple', 'yellow']
    for i, g in enumerate(gg):
        edges = g.get_edges()
        edges = [[g.get_vertex(k[0]).get_location(),
                  g.get_vertex(k[1]).get_location()]
                 for k in edges.keys()]
        lines = mc.LineCollection(edges, linewidths=1, color=color_list[i])
        ax.add_collection(lines)
    return True
