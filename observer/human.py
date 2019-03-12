import pylab as plt
import numpy as np
from matplotlib import collections as mc
from matplotlib import colors
from collections import OrderedDict

def render(*gg):
    """
    function to plot the graph g 
    """
    fig, ax = plt.subplots()
    for g in gg:
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

        ax.scatter(
            locations[:, 0], locations[:, 1],
            color='blue')  # , s=5*s)
        ax.scatter(
            deposits[:, 0], deposits[:, 1],
            color='red'
        )

    colors_dict = OrderedDict(colors.BASE_COLORS, **colors.CSS4_COLORS)
    cols = list(colors_dict.items())

    for i, g in enumerate(gg):
        edges = g.get_edges()
        edges = [[g.get_vertex(k[0]).get_location(),
                  g.get_vertex(k[1]).get_location()]
                 for k in edges.keys()]
        lines = mc.LineCollection(edges, linewidths=1, color=cols[i][1])
        ax.add_collection(lines)
    plt.show()
    return True
