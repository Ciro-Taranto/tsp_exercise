"""
Tasks:
    -Generate distributions of locations.
    -Define the weight of the edges.
    -Define the load to be collected from each collection point.
"""
import numpy as np
import random
from algorithms.algos import check_if_strongly_connected
from objects.graph import Graph


def distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))


def garbage_collection_points_layout(number_of_points, number_of_deposits):
    """
    Args:
        number_of_points (int): number of collection points
        number_of_deposits(int): number fo deposits
    Returns:
        -a list of 2d coordinates for the collection points.
        -a list of 2d coordinates for the deposits.
        -a list with the load to be collected from each stop.
        -a   (number_of_points+number_of_deposits)
            x(number_of_points+number_of_deposits)
            array with the distance between the points
    """
    collection_points = {'l{}'.format(i): val for i, val in enumerate(
        np.random.rand(number_of_points, 2))}
    deposit_points = {'d{}'.format(i): val for i, val in enumerate(
        np.random.rand(number_of_deposits, 2))}
    load_per_point = {key: np.random.rand()
                      for key in collection_points.keys()}
    all_points = {**collection_points, **deposit_points}
    graph_edges = {}
    for i, key1 in enumerate(all_points):
        for key2 in list(all_points)[:i]:
            graph_edges[(key1, key2)] = distance(
                all_points[key1], all_points[key2])
            graph_edges[(key2, key1)] = graph_edges[(key1, key2)]
    return all_points, load_per_point, graph_edges


def random_graph_layout(number_of_loads, number_of_deposits,
                        number_of_arcs=0.2, traffic=1.0):
    """
    Generate a random graph.
    No restrictrion on weights.
    Symmetric edges.
    After the generation a mst is computed, to check that
    the graph is connected. If this is not the case some
    edges are added to fix this.
    Args:
        number_of_points (int): number of collection points
        number_of_deposits(int): number fo deposits
        number_of_arcs(int or float): If int number of arcs to be added
            if float fractions of possible connections to be included.
            Please, keep in mind that if the resulting graph is not connected
            new arcs will be added.
        traffic(float)[0.,1.]: If 1. then only random connections,
            otherwise w(i,j)=t*random+(1-t)*distance(i,j)
    """
    collection_points = {'l{}'.format(i): val for i, val in enumerate(
        np.random.rand(number_of_loads, 2))}
    deposit_points = {'d{}'.format(i): val for i, val in enumerate(
        np.random.rand(number_of_deposits, 2))}
    load_per_point = {key: np.random.rand()
                      for key in collection_points.keys()}

    all_points = {**collection_points, **deposit_points}
    points_index = {i: val for i, val in enumerate(all_points.keys())}
    number_of_points = len(all_points)
    total_arcs = int(number_of_points*(number_of_points-1)/2)

    if isinstance(number_of_arcs, float):
        number_of_arcs = int(number_of_arcs*total_arcs)
    number_of_arcs = int(min(number_of_arcs, total_arcs))
    lower_triangle = np.tril_indices(number_of_points, -1)
    selected_arc_index = np.random.choice(
        range(total_arcs), number_of_arcs, replace=False)
    graph_edges = {(points_index[lower_triangle[0][i]],
                    points_index[lower_triangle[1][i]]):
                   np.random.rand() for i in selected_arc_index}
    graph_edges.update({(k[1], k[0]): v for k, v in graph_edges.items()})

    g = Graph(locations=all_points, weights=graph_edges)

    iter_vertices = list(g.get_vertices())
    random.shuffle(iter_vertices)
    vertices_ids = set(iter_vertices)

    for vert_id in iter_vertices:
        vert_connections = g.get_vertex(vert_id).get_connections()
        if len(vert_connections) < 2:
            if len(vert_connections) == 1:
                vert_connections = set(
                    (next(iter(vert_connections)).get_id(), vert_id))
            else:
                vert_connections = [v.get_id() for v in vert_connections]
                vert_connections.append(vert_id)
                vert_connections = set(vert_connections)
            candidates = vertices_ids - vert_connections
            g.add_edge(vert_id, np.random.choice(
                list(candidates)), np.random.rand())
    graph_edges = g.get_edges(get_all=True)
    print("graph_edges = {}".format(len(graph_edges)/2))

    fully_connected = check_if_strongly_connected(g)
    while fully_connected is not True:
        disconnected = np.random.choice(
            [k for k, v in fully_connected.items() if v is False])
        connected = np.random.choice(
            [k for k, v in fully_connected.items() if v is True])
        w = np.random.rand()
        g.add_edge(disconnected, connected, w)
        g.add_edge(connected, disconnected, w)
        fully_connected = check_if_strongly_connected(g)
    graph_edges = g.get_edges(get_all=True)
    print("graph_edges = {}".format(len(graph_edges)/2))

    # Add component of distance (non random)
    if traffic < 1.:
        for k in graph_edges.keys():
            d = distance(all_points[k[0]], all_points[k[1]])
            graph_edges[k] = traffic*graph_edges[k] +\
                (1.-traffic)*d/np.sqrt(2.)
    # return g
    return all_points, load_per_point, graph_edges
