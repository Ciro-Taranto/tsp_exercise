"""
Tasks:
    -Generate distributions of locations.
    -Define the weight of the edges.
    -Define the load to be collected from each collection point.
"""
import numpy as np


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
    # for p1 in range(len(all_points)):
    #     for p2 in range(p1):
    #         graph_edges[(p1, p2)] = distance(all_points[p1], all_points[p2])
    # graph_edges = {k: v for k, v in sorted(
    #     graph_edges.items(), key=lambda kv: kv[1])}
    return all_points, load_per_point, graph_edges
