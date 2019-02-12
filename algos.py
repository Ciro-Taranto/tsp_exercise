from graph import Graph
import numpy as np
from collections import defaultdict


def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    # Attach smaller rank tree under root of
    # high rank tree (Union by Rank)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot

    # If ranks are same, then make one as root
    # and increment its rank by one
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def cut_edges(included, all_edges):
    edges = [(k[0], k[1]) for k in all_edges if
             (k[0] in included and k[1] not in included)
             or (k[0] not in included and k[1] in included)]
    return edges


def mst_prim(locations, weights, start_vert):
    """
    Prim algorithm to find MST on a graph.
    Adapted from:
    https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
    """
    g = Graph(locations=locations, weights=weights)
    mst = Graph()
    included = []
    included_edges = {}

    # Assumption: undirected graph, symmetric weight
    edges = sort_edges(weights, locations)

    parent = {vert: None for vert in g.get_vertices()}
    mstSet = {vert: False for vert in g.get_vertices()}

    # Start from the start location
    included.append(start_vert)
    mst.add_vertex(start_vert, locations[start_vert])
    while mst.num_vertices < g.num_vertices:
        cut = cut_edges(included, edges)
        [k1, k2] = cut_edges(included, edges)[0]
        if k1 in included:
            candidate = k2
        else:
            candidate = k1
        included.append(candidate)
        mst.add_vertex(candidate, locations[candidate])
        mst.add_edge(k1, k2, weights[(k1, k2)])
        mst.add_edge(k2, k1, weights[(k2, k1)])
    return mst


def sort_edges(weights, locations):
    """
    Given weights and locations,
    sort edges in increasing value of cost
    and returns the pairs
    """
    edges = {}
    for i, key1 in enumerate(locations):
        for key2 in list(locations)[: i]:
            edges[(key1, key2)] = weights[(key1, key2)]

    # Get the sorted list of the edges
    edges = sorted(edges, key=edges.get)
    return edges


def evaluate_mst(mst):
    """
    Evaluate the cost of the graph object
    in mst
    """
    edges = mst.get_edges()
    return np.sum([val for key, val in edges.items()])


def mst_kruskal(locations, weights):
    g = Graph(locations=locations, weights=weights)
    mst = Graph()

    edges = sort_edges(weights, locations)

    parent = {}
    rank = {}
    for vert in g.get_vertices():
        parent[vert] = vert
        rank[vert] = 0

    # Number of edges to be taken is equal to V-1
    for edge in edges:

        # Step 2: Pick the smallest edge and increment
        # the index for next iteration
        [u, v] = edge
        x = find(parent, u)
        y = find(parent, v)

        # If including this edge does't cause cycle,
        # include it in result and increment the index
        # of result for next edge
        if x != y:
            if u not in mst.get_vertices():
                mst.add_vertex(u, locations[u])
            if v not in mst.get_vertices():
                mst.add_vertex(v, locations[v])
            weight = weights[(u, v)]
            mst.add_edge(u, v, weight)
            mst.add_edge(v, u, weight)
            union(parent, rank, x, y)
    return mst


def heuristic(edges_matrix):
    """
    Heuristic function for the TSP,
    using the minimum-spanning-tree.
    Cost(MSP) <= Cost(TSP)
    Args:
        edges_matrix(np.array): weights of the edges
    Returns:
        cost of the minimum spanning tree
    """
