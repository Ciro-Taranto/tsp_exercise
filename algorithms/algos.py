import numpy as np
from objects.graph import Graph


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
    # Admittedly, the following lines are horrible even if
    # they will work eventually.

    edges = {}
    for key, val in weights.items():
        if (key[1], key[0]) not in edges.keys():
            edges[key] = val

    # Get the sorted list of the edges
    edges = sorted(edges, key=edges.get)
    return edges


def evaluate_solution(solution):
    """
    Evaluate the cost of the graph object
    in mst
    """
    edges = solution.get_edges()
    return np.sum([val for key, val in edges.items()])


def transversal(g, visited, v):
    visited[v] = True
    for neighbor in g.get_vertex(v).get_connections():
        n = neighbor.get_id()
        if not visited[n]:
            transversal(g, visited, n)
    return visited


def check_if_strongly_connected(g, start=None):
    """
    g is the graph to check
    """
    visited = {key: False for key in g.get_vertices()}
    if start is None:
        start = list(g.get_vertices())[0]
    try:
        transversal(g, visited, start)
    except RecursionError:
        print("Still to implement a different check for many nodes")
        raise NotImplementedError
    if all(val for key, val in visited.items()):
        return True
    else:
        return visited


def mst_kruskal(locations, weights, connectivity=False, return_parent=False):
    """
    If return_parent is True it will be returned the dictionary of the ranks
    """
    mst = Graph()

    edges = sort_edges(weights, locations)

    parent = {}
    rank = {}
    for vert in locations.keys():
        parent[vert] = vert
        rank[vert] = 0

    for edge in edges:

        # Step 2: Pick the smallest edge and increment
        # the index for next iteration
        [u, v] = edge
        x = find(parent, u)
        y = find(parent, v)

        # If including this edge does't cause cycle,
        # include it in result and increment the index
        # of result for next edge
        if x != y or connectivity:
            if u not in mst.get_vertices():
                mst.add_vertex(u, locations[u])
            if v not in mst.get_vertices():
                mst.add_vertex(v, locations[v])
            weight = weights[(u, v)]
            mst.add_edge(u, v, weight)
            mst.add_edge(v, u, weight)
            union(parent, rank, x, y)

    if return_parent:
        return parent
    else:
        return mst
