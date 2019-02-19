#######################################################
#  Based on: https://github.com/aimacode/aima-python  #
#######################################################

from algorithms import algos
from objects.graph import Graph
from objects.solver import Solver

from problems.abstract_problems import Problem, Node
from utils import memoize, PriorityQueue
import numpy as np


class TravelingSalesman(Solver):
    """
    Implementation of TSP adapted to implementation of 
    Graph class and ready to be solved with the astar search.

    It is not very elegant here: it is implemented extending the Problem
    class, but what we really have here is a Solver object.
    [TODO]: Create a Solver class
    """

    def __init__(self, locations, weights, start=None,
                 limit_actions=None):
        """
        Args: 
            locations(dict):{name:(x,y)}
            edges(dict):{('from','to'):weight}
            start(str): Node for the exploration starting
            limit_actions(int): max number of neighbors to consider
        """
        self.graph = Graph(locations=locations, weights=weights)
        self.locations = locations
        self.total_locations = len(locations)
        self.weights = weights
        self.edges = algos.sort_edges(weights, locations)
        # Note: we represent states as tuples,
        # as these are hashable
        if start is None:
            initial = (next(iter(self.graph)).get_id(),)
        else:
            initial = (start,)
        Problem.__init__(self, initial, None)
        self.start_neighbors = self._sorted_neighbors_from_start()
        self.limit_actions = limit_actions

    def actions(self, A):
        """
        The state A is represented as a tuple.
        From there the agent can move to any of its neighbors
        that have not been visited yet.
        """
        if isinstance(A, tuple):
            neighbors = self.graph.get_vertex(A[-1]).get_connections(sort=True)
            # Space for improvement here
            # Remove already visited locations
            neighbors = [x for x in neighbors if x.get_id() not in list(A)]
        else:
            raise Exception("Type of {} not recognized".format(A))

        if self.limit_actions is None:
            return neighbors
        else:
            return neighbors[:self.limit_actions]

    def result(self, state, action):
        """
        The result of going to a neighbor is that the neighbor
        would be added to the list of visited.
        Take care that the action is an instance of Vertex
        So we need to extract its id
        """
        return state + (action.get_id(), )

    def path_cost(self, cost_so_far, state, action, next_state):
        """
        The action is an instance of Vertex,
        while state is the path so far
        """
        v = self.graph.get_vertex(state) if isinstance(
            state, str) else self.graph.get_vertex(state[-1])
        weight = action.get_weight(v)
        return cost_so_far + weight

    def h(self, node):
        """
        h function is derived from the MST.
        """
        mst = self._mst_kruskal(self.locations, self.weights,
                                self.edges, node.state)
        mst_cost = algos.evaluate_solution(mst)
        go_back_to_start_cost = self._distance_from_subgraph(node.state)
        return mst_cost + go_back_to_start_cost

    def goal_test(self, state):
        """
        If all the locations have been visited the state is goal
        """
        return len(state) == self.graph.num_vertices

    def solve(self):
        """
        Execute the solution
        """
        solution_node = astar_search(self)
        if solution_node is False:
            print('The algorithm could not find a solution')
            return False
        solution_graph = self.graph.build_graph_solution(solution_node.state)
        print(algos.evaluate_solution(solution_graph))
        return solution_graph

    def _sorted_neighbors_from_start(self):
        """
        Since the distance from start will be called several times
        it makes sense to store it
        """
        start_vertex = self.graph.get_vertex(self.initial[0])
        neighbors = start_vertex.get_connections(sort=True)
        return [[n.get_id(), n.get_weight(start_vertex)] for n in neighbors]

    def _distance_from_subgraph(self, state):
        if len(state) == self.total_locations:
            neighbors = [n[0] for n in self.start_neighbors]
            try:
                ind = neighbors.index(state[-1])
                return self.start_neighbors[ind][1]
            except ValueError:
                return np.Inf
        for n in self.start_neighbors:
            if n[0] not in state:
                return n[1]
        return np.Inf

    def _mst_kruskal(self, locations, weights, edges, state):
        """
        Adapted version of the kruskal algo,
        that allows to execute it several times without 
        altering the data. 
        The drawback is that when the nodes are reduced, the
        execution does not speed up, one still have to transverse
        all the edges.
        """
        mst = Graph()

        parent = {}
        rank = {}
        for vert in locations.keys():
            parent[vert] = vert
            rank[vert] = 0

        # Number of edges to be taken is equal to V-1
        for edge in edges:
            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            [u, v] = edge
            # With the if condition we will not avoid
            # transversing all the edges.
            # But we avoid a lot of data manipulation
            if u not in state and v not in state:
                x = algos.find(parent, u)
                y = algos.find(parent, v)

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
                    algos.union(parent, rank, x, y)
        return mst


def best_first_graph_search(problem, f):
    """
    Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if lf.get_edge_weight(order[0], order[-1])f is node.depth then we have breadth-first search.
    There is a subtle        g.add_edge(order[0], order[-1], w)ty: the line "f = memoize(f, 'f')" means that the f
    values will be ca        g.add_edched on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned.
    """
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    i = 0
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if f(node) < np.Inf:
                return node
            else:
                return False
        explored.add(node.state)
        for child in node.expand(problem):
            # This is dangerous, but I am assuming a given state
            # Can be reached only by one path
            frontier.append(child)
            # if child.state not in explored and child not in frontier:
            #     frontier.append(child)
            # elif child in frontier:
            #     incumbent = frontier[child]
            #     if f(child) < f(incumbent):
            #         del frontier[incumbent]
            #         frontier.append(child)
        i += 1
        if i % 10000 == 0:
            print('Checked already {} nodes'.format(i))
            print(node.state)
    print('Solution could not be found within the limits imposed')
    return False


def astar_search(problem, h=None):
    """
    A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass.
    """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))
