# Generate a non optimal solution of the TSP

from objects.graph import Graph
from problems.abstract_problems import Problem, Node
import numpy as np


class ConstraintSatisfaction(Problem):
    """
    Object implementing a constraint satisfaction problem,
    using some heuristics.
    """

    def __init__(self, graph, lucky=False, luck_limit=np.Inf, **kwargs):
        """
        The state of the problem is represented by a tuple of two tuples.
        It is assumed that locations and edges satisfy some basics rules:
        -The graph is connected
        -All the edges have at least two connections
        """
        self.graph = graph
        self.locations = self.graph.get_locations()
        self.edges = self.graph.get_edges(get_all=True)
        self.total_locations = graph.num_vertices
        self.all_locations = set(self.locations.keys())
        self.unique_edges = self.graph.get_edges()
        self.vertex_connections = self.graph.get_vertices_connections()
        self.lucky = lucky
        self.luck_limit = luck_limit
        initial = self.find_initial()
        Problem.__init__(self, initial, None)

    def find_initial(self):
        connectivity = {key: len(val)
                        for key, val in self.vertex_connections.items()}
        most_complicated_node = sorted(connectivity, key=connectivity.get)[0]
        state = ((most_complicated_node,), (most_complicated_node,))
        print(state)
        return state

    def actions(self, state):
        """
        The actions must be sorted according to the 
        Minimum-Remaining-Values Heuristic
        """
        remaining_nodes = self.all_locations - set(state[0])-set(state[1])
        if len(state[1]) < len(state[0]):
            # If the left list is longer than the right one,
            # act on the right one
            last_added = state[1][-1]
            act = 1
        else:
            # Otherwise act on the left
            last_added = state[0][-1]
            act = 0

        candidates = [[key, len([i for i in self.vertex_connections[key]
                                 if i in remaining_nodes]), val]
                      for key, val
                      in self.vertex_connections[last_added].items()
                      if key in remaining_nodes]
        if self.lucky:
            # This will favor cheap edges
            desirable = np.argsort([i[2] for i in candidates])[::-1]
        else:
            # This will favor vertices loosely connected
            desirable = np.argsort([i[1] for i in candidates])[::-1]

        ret = [[candidates[i][0], act] for i in desirable]
        return ret

    def result(self, state, action):
        """
        The result of the action is updating the state
        Take care because here a deepcopy is required
        """
        if action[1] == 0:
            ret = tuple((state[0]+(action[0],), state[1]))
            return ret
        else:
            return tuple((state[0], state[1]+(action[0],)))

    def goal_test(self, state):
        """
        Check that all the locations have been visited + 
        the last two locations are connected
        """
        if len(state[0])+len(state[1])-1 == self.graph.num_vertices:
            # Then check if the last two nodes share a connection
            return (state[0][-1], state[1][-1]) in self.edges.keys()
        else:
            return False

    def solve(self):
        """
        Execute the solution
        """
        solution_node = depth_first_tree_search(
            self, luck_limit=self.luck_limit)
        if solution_node is False:
            print('The algorithm could not find a solution')
            return False
        solution = list(solution_node.state[0]) + \
            list(solution_node.state[1])[::-1][:-1]
        solution_graph = self.graph.build_graph_solution(solution)
        return solution_graph


# [TODO]: Move this function in appropriate module
def depth_first_tree_search(problem, luck_limit=np.Inf):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = [Node(problem.initial)]  # Stack
    i = 0
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            print(i)
            return node
        frontier.extend(node.expand(problem))
        if i % 100000 == 0:
            print("Already checked {} nodes".format(i))
        i += 1
        if i > luck_limit:
            return False
    return False
