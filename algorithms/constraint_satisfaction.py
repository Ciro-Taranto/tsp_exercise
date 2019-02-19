# Generate a non optimal solution of the TSP

from objects.graph import Graph
from problems.abstract_problems import Problem, Node
# [TODO] Move position of astar_search in more appropriate position!
from algorithms.search import astar_search


class ConstraintSatisfaction(Problem):
    """
    Object implementing a constraint satisfaction problem,
    using some heuristics.
    """

    def __init__(self, locations, edges):
        """
        The state of the problem is represented by a tuple of two tuples.
        It is assumed that locations and edges satisfy some basics rules:
        -The graph is connected
        -All the edges have at least two connections
        """
        self.locations = locations
        self.edges = edges
        self.total_locations = len(locations)
        self.all_locations = set(self.locations.keys())
        self.graph = Graph(locations=locations, weights=edges)
        self.unique_edges = self.graph.get_edges()
        self.vertex_connections = self.graph.get_vertices_connections()
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
        remaining_nodes = self.all_locations - set(state[0])-set(state[0])
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
                                 if i in remaining_nodes]),
                       val, act]
                      for key, val
                      in self.vertex_connections[last_added].items()
                      if key in remaining_nodes]

        return candidates

    def result(self, state, action):
        """
        The result of the action is updating the state
        Take care because here a deepcopy is required
        """
        if action[3] == 0:
            return (state[0]+(action[0],), state[1])
        else:
            return (state[0], state[1]+(action[0],))

    def path_cost(self, cost_so_far, state, action, next_state):
        """
        We want to do two things: minimizing the cost
        and have few iterations in the approximate solution.
        To minimize the cost we want to minimize the weight (action[2])
        To minimize the iterations we want to always pick the node with less
        connections (action[1]).
        Any linear combination of these two is a reasonable candidate.
        """
        # This coeff will have to be fixed in a smart way!
        coeff = 0.0
        # is cost so far really relevant?
        return cost_so_far + 1-coeff*(action[1])+coeff*(action[2])

    def h(self, node):
        """
        We do not have a heuristic
        """
        return 0

    def goal_test(self, state):
        """
        If all the locations have been visited the state is goal.
        -1 is because the initial location is in both sets
        """
        return len(state[0])+len(state[1])-1 == self.graph.num_vertices

    def solve(self):
        """
        Execute the solution
        """
        solution_node = depth_first_tree_search(self)
        if solution_node is False:
            print('The algorexpandthm could not find a solution')
            return False
        return type(solutionexpandnode)

        # solution_graph = lexpandst
        # solution_graph = sexpandlf.graph.build_graph_solution(solution_node.state)
        # print(algos.evaluaexpande_solution(solution_graph))
        # return solution_graph

    def backtrack(self, state):
        if self.goal_test(state):
            return state


# [TODO]: Move this function in appropriate module
def depth_first_tree_search(problem):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None
