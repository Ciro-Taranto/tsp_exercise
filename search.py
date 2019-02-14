#######################################################
#  Based on: https://github.com/aimacode/aima-python  #
#######################################################

from graph import Graph
import algos
from utils import memoize, PriorityQueue


class Problem(object):
    """
    The abstract class for a formal problem.
    """

    def __init__(self, initial, goal=None):
        """
        The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.
        """
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state.
        """
        raise NotImplementedError

    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        """
        raise NotImplementedError

    def goal_test(self, state):
        """
        Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.
        """
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """
        Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1.
        If the problem
        is such that the path doesn't matter, this function will only look at
        state2.
        If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """
        For optimization problems, each state has a value.
        Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
# ______________________________________________________________________________


class Node:
    """
    A node in a search tree.
    Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.
    Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class.
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """
        Create a search tree Node, derived from a parent by an action.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        # [CT]: I do not get this one, unless we want to override it
        return self.state < node.state

    def expand(self, problem):
        """
        List the nodes reachable in one step from this node.
        """
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.path_cost(self.path_cost, self.state,
                                           action, next_state))
        return next_node

    def solution(self):
        """
        Return the sequence of actions to go from the root to this node.
        """
        return [node.action for node in self.path()[1:]]

    def path(self):
        """
        Return a list of nodes forming the path from the root to this node.
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


class TravelingSalesman(Problem):
    """
    Implementation of TSP adapted to the
    Graph class.
    Although this is not elegant the problem instance
    is defined through dictionaries of vertices and edges.
    """

    def __init__(self, locations, weights, start=None,
                 limit_actions=None, goal=None):
        """
        Initialize the problem.
        The locations can be addressed by lists.
        """
        self.graph = Graph(locations=locations, weights=weights)
        self.locations = locations
        self.weights = weights
        self.edges = algos.sort_edges(weights, locations)
        # Note: we represent states as tuples,
        # as these are hashable
        if start is None:
            initial = (next(iter(self.graph)).get_id())
        else:
            initial = (start)
        Problem.__init__(self, initial, goal)
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
        elif isinstance(A, str):
            neighbors = self.graph.get_vertex(A).get_connections(sort=True)
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
        # A bit of care with tuples and lists
        if isinstance(state, str):
            s2 = [state]
        else:
            s2 = list(state)
        return tuple(s2+[action.get_id()])

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

    def astar_search(self):
        solution_node = astar_search(self)
        solution_graph = self.graph.build_graph_solution(solution_node.state)
        print(algos.evaluate_solution(solution_graph))
        return solution_graph

    def _sorted_neighbors_from_start(self):
        """
        Since the distance from start will be called several times
        it makes sense to store it
        """
        start_vertex = self.graph.get_vertex(self.initial)
        neighbors = start_vertex.get_connections(sort=True)
        return [[n.get_id(), n.get_weight(start_vertex)] for n in neighbors]

    def _distance_from_subgraph(self, state):
        for n in self.start_neighbors:
            if n[0] not in state:
                return n[1]
        return 0.0

    def _mst_kruskal(self, locations, weights, edges, state):
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
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
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
            return node
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
