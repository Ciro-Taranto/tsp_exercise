#######################################################
#  Based on: https://github.com/aimacode/aima-python  #
#######################################################

from graph import Graph
from collections import OrderedDict
import heapq
import algos


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
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
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
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.path_cost(self.path_cost, self.state,
                                           action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
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


class FrontierPQ:
    "A Frontier ordered by a cost function; a Priority Queue."

    def __init__(self, initial, costfn=lambda node: node.path_cost):
        "Initialize Frontier with an initial Node, and specify a cost function."
        self.heap = []
        self.states = {}
        self.costfn = costfn
        self.add(initial)

    def add(self, node):
        "Add node to the frontier."
        cost = self.costfn(node)
        heapq.heappush(self.heap, (cost, node))
        self.states[node.state] = node

    def pop(self):
        "Remove and return the Node with minimum cost."
        (cost, node) = heapq.heappop(self.heap)
        self.states.pop(node.state, None)  # remove state
        return node

    def replace(self, node):
        "Make this node replace a previous node with the same state."
        if node.state not in self:
            raise ValueError('{} not there to replace'.format(node.state))
        for (i, (cost, old_node)) in enumerate(self.heap):
            if old_node.state == node.state:
                self.heap[i] = (self.costfn(node), node)
                heapq._siftdown(self.heap, 0, i)
                return

    def __contains__(self, state): return state in self.states

    def __len__(self): return len(self.heap)


def uniform_cost_search(problem, costfn=lambda node: node.path_cost):
    frontier = FrontierPQ(Node(problem.initial), costfn)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = node.child(problem, action)
            if child.state not in explored and child not in frontier:
                frontier.add(child)
            elif child in frontier and frontier.cost[child] < child.path_cost:
                frontier.replace(child)


class TravelingSalesman(Problem):
    """
    Implementation of TSP adapted to the
    Graph class.
    Although this is not elegant the problem instance
    is defined through dictionaries of vertices and edges.
    """

    def __init__(self, locations, weights, start=None):
        """
        Initialize the problem.
        The locations can be addressed by lists.
        """
        self.graph = Graph(locations=locations, weights=weights)
        self.edges = algos.sort_edges(weights)
        if start is not None:
            initial = [next(iter(graph)).get_id()]
        else:
            initial = [start]
        Problem.__init__(self, initial, goal)
        self.start_neighbors = self._sorted_neighbors_from_start()

    def actions(self, A, limit=None):
        """
        The state A is represented as a list.
        From there the agent can move to any of its neighbors
        that have not been visited yet.
        Limit is used to restrict the actions:
        There is no need to move too far with a step!
        """
        neighbors = self.graph.get_vertex(A[-1]).get_connections(sort=True)
        # Space for improvement here
        # Remove already visited locations
        neighbors = [x for x in neighbors if x not in A]
        if limit is None:
            return neighbors
        else:
            return neighbors[:limit]

    def result(self, state, action):
        """
        The result of going to a neighbor is that the neighbor
        would be added to the list of visited.
        Take care that the action is an instance of Vertex
        So we need to extract its id
        """
        s2 = state.copy()
        return s2.append(action.get_id())

    def path_cost(self, cost_so_far, state, action):
        """
        The action is an instance of Vertex,
        while state is the path so far
        """
        weigth = action.get_weight(self.graph.get_vertex(state[-1]))
        return cost_so_far + weight

       def h(self, state, action):
        """
        h function is derived from the MST. 
        """
        candidate_state = state.copy()
        candidate_state.append(actiong.get_id())
        mst = self._mst_kruskal(self.locations, self.weights,
                                self.edges, candidate_state)
        mst_cost = algos.evaluate_solution(mst)
        go_back_to_start_cost = self._distance_from_subgraph(candidate_state)
        return mst_cost + go_back_to_start_cost

    def goal_test(self, state):
        """
        If all the locations have been visited the state is goal
        """
        return len(state) == self.graph.num_vertices 

    def _sorted_neighbors_from_start(self):
        """
        Since the distance from start will be called several times
        it makes sense to store it 
        """
        start_vertex = self.graph.get_vertex(self.initial[0])
        neighbors = start_vertex.get_connections(sort=True)
        return [[n.get_id(), n.get_weight(start_vertex)] for n in neighbors]

    def _distance_from_subgraph(self, state):
        for n in self.start_neighbors:
            if n[0] not in state:
                return n[1]

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


class Search():
    """
    Class to implement an instance of search using A*.
    Based on: 
    https://github.com/aimacode/aima-python 
    """
    def __init__():
        return True

    def uniform_cost_search(self, problem, costfn=lambda node: node.path_cost):
        frontier = FrontierPQ(Node(problem.initial), costfn)
        explored = set()
        while frontier:
            node = frontier.pop()
            if problem.is_goal(node.state):
                return node
            explored.add(node.state)
            for action in problem.actions(node.state):
                child = node.child(problem, action)
                if child.state not in explored and child not in frontier:
                    frontier.add(child)
                elif child in frontier and frontier.cost[child] < child.path_cost:
                    frontier.replace(child)

    def astar_search(self, problem, heuristic):
        def costfn(node): return node.path_cost + heuristic(node.state)
        return self.uniform_cost_search(problem, costfn)
