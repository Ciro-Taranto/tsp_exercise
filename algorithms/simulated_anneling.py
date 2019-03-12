import random
import time
import numpy as np
import matplotlib.pyplot as plt
from objects.graph import Graph
from objects.solver import Solver
from algorithms import algos
from algorithms.constraint_satisfaction import ConstraintSatisfaction


def approximate_traveling_salesman(locations, weights, start=None):
    """
    Approximation to TSP: Start from a location and pick nearest location
    yet to be added.
    """
    # Highly inefficient, used only at the beginning of
    #  simulated annealing.
    # [TODO]: Implement a copy method for graphs
    g = Graph(locations=locations, weights=weights)
    total_vertices = len(locations)

    heuristic = Graph()

    if start is None:
        start = next(iter(g)).get_id()
    heuristic.add_vertex(start, g.get_vertex(start).get_location())

    # Terminate the loop once all poins have been added
    while heuristic.num_vertices < total_vertices:
        # Find the nearest neighbor of the last vertex
        nearest = g.get_vertex(start).get_nearest().get_id()
        # Add it to the result graph, with proper weight
        heuristic.add_vertex(nearest, g.get_vertex(nearest).get_location())
        weight = g.get_vertex(nearest).get_weight(g.get_vertex(start))
        heuristic.add_edge(nearest, start, weight)
        heuristic.add_edge(start, nearest, weight)
        # Delete it from the utility graph
        g.delete_vertex(start)
        # Set the last vertex as a start
        start = nearest
    # Return the solution
    return heuristic


class SimAnneal(Solver):
    def __init__(self, graph, T=None, alpha=0.995,
                 stopping_T=1e-6, stopping_iter=100000, start=None,
                 start_solution=None, get_lucky=False,
                 luck_limit=int(1e6), **kwargs):
        """
        Solver for simulated annealing.
        Start from an approximate solution and propose moves.
        Approximate solution: chose vertex and pick the next
        Args:
        :param graph:Graph, instance of graph to solve
            T(float): Starting temp, otherwise <w(approximated_sol)>.
            stopping_iter(int): maximum iteration
            start(str): vert to start from
            start_solution(list):['v1','v2',...] order of visit of vertices
            get_lucky(bool): If true, the initialization will first try to
                start from a greedy approximation. If this is too expensive,
                fall back on the MRV approach to find a start solution
            luck_limit: how many nodes are worth trying before falling back on
                MVR approach
        """
        self.graph = graph
        self.locations = self.graph.get_locations()
        self.N = self.graph.num_vertices
        self.edges = self.graph.get_edges(get_all=True)
        self.start = start if start is not None else next(
            iter(self.graph)).get_id()

        # We need a temperature scale if this is not given:
        # one possibility is the average distance between two edges
        self.alpha = alpha
        assert self.alpha < 1.
        self.stopping_T = stopping_T
        self.stopping_iter = stopping_iter
        self.iter = 1
        self.get_lucky = get_lucky
        self.luck_limit = luck_limit
        print(type(start_solution))
        print(start_solution is None)
        if start_solution is not None:
            self.curr_solution = start_solution
        else:
            print('Initializing solution')
            self.curr_solution = self._initialize_solution()

        # The starting temperature is extremely important:
        # It has to be fixed according to some length scale,
        # which in turns depends on the number of points
        if isinstance(self.curr_solution, Graph):
            self.curr_order = self.curr_solution.adding_order
            # At the end of the list append the first element again
            self.curr_order = self.curr_order + [self.curr_order[0]]
            self.curr_solution_val = algos.evaluate_solution(
                self.graph.build_graph_solution(self.curr_order[:-1]))
            edges = self.curr_solution.get_edges()
            self.T = T if T is not None else sum(
                [v for k, v in edges.items()])/len(edges.items())/2
            print(self.curr_solution_val)
            print("Starting Temperature = {}".format(self.T))
            self.T_start = self.T
            self.fitness_list = [self.curr_solution_val]
            self.best_list = [self.curr_solution_val]
        else:
            print('The problem instance has no solution. Sad but true')

    @classmethod
    def from_locations_and_edges(cls, locations, edges):
        graph = Graph(locations=locations, weights=edges)
        return SimAnneal(graph)

    def _initialize_solution(self):
        print('Initializing solution as csp')
        cp = ConstraintSatisfaction(self.graph, lucky=self.get_lucky, luck_limit=self.luck_limit)
        sol = cp.solve()
        if sol is False:
            cp = ConstraintSatisfaction(self.graph)
            sol = cp.solve()
        print(type(sol))
        return sol

    def _p_accept(self, cost):
        """
        It is assumed fitness_difference > 0
        meaning: candidate cost higher than current cost
        If fitness_difference < 0 the function does not need to be called!
        """
        return np.exp(-cost/self.T)

    def _accept(self, cost):
        if cost < 0.:
            return True
        else:
            if random.random() < self._p_accept(cost):
                return True
            else:
                return False

    def _check_swappable(self, candidate_id, candidate_order,
                         second_candidate_order,
                         break_direction):
        # First: if we are breaking a bond in direction plus we must have
        # candidate_order > second_candidate_order
        c1 = ((candidate_order > second_candidate_order
               and break_direction == 1) or
              (candidate_order < second_candidate_order
               and break_direction == -1))
        if c1 and second_candidate_order != 0:
            second_next_vert = self.graph.get_vertex(
                self.curr_order[second_candidate_order-break_direction])
            substitutions = second_next_vert.get_connections(
                retrieve_id=True)
            return candidate_id in substitutions
        return False

    def _execute_swap(self, l, r):
        # This function modifies the state, it is not idempotent
        if l > r:
            l, r = r, l
        self.curr_order[l: r + 1] = self.curr_order[l: r + 1][::-1]

    def _try_to_swap(self, candidate_id, candidate_order,
                     second_candidate_id,
                     second_candidate_order, break_direction):
        # Find the ids of the bonds to break
        next_candidate_id = self.curr_order[candidate_order+break_direction]
        second_next_candidate_id = self.curr_order[second_candidate_order -
                                                   break_direction]
        # Find the cost of the bonds to break
        tie_break1 = self.edges[(candidate_id, next_candidate_id)]
        tie_break2 = self.edges[(
            second_candidate_id, second_next_candidate_id)]
        tie_add1 = self.edges[(candidate_id, second_next_candidate_id)]
        tie_add2 = self.edges[(second_candidate_id, next_candidate_id)]
        cost = (tie_add1 + tie_add2) - (tie_break1 + tie_break2)

        # If the cost is acceptable execute swap
        if self._accept(cost):
            self._execute_swap(
                candidate_order, second_candidate_order)
            self.curr_solution_val += cost
            if self.curr_solution_val < min(self.fitness_list):
                self.best_list.append(self.curr_solution_val)
            self.fitness_list.append(self.curr_solution_val)
        return True

    def solve(self, **kwargs):
        """
        Execute the solution
        """
        start_time = time.time()
        while self.T > self.stopping_T and self.iter < self.stopping_iter:

            # Select one candidate and the bond to break
            # relative to this candidate
            candidate_order = random.randint(1, self.N-1)
            candidate_id = self.curr_order[candidate_order]
            break_direction = np.random.choice([+1, -1])

            # The candidate can be substituted by any vertex that has a
            # bond with the vertex next to it
            next_vert = self.graph.get_vertex(self.curr_order[
                candidate_order + break_direction])
            possible_substitutes = next_vert.get_connections(
                retrieve_id=True)
            del possible_substitutes[possible_substitutes.index(candidate_id)]

            while possible_substitutes:
                # For the operation to be successful the candidate must fit
                # into the place where we want to put it
                second_candidate_id = np.random.choice(possible_substitutes)
                second_candidate_order = self.curr_order.index(
                    second_candidate_id)
                # [CT]: Another possible implementation is with try/catch.
                # [CT]: Maybe cleaner?
                # If the swap is possible, we consider it
                if self._check_swappable(candidate_id, candidate_order,
                                         second_candidate_order,
                                         break_direction
                                         ):
                    self._try_to_swap(candidate_id, candidate_order,
                                      second_candidate_id,
                                      second_candidate_order,
                                      break_direction)
                    break
                # Otherwise we delete the candidate and go on with the next
                del possible_substitutes[possible_substitutes.index(
                    second_candidate_id)]

            # Decrease temperature, increase iterations
            self.T *= self.alpha
            self.iter += 1.
            print_number = 10000
            if int(self.iter) % 10000 == 0:
                print('Performed {} iterations'.format(self.iter))
                elapsed = time.time()-start_time
                print('{} iterations per second'.format(print_number/elapsed))
                start_time = time.time()

        # Finally build the solution graph
        self.curr_solution = self.graph.build_graph_solution(
            self.curr_order[:-1])
        print(algos.evaluate_solution(self.curr_solution))
        return self.curr_solution

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times
        """
        for i in range(times):
            self.T = self.T_start
            self.iteration = 1
            self.solve()
        return self.solve()
