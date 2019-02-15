import random
import numpy as np
import matplotlib.pyplot as plt
from objects.graph import Graph
from objects.solver import Solver
from algorithms import algos


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
    def __init__(self, locations, edges, T=None, alpha=0.995,
                 stopping_T=1e-6, stopping_iter=100000, start=None,
                 start_solution=None):
        """
        Solver for simulated annealing. 
        Start from an approximate solution and propose moves.
        Approximate solution: chose vertex and pick the next
        Args: 
            locations(dict):{name:(x,y)}
            edges(dict):{('from','to'):weight}
            T(float): Starting temp, otherwise <w(approximated_sol)>.
            stopping_iter(int): maximum iteration 
            start(str): vert to start from
            start_solution(list):['v1','v2',...] order of visit of vertices
        """
        self.locations = locations
        self.N = len(locations)
        self.edges = edges
        self.graph = Graph(locations=locations, weights=edges)
        self.start = start if start is not None else next(
            iter(self.graph)).get_id()

        # We need a temperature scale if this is not given:
        # one possibility is the average distance between two edges
        self.alpha = alpha
        assert self.alpha < 1.
        self.stopping_T = stopping_T
        self.stopping_iter = stopping_iter
        self.iter = 1

        # Initialize the best solution to the one of the greedy algo
        if start_solution is not None:
            self.curr_solution = start_solution
        else:
            self.curr_solution = approximate_traveling_salesman(
                locations, edges, start=start)
        # The starting temperature is extremely importat:
        # It has to be fixed according to some length scale,
        # which in turns depends on the number of points
        self.curr_order = self.curr_solution.adding_order
        self.curr_solution_val = algos.evaluate_solution(
            self.graph.build_graph_solution(self.curr_order))
        edges = self.curr_solution.get_edges()
        self.T = T if T is not None else sum(
            [v for k, v in edges.items()])/len(edges.items())/2
        print(self.curr_solution_val)
        print("Starting Temperature = {}".format(self.T))
        self.T_start = self.T
        self.fitness_list = [self.curr_solution_val]
        self.best_list = [self.curr_solution_val]

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

    def solve(self):
        """
        Execute the solution
        """
        while self.T > self.stopping_T and self.iter < self.stopping_iter:

            # Find two candidates whose positions will be swapped
            # [TODO]: is N-2 correct? does it allow for last change?
            l, r = np.random.choice(range(1, self.N-1), 2, replace=False)
            # l = random.randint(1, self.N-2)
            # r = random.randint(1, self.N-2)
            if l > r:
                l, r = r, l
            first_node = self.curr_order[l]
            second_node = self.curr_order[r]

            # Find cost of breaking ties and creating new ones
            tie_break1 = self.edges[(first_node, self.curr_order[l-1])]
            tie_break2 = self.edges[(second_node, self.curr_order[r+1])]

            tie_add1 = self.edges[(first_node, self.curr_order[r+1])]
            tie_add2 = self.edges[(second_node, self.curr_order[l-1])]

            cost = (tie_add1 + tie_add2) - (tie_break1 + tie_break2)

            if self._accept(cost):
                # self._show_debug_info(cost)

                # If the candidate is accepted update order and current cost
                # print(cost)
                self.curr_order[l: r + 1] = self.curr_order[l:r+1][::-1]
                self.curr_solution_val += cost
                if self.curr_solution_val < min(self.fitness_list):
                    self.best_list.append(self.curr_solution_val)

            # Update the fitness list
            self.fitness_list.append(self.curr_solution_val)

            # Decrease temperature, increase iterations
            self.T *= self.alpha
            self.iter += 1.

        # Finally build the solution graph
        self.curr_solution = self.graph.build_graph_solution(self.curr_order)
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
        return self.anneal()
