import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
from objects.graph import Graph
from algorithms import algos
from algorithms.simulated_anneling import SimAnneal
from algorithms.constraint_satisfaction import ConstraintSatisfaction

class NewGenetic():
    def __init__(self, locations, edges, population_size=100,
                 random_size=20, elite_size=None,
                 luck_ratio=0.2, power=2, coeff=1.5,
                 mutation_rate=0.005, super_mutant_rate=0.05,
                 generations=10,
                 hybrid=False):
        """
        Solver for TSP with genetic algorithm.
        Start from a population and breed the fittest individuals.
        The algo will work in the space of edges rather than vertices.
        =======================================================================================
        A state in the population is represented as a list of {0,1},
        with 0 meaning that the edge is included and 1 meaning that the edge is not included.
        =======================================================================================
        Args:
            locations(dict):{name:(x,y)}
            edges(dict):{('from','to'):weight}
            population_size(int): n of individuals in the population
            random_size(int): n of indiduals that can reproduce
            elite_size(int): fittest individuals that get cloned.
                If None, 20% of random_size
            luck_ratio[0,1]: 0.0 pure fitness, 1.0 pure random
            mutation_rate[0,1]: p of mutation on single location
            super_mutant_rate[0,1]: p of mutation upon initialization
            generations(int): how many generations
            hybrid(bool): If True will start from a population obtained
                muting the simulated annealing solution
            power(int): not_visited_cost=(v-1)**power
                with power>1 will penalize more
                having "hubs" of locations visited too often
            coeff(int): the fitness will be evaluated as
                1/(route_length + coeff*not_visited_cost)
        """
        # Initialize the graph to study
        self.locations = locations
        self.edges = OrderedDict(edges)
        self.index_edge = {i: val for i, val in enumerate(self.edges.keys())}
        self.edge_weights = np.array([val for key,val in self.edges.items()])
        self.graph = Graph(locations=self.locations, weights=edges)
        self.location_list = self.graph.adding_order
        self.number_of_locations = len(self.location_list)
        self.number_of_edges = len(edges)

        # Initialize properties of the solver
        self.population_size = population_size
        self.random_size = random_size
        self.elite_size = elite_size or random_size*0.20
        self.elite_size = int(self.elite_size)
        self.generations = generations
        self.print_every = self.generations//5
        self.luck_ratio = luck_ratio
        self.hybrid = hybrid
        if self.luck_ratio < 0.0 or self.luck_ratio > 1.0:
            self.luck_ratio = 0.5
            print("Resetting luck_ratio to ", self.luck_ratio)
        self.mutation_rate = mutation_rate
        self.super_mutant_rate = super_mutant_rate
        self.power = power
        self.coeff = coeff
        self.progress = []
        self.best_route = []


    def _initialize_population(self):
        """
        Initialize the population
        :return: a list of lists, each element informing of the presence, or not,
        of the link in the solution
        """
        if self.hybrid:
            pre_solver = SimAnneal(self.locations, dict(self.edges),
                                   alpha=0.9995,
                                   stopping_iter=int(1e5),
                                   get_lucky=True)
        else:
            pre_solver = ConstraintSatisfaction(self.locations, dict(self.edges),
                                                lucky=True, luck_limit=int(1e5))
        pre_solution = pre_solver.solve()
        d = []
        for i in range(-1,len(pre_solution.adding_order)-1):
            d.append((
                     pre_solution.adding_order[i],
                     pre_solution.adding_order[i+1]))
        drosophila = np.array([1 if e in d else 0 for e in self.edges.keys()])
        return [self._fast_mutate(
                drosophila, self.super_mutant_rate) for i in range(self.population_size)]

    def _fast_mutate(self, individual, mutation_rate):
        """
        Mutate individual. Mutations will be fast, but mostly wrong.
        :param individual: The individual to mutate
        :param super_mutation: If None, mutate at standard rate.
        Otherwise at very high rate
        :return:
        """
        individual = individual.copy()
        number_of_mutations = np.random.binomial(
            self.number_of_locations, mutation_rate)
        active_bonds = np.where(individual==1)[0]
        inactive_bonds = np.where(individual != 1)[0]
        deactivate_bonds = np.random.choice(active_bonds, number_of_mutations, replace=False)
        activate_bonds = np.random.choice(inactive_bonds, number_of_mutations, replace=False)
        for i,j in zip(activate_bonds, deactivate_bonds):
            # I cannot get the slicing to work, high loss in efficiency
            individual[i] = 1
            individual[j] = 0
        assert np.sum(individual) == self.number_of_locations
        return individual

    def _mutant_schedule(self, generation):
        sigma = self.generations**2/10.
        return np.exp(-generation**2/sigma)*self.mutation_rate

    def _mutate_population(self, population, mutation_rate):
        return [self._fast_mutate(individual, mutation_rate) for individual in population]

    def _route_length(self,route):
        return np.sum(np.dot(route,self.edge_weights))

    def _visited_locations(self, route):
        """
        Utility to provide visited locations
        :param route: individual being analyzed
        :return: {location:[incoming-1,outgoing-1]}
        """
        # Initialized to -1 s.t. the desired value is 0
        visits = {key: [-1, -1] for key in self.locations.keys()}
        for index,val in enumerate(route):
            # Possible very inefficient
            edge = self.index_edge[index]
            visits[edge[0]][0] += val
            visits[edge[1]][1] += val
        return visits

    def _location_mismatch_cost(self, visits):
        s = [np.abs(val[0]) +
             np.abs(val[1])
             for key, val in visits.items()]
        errs = sum(s)
        return errs

    def get_edges(self, individual):
        return [self.index_edge[i] for i in np.where(individual == 1.)[0]]

    def _count_connected_sets(self,individual):
        """
        Number of connected sets, evaluated iteratively
        [TODO]:  Implement with laplacian matrix
        :param individual:
        :return: the number of connected sets
        """
        visited = self.location_list.copy()
        used_edges = self.get_edges(individual)
        connected_sets = 0
        while len(visited) > 0:
            start = visited[0]
            self._transverse(start, visited, used_edges)
            connected_sets += 1
        return connected_sets

    def _transverse(self, node, visited, individual):
        del visited[visited.index(node)]
        waiting_list = [edge[1] for edge in individual if edge[0] == node]
        for item in waiting_list:
            if item in visited:
                return self._transverse(item, visited, individual)
        return

    def _rank_routes(self,population):
        # Compute the length of the routes

        pop_route_len = {i: self._route_length(route)
                                   for i, route in enumerate(population)}

        max_len = np.max([val for key,val in pop_route_len.items()])

        pop_vis = {i: self._visited_locations(route)
                             for i, route in enumerate(population)}

        location_mismatch_cost = {i: self._location_mismatch_cost(pop_vis[i])
                                  for i in pop_vis.keys()}

        connected_sets = {i: self._count_connected_sets(route)
                          for i, route in enumerate(population)}

        cost = {i: pop_route_len[i]/max_len + self.coeff * (connected_sets[i]**self.power + location_mismatch_cost[i])
                for i in pop_route_len.keys()}

        fitness_results = {i: 1./cost[i]
                           for i in pop_route_len.keys()}

        return sorted(fitness_results.items(), key=operator.itemgetter(1),
                      reverse=True)

    def _mating_pool_selection(self, ranked_routes, population):
        """
        Simplified version of mating pool selection
        ranked_routes(dict):(index:fitness)
        population(list): [candidate1,candidate2,...]
        """
        selection_results = ranked_routes[:self.elite_size].copy()
        other_candidates = ranked_routes[self.elite_size:]
        p = [r[1]**2 for r in other_candidates] # To the power of 2 to further favor the most fit
        p = p/sum(p) # normalize to get a probability
        lucky_individuals = np.random.choice(range(len(other_candidates)), self.random_size, p=p,replace=False)
        selection_results += [other_candidates[i] for i in lucky_individuals]
        mating_pool = [[population[i[0]], i[1]] for i in selection_results]
        return mating_pool

    def _breed_population(self, mating_pool):
        """
        Given the population of individuals to reproduce and their fitness
        breed them to obtain the new generation
        :param mating_pool: [[individual1,fitness1],[i2,f2],...]
        :return: children generation [child1, child2, child3,...]
        """
        children_population = [mating_pool[i][0] for i in range(self.elite_size)]

        for i in range(self.elite_size, self.population_size):
            [i1, i2] = random.sample(range(len(mating_pool)), 2)
            children_population.append(self._fast_breed(mating_pool[i1], mating_pool[i2]))
        return children_population

    def _fast_breed(self, parent1, parent2):
        # The parent with highest fitness contributes the most
        r = parent1[1]/(parent1[1]+parent2[1])

        # Gene length is between 1 and N-1
        gene_length = max(1, min(self.number_of_locations-1, int(r*self.number_of_locations)))

        # Create the gene
        active_p1 = np.where(parent1[0]==1)[0] # From these we need gene_length
        active_p2 = np.where(parent2[0]==1)[0] # From these we need self.number_of_locations-gene_length

        # There is no concept of ordering in the edges
        # Since this is random the only thing we care about is taking them
        gene = np.random.choice(active_p1, gene_length, replace=False)
        allowed = [ind for ind in active_p2 if ind not in gene]
        # which implementation is faster?
        # allowed = list(set(active_p2) - set(gene))
        rest_of_individual = np.random.choice(allowed, self.number_of_locations-gene_length, replace=False)
        active_individual = list(gene) + list(rest_of_individual)
        individual = np.array([1 if i in active_individual else 0 for i in range(self.number_of_edges)])
        assert np.sum(individual) == self.number_of_locations
        return individual

    def _next_generation(self, population, mutation_rate):
        """
        Evolve population
        :param population: population of individuals
        :return: next generation
        """
        ranked_routes = self._rank_routes(population)

        self.progress.append(1. / ranked_routes[0][1])

        mating_pool = self._mating_pool_selection(
            ranked_routes, population)

        children_generation = self._breed_population(mating_pool)

        next_generation = self._mutate_population(children_generation, mutation_rate)

        return next_generation

    def solve(self, plot=False):
        """
        Solve the problem instance, with the set of parameters given at instantiation.
        Args:
            plot: if True at the end will provide a plot of the
                best fitness along generations
        """
        pop = self._initialize_population()
        old_time = time.time()
        #  Here all the evolution happens
        for i in range(1,self.generations+1):
            if i % self.print_every == 0:
                old_time = self._plot_and_print(pop, old_time)
            pop = self._next_generation(pop, self._mutant_schedule(i))

        # Just to return the results
        rr = self._rank_routes(pop)
        best_route_index = rr[0][0]
        best_route = pop[best_route_index]
        sol = self.graph.build_graph_solution_from_edges(self.get_edges(best_route))
        self.best_route = best_route
        if plot:
            plt.plot(self.progress, '-o')
        return sol

    def _plot_and_print(self, pop, old_time):
        """
        Simple infos repeted after some generations
        :param pop:
        :param old_time:
        :return: time.time() in order to keep the process running
        """
        rr = self._rank_routes(pop)
        best_route_index = rr[0][0]
        best_route = pop[best_route_index]
        sol = self.graph.build_graph_solution_from_edges(self.get_edges(best_route))
        print(algos.evaluate_solution(sol), rr[0][1])
        print('Violations:', self._location_mismatch_cost(self._visited_locations(best_route)))
        print('Connected sets:', self._count_connected_sets(best_route))
        print("Time per generation={}".format((time.time() - old_time) / self.print_every))
        # sol.render()
        return time.time()


