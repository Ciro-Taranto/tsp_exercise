#######################################
# Implementation of genetic algorithm #
#######################################
from graph import Graph
import numpy as np
import random
import operator
import algos
import matplotlib.pyplot as plt
from simulated_anneling import SimAnneal


class Genetic():
    def __init__(self, population_size=100, random_size=20, elite_size=None,
                 luck_ratio=0.5, mutation_rate=0.005, generations=10):
        self.population_size = population_size
        self.random_size = random_size
        self.elite_size = elite_size or random_size*0.25
        self.elite_size = int(self.elite_size)
        self.generations = generations
        self.luck_ratio = luck_ratio
        if self.luck_ratio < 0.0 or self.luck_ratio > 1.0:
            self.luck_ratio = 0.5
            print("Resetting luck_ratio to ", self.luck_ratio)
        self.mutation_rate = mutation_rate
        return None

    def instantiate_problem(self, locations, weights):
        self.locations = locations
        self.weights = weights
        self.graph = Graph(locations=self.locations, weights=self.weights)
        self.location_list = self.graph.adding_order
        self.number_of_locations = len(self.location_list)
        return True

    def create_random_route(self):
        return random.sample(self.location_list, self.number_of_locations)

    def initial_population(self):
        population = []
        for i in range(self.population_size):
            # This does not guarantee that all the members of the population
            #  are different [meaningless for large enough sample]
            population.append(self.create_random_route())
        return population

    def route_lenght(self, route):
        total_len = sum([self.graph.get_edge_weight(route[i], route[i+1])
                         for i in range(-1, self.number_of_locations-1)])
        return total_len

    def route_fitness(self, route):
        route_val = self.route_lenght(route)
        fitness = 1./route_val
        return fitness

    def rank_routes(self, population):
        fitness_results = {i: self.route_fitness(
            route) for i, route in enumerate(population)}
        return sorted(fitness_results.items(), key=operator.itemgetter(1),
                      reverse=True)

    def cumsum(self, l):
        """
        l is a list of type
        [(i1,val),(i2,val),...]
        """
        cs = [l[0][1]]
        for i in range(1, len(l)):
            cs.append(l[i][1]+cs[i-1])
        return cs

    def mating_pool_selection(self, ranked_routes, population):
        """
        Returns the mating pool elements and their fitness:
        ranked_routes(dict):(index:fitness)
        population(list): [candidate1,candidate2,...]
        """
        random_size = self.random_size
        elite_size = self.elite_size
        luck_ratio = self.luck_ratio
        selection_results = []
        summed_fitness = sum([r[1] for r in ranked_routes])
        cumulative_sum = self.cumsum(ranked_routes)  # From 0. to 1.
        # Worse elements have a higher value.
        # To add some spice, some of them will get lucky.
        lucky_guys = (1.-luck_ratio)*np.array(cumulative_sum)/summed_fitness
        random_luck = np.random.random(len(lucky_guys))
        lucky_guys = lucky_guys + random_luck
        lucky_guys[:elite_size] = 2.0

        # This is not efficient. Sorting twice
        lucky_order = np.argsort(lucky_guys)

        # by defult append the results in the elite_size
        for i in range(elite_size):
            selection_results.append(ranked_routes[i])
        # and the append the lucky ones
        for i in range(self.random_size):
            selection_results.append(ranked_routes[lucky_order[i]])

        mating_pool = [[population[i[0]], i[1]] for i in selection_results]
        return mating_pool

    def breed(self, parent1, parent2, mu=0.5):
        """
        Returns the offspring of two parents.
        parent1,parent2 are lists [parent,fitness]
        """
        # The parent with highest fitness contributes the most
        r = parent1[1]/(parent1[1]+parent2[1])
        gene_length = min(self.number_of_locations-1,
                          max(int(r*self.number_of_locations *
                                  random.gauss(1., mu)), 1))
        start_gene = random.randint(0, self.number_of_locations-1-gene_length)

        # create the gene
        gene = parent1[0][start_gene:start_gene+gene_length+1]
        missing = [i for i in parent2[0] if i not in gene]
        child = missing[:start_gene]+gene+missing[start_gene:]
        if len(child) != self.number_of_locations:
            print('Error in breeding funcion.')
            print('child of abnormal lenght:', len(child))
        return child

    def breed_population(self, mating_pool):
        """
        mating_pool(list): [individual, fitness]
        """
        children_population = []

        for i in range(self.elite_size):
            True
            children_population.append(mating_pool[i][0])

        for i in range(self.elite_size, self.population_size):
            [i1, i2] = np.random.choice(len(mating_pool), 2, replace=False)
            child = self.breed(mating_pool[i1], mating_pool[i2])
            children_population.append(child)

        return children_population

    def mutate(self, individual, super_mutation=None):
        """
        The mutation rate does not refer to the individual
        it refers to each location
        The mutation is just a swap of two elements
        """
        individual = individual.copy()
        if super_mutation is not None:
            mutation_rate = super_mutation
        else:
            mutation_rate = self.mutation_rate
        number_of_mutations = np.random.binomial(
            self.number_of_locations, mutation_rate)
        for i in range(number_of_mutations):
            [l1, l2] = np.random.choice(
                range(self.number_of_locations), 2, replace=False)
            individual[l1], individual[l2] = individual[l2], individual[l1]
        return individual

    def mutate_population(self, population):
        return [self.mutate(individual) for individual in population]

    def next_generation(self, current_generation, i, pr):
        ranked_routes = self.rank_routes(current_generation)
        self.progress.append(1./ranked_routes[0][1])
        mating_pool = self.mating_pool_selection(
            ranked_routes, current_generation)
        children_generation = self.breed_population(mating_pool)
        next_generation = self.mutate_population(children_generation)
        if pr:
            print('Champion of generation {}: {}'.format(
                i, 1/ranked_routes[0][1]))
        return next_generation

    def genetic_algorithm(self, plot=False, pr=False, hybrid=False):
        self.progress = []
        if hybrid:
            pop = self.hybrid_annealing()
        else:
            pop = self.initial_population()
        print([1./i[1] for i in self.rank_routes(pop)[:5]])

        for i in range(self.generations):
            pop = self.next_generation(pop, i, pr)

        rr = self.rank_routes(pop)
        best_route_index = rr[0][0]
        best_route = pop[best_route_index]
        sol = self.graph.build_graph_solution(best_route)
        if plot:
            plt.plot(self.progress, '-o')
        if not pr:
            print(1./rr[0][1])
        return sol

    def hybrid_annealing(self):
        sim = SimAnneal(self.locations, self.weights,
                        alpha=0.9995, stopping_iter=1e5)
        drosophila = sim.anneal().adding_order
        initial_population = [drosophila]

        # Set a relatively high number,
        # otherwise all the population will be the same!
        mutant_rate = 0.05
        for i in range(1, self.population_size):
            initial_population.append(self.mutate(
                drosophila, super_mutation=mutant_rate))
        return initial_population
