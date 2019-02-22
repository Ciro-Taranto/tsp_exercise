import numpy as np
import random
import operator
import matplotlib.pyplot as plt
from objects.solver import Solver

from objects.graph import Graph
from algorithms import algos
from algorithms.simulated_anneling import SimAnneal


class Genetic():
    def __init__(self, locations, edges, population_size=100,
                 random_size=20, elite_size=None,
                 luck_ratio=0.2, power=1, coeff=1,
                 mutation_rate=0.005, generations=10,
                 hybrid=False):
        """
        Solver for TSP with genetic algorithm.
        Start from a population and breed the fittest individuals.
        Breeding and mutation strategies are fixed.
        The algo will work in the space of edges rather than vertices.
        A state in the initial population
        is going to be represented as a sequence of N consecutive edges.
        By mutating this property will be lost.
        Args:
            locations(dict):{name:(x,y)}
            edges(dict):{('from','to'):weight}
            population_size(int): n of individuals in the population
            random_size(int): n of indiduals that can reproduce
            elite_size(int): fittest individuals that get cloned.
                If None, 20% of random_size
            luck_ratio[0,1]: 0.0 pure fitness, 1.0 pure random
            mutation_rate[0,1]: p of mutation on single location
            generations(int): how many generations
            hybrid(bool): If True will start from a population obtained
                muting the simulated annealing solution
            power(int): not_visited_cost=(v-1)**power
                with power>1 will penalize more
                having "hubs" of locations visited too often
            coeff(int): the fitness will be evaluated as
                1/(route_length + coeff*not_visited_cost)
        """
        self._instantiate_problem(locations, edges)  # not elegant
        self.population_size = population_size
        self.random_size = random_size
        self.elite_size = elite_size or random_size*0.25
        self.elite_size = int(self.elite_size)
        self.generations = generations
        self.luck_ratio = luck_ratio
        self.hybrid = hybrid
        if self.luck_ratio < 0.0 or self.luck_ratio > 1.0:
            self.luck_ratio = 0.5
            print("Resetting luck_ratio to ", self.luck_ratio)
        self.mutation_rate = mutation_rate
        self.power = power
        self.coeff = coeff
        self.mutations = 0
        return None

    def _instantiate_problem(self, locations, weights):
        self.locations = locations
        self.edges = weights
        # not the nicest way... but still
        self.edge_sorter = {val: i for i, val in enumerate(self.edges.keys())}
        self.graph = Graph(locations=self.locations, weights=self.edges)
        self.location_list = self.graph.adding_order
        self.number_of_locations = len(self.location_list)
        return True

    def _select_candidate(self, candidate_vert, individual):
        # Try to pick one edge that guarantees continuity
        while candidate_vert:
            v_id = np.random.choice(candidate_vert)
            candidate = (last_visited_vertex_id, v_id)
            if candidate not in individual:
                return candidate
            del candidate_vert[candidate_vert.index(v_id)]
        # If the above condition cannot be met
        # Pick one edge at random
        candidates = set(self.edges.keys())-set(individual)
        return random.choice(list(candidates))

    def _create_random_route(self):
        """
        A random route is a random selection of edges
        """
        unsorted_route = random.sample(
            self.edges.keys(), self.number_of_locations)
        sorter = {edge: self.edge_sorter[edge] for edge in unsorted_route}
        return sorted(sorter, key=sorter.get)

    def _initial_population(self):
        return [self._create_random_route()
                for i in range(self.population_size)]

    def _route_lenght(self, route):
        return sum([self.edges[i] for i in route])

    def _visited_locations(self, route):
        # Initialized to -1 s.t. the desired value is 0
        visits = {key: [-1, -1] for key in self.locations.keys()}
        duplicated = {edge: sum([1 for e in route if e == edge])
                      for edge in route}
        duplicated_cost = sum([val-1 for key, val in duplicated.items()])
        doubling_cost = 0
        for edge in route:
            visits[edge[0]][0] += 1
            visits[edge[1]][1] += 1
            if edge[::-1] in route:
                doubling_cost += 1
        return visits, doubling_cost + duplicated_cost

    def _lmc(self, route):
        """
        lmc stands for location mismatch cost
        """
        visits, doubling_cost = self._visited_locations(route)
        s = [np.abs(val[0])**self.power +
             np.abs(val[1])**self.power
             for key, val in visits.items()]
        return sum(s) + doubling_cost

    def _rank_routes(self, population):
        len_results = {i: self._route_lenght(
            route) for i, route in enumerate(population)}
        mismatch_results = {i: self._lmc(
            route) for i, route in enumerate(population)}
        norm1 = max([val for key, val in len_results.items()])
        norm2 = max([val for key, val in mismatch_results.items()])
        norm2 = norm2 if norm2 > 0 else 1.
        fitness_results = {
            i: 1./(len_results[i]/norm1 +
                   self.coeff*mismatch_results[i]/norm2
                   ) for i in len_results.keys()}
        return sorted(fitness_results.items(), key=operator.itemgetter(1),
                      reverse=True)

    def _cumsum(self, l):
        """
        l is a list of type
        [(i1,val),(i2,val),...]
        """
        cs = [l[0][1]]
        for i in range(1, len(l)):
            cs.append(l[i][1]+cs[i-1])
        return cs

    def _mating_pool_selection(self, ranked_routes, population):
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
        cumulative_sum = self._cumsum(ranked_routes)  # From 0. to 1.
        # Worse elements have a higher value.
        # To add some spice, some of them will get lucky.
        lucky_guys = (1.-luck_ratio)*np.array(cumulative_sum)/summed_fitness
        random_luck = np.random.random(len(lucky_guys))
        lucky_guys = lucky_guys + luck_ratio * random_luck
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

    def _breed(self, parent1, parent2, mu=0.5):
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

        # Create the gene
        gene = parent1[0][start_gene:start_gene+gene_length+1]

        # Check if any of the edges in the gene is present elsewhere:
        # Avoid edges duplication
        p = parent2[0][:start_gene] + parent2[0][start_gene+gene_length+1:]
        duplicated_edges = [e for e in p
                            if e in gene]
        candidate_substitutes = [
            e for e in parent2[start_gene:start_gene+gene_length+1]
            if e not in gene]
        if len(candidate_substitutes) >= len(duplicated_edges):
            substitutes = random.sample(
                candidate_substitutes, len(duplicated_edges))
        else:
            # This will not guarantee e complete deduplication
            # But should still be fine enough
            substitutes = candidate_substitutes + \
                random.sample(self.edges.keys(), len(
                    duplicated_edges)-len(candidate_substitutes))

        child = parent2[0][:start_gene]+gene + \
            parent2[0][start_gene+gene_length+1:]

        # Remove edges from rich nodes and give them to poor ones
        child = self._robin_hood(child, self._visited_locations(child)[0])
        return child

    def _add_pos_neg(self, visits):
        """
        Create the positive/negative dictionaries
        """
        negatives = [{}, {}]
        positives = [{}, {}]
        for key, val in visits.items():
            for i in range(len(val)):
                if val[i] > 0:
                    positives[i][key] = val[i]
                elif val[i] < 0:
                    negatives[i][key] = val[i]
        return negatives[0], positives[0], negatives[1], positives[1]

    def _reconnect(self, individual, neg, pos, visits, ind):
        # Again this function is bad because changes its internal state
        for key in neg.keys():
            if len(pos) == 0:
                break
            # find one with a positive value
            candidate = random.choice(list(pos.keys()))
            rcandidates = [e[ind-1] for e in individual
                           if e[ind] == candidate]
            if ind == 0:
                both_candidates = [(candidate, e) for e in rcandidates
                                   if (candidate, e) in self.edges.keys()]
            if ind == 1:
                both_candidates = [(e, candidate) for e in rcandidates
                                   if (e, candidate) in self.edges.keys()]
            if len(both_candidates) > 0:
                winner = random.choice(both_candidates)
            if ind == 0:
                del individual[individual.index((candidate, winner[1]))]
                individual.append(winner)
            else:
                del individual[individual.index((winner[0], candidate))]
                individual.append(winner)
            neg[key] += 1
            pos[candidate] -= 1
            if pos[candidate] == 0:
                del pos[candidate]

            visits[key][ind] += 1
            visits[candidate][ind] -= 1

        return True

    def _robin_hood(self, individual, visits):
        # Computationally this is very inefficient
        visits = {key: val for key, val in visits.items() if val != [0, 0]}
        lneg, lpos, rneg, rpos = self._add_pos_neg(visits)
        self._reconnect(individual, lneg, lpos, visits, 0)
        self._reconnect(individual, rneg, rpos, visits, 1)
        visits = {key: val for key, val in visits.items() if val != [0, 0]}
        return individual

    def _breed_population(self, mating_pool):
        """
        mating_pool(list): [individual, fitness]
        """
        children_population = []

        for i in range(self.elite_size):
            children_population.append(mating_pool[i][0])

        for i in range(self.elite_size, self.population_size):
            [i1, i2] = random.sample(
                range(len(mating_pool)), 2)
            child = self._breed(mating_pool[i1], mating_pool[i2])
            children_population.append(child)

        return children_population

    def _inconsistent_mutate(self, individual, super_mutation=None):
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
            individual[random.randint(0, len(individual)-1)] = random.choice(
                list(self.edges.keys()))
        individual = {edge: self.edge_sorter[edge] for edge in individual}
        return sorted(individual, key=individual.get)

    def _mutate(self, individual, super_mutation=None):
        """
        Only consistent mutations will be allowed.
        This will force the algo towards some beam
        stimulated annealing
        """
        individual = individual.copy()
        if super_mutation is not None:
            mutation_rate = super_mutation
        else:
            mutation_rate = self.mutation_rate
        number_of_mutations = np.random.binomial(
            self.number_of_locations, mutation_rate)

        for i in range(number_of_mutations):
            bond_to_break = individual[random.randint(0, len(individual)-1)]
            left = self.graph.get_vertex(bond_to_break[0])
            right = self.graph.get_vertex(bond_to_break[1])
            connections_left = left.get_connections(retrieve_id=True)
            del connections_left[connections_left.index(bond_to_break[1])]
            connections_right = right.get_connections(retrieve_id=True)
            del connections_right[connections_right.index(bond_to_break[0])]
            candidates = []
            for r2 in connections_left:
                try:
                    l2 = [e[0] for e in individual if e[1] == r2][0]
                except:
                    l2 = None
                if l2 in connections_right:
                    candidates.append((l2, r2))
            if len(candidates) > 0:
                bond_to_break2 = random.choice(candidates)
                # mutate!
                self.mutations += 1
                individual[individual.index(bond_to_break)] = (
                    bond_to_break2[0], bond_to_break[1])
                individual[individual.index(bond_to_break2)] = (
                    bond_to_break[0], bond_to_break2[1])

        individual = {edge: self.edge_sorter[edge] for edge in individual}
        return sorted(individual, key=individual.get)

    def _mutate_population(self, population):
        return [self._mutate(individual) for individual in population]

    def _next_generation(self, current_generation, i, pr):
        ranked_routes = self._rank_routes(current_generation)
        self.progress.append(1./ranked_routes[0][1])
        mating_pool = self._mating_pool_selection(
            ranked_routes, current_generation)
        children_generation = self._breed_population(mating_pool)
        next_generation = self._mutate_population(children_generation)
        if pr:
            print('Champion of generation {}: {}'.format(
                i, 1/ranked_routes[0][1]))
        return next_generation

    def solve(self, plot=False, pr=False):
        """
        Solve the problem instance.
        plot: if True at the end will provide a plot of the
            best fitness along generations
        pr: if True will print some output
        hybrid: if True
        """
        self.mutations = 0
        self.progress = []
        if self.hybrid:
            pop = self.hybrid_annealing()
        else:
            pop = self._initial_population()
        for i in range(self.generations):
            if i % 500 == 0:
                self._plot_and_print(pop)
            pop = self._next_generation(pop, i, pr)

        rr = self._rank_routes(pop)
        best_route_index = rr[0][0]
        best_route = pop[best_route_index]
        sol = self.graph.build_graph_solution_from_edges(best_route)
        if plot:
            plt.plot(self.progress, '-o')
        return sol

    def _plot_and_print(self, pop):
        rr = self._rank_routes(pop)
        best_route_index = rr[0][0]
        best_route = pop[best_route_index]
        sol = self.graph.build_graph_solution_from_edges(best_route)
        print(algos.evaluate_solution(sol), rr[0][1])
        print('Violations:', self._lmc(best_route))
        sol.render()
        return True

    def hybrid_annealing(self):
        sim = SimAnneal(self.locations, self.edges,
                        alpha=0.9995, stopping_iter=1e5,
                        get_lucky=True)
        drosophila = sim.solve()
        d = []
        for i in range(-1, len(drosophila.adding_order)-1):
            d.append(
                (drosophila.adding_order[i], drosophila.adding_order[i+1]))
        drosophila.render()
        drosophila = d
        print("Violations on drosophila:", self._lmc(drosophila))
        initial_population = [drosophila]
        # Set a relatively high number,
        # otherwise all the population will be the same!
        mutant_rate = 0.02
        for i in range(1, self.population_size):
            initial_population.append(self._mutate(
                drosophila, super_mutation=mutant_rate))
        print("Violations on population", min(
            [self._lmc(i) for i in initial_population]))
        return initial_population
