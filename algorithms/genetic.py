import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import time

from objects.graph import Graph
from algorithms import algos
from algorithms.simulated_anneling import SimAnneal
from algorithms.constraint_satisfaction import ConstraintSatisfaction


class Genetic():
    def __init__(self, graph, population_size=100,
                 random_size=20, elite_size=None,
                 luck_ratio=0.2,
                 mutation_rate=0.005, super_mutant_rate=0.05,
                 generations=10,
                 hybrid=False, **kwargs):
        """
        Solver for TSP with genetic algorithm.
        Start from a population and breed the fittest individuals.
        Breeding and mutation strategies are fixed.
        The algo will work in the space of edges rather than vertices.
        A state in the initial population
        is going to be represented as a sequence of N consecutive edges.
        By mutating this property will be lost.
        Args:
            :param graph: Graph, instance of graph to solve
            population_size(int): n of individuals in the population
            random_size(int): n of indiduals that can reproduce
            elite_size(int): fittest individuals that get cloned.
                If None, 20% of random_size
            luck_ratio[0,1]: 0.0 pure fitness, 1.0 pure random
            mutation_rate[0,1]: p of mutation on single location
            generations(int): how many generations
            hybrid(bool): If True will start from a population obtained
                muting the simulated annealing solution
        """
        # Initialize the graph to study
        self.graph = graph
        self.locations = self.graph.get_locations()
        self.edges = self.graph.get_edges(get_all=True)
        self.edge_sorter = {val: i for i, val in enumerate(self.edges.keys())}
        self.location_list = self.graph.adding_order
        self.number_of_locations = len(self.location_list)

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
        self.mutations = 0
        self.progress = []
        self.best_route = []

    @classmethod
    def from_locations_and_edges(cls, locations, edges):
        graph = Graph(locations=locations, weights=edges)
        return Genetic(graph)

    def _mutant_schedule(self, generation):
        sigma = self.generations**2/10.
        return np.exp(-generation**2/sigma)*self.mutation_rate

    def _route_length(self, route):
        return sum([self.edges[i] for i in route])

    def _visited_locations(self, route):
        """
        Utility to provide visited locations
        :param route: individual being analyzed
        :return: {location:[incoming-1,outgoing-1]}
        """
        # Initialized to -1 s.t. the desired value is 0
        visits = {key: [-1, -1] for key in self.locations.keys()}
        for edge in route:
            visits[edge[0]][0] += 1
            visits[edge[1]][1] += 1
        return visits

    def _location_mismatch_cost(self, route):
        visits = self._visited_locations(route)
        s = [np.abs(val[0]) +
             np.abs(val[1])
             for key, val in visits.items()]
        errs = sum(s)
        return errs

    def _rank_routes(self, population):
        # Working under the assumption that all routes are valid
        fitness_results = {i: 1./(self._route_length(
            route)*self._count_connected_sets(route)**2)
                           for i, route in enumerate(population)}
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
        p = [r[1] for r in other_candidates]
        p = p/sum(p) # normalize to get a probability
        lucky_individuals = np.random.choice(range(len(other_candidates)), self.random_size, p=p, replace=False)
        selection_results += [other_candidates[i] for i in lucky_individuals]
        mating_pool = [[population[i[0]], i[1]] for i in selection_results]
        return mating_pool

    def _breed(self, parent1, parent2, mu=0.5):
        """
        Returns the offspring of two parents.
        parent1,parent2 are lists [parent,fitness]
        """
        # The parent with highest fitness contributes the most
        r = parent1[1]/(parent1[1]+parent2[1])
        # Gene length is between 1 and N-1
        gene_length = min(self.number_of_locations-1,
                          max(int(r*self.number_of_locations *
                                  random.gauss(1., mu)), 1))
        # Create the gene
        start_gene = random.randint(0, self.number_of_locations - 1 - gene_length)
        gene = parent1[0][start_gene:start_gene+gene_length+1]
        removed_gene = parent2[0][start_gene:start_gene+gene_length+1]
        remaining_gene = parent2[0][:start_gene]+parent2[start_gene+gene_length+1:]

        # Find the duplicated bonds:
        whitelist = [e for e in remaining_gene if (e not in gene and tuple(reversed(e)) not in gene)]
        i = 0
        while len(whitelist) < (self.number_of_locations - gene_length - 1):
            i += 1
            if i > 10000:
                raise Exception("Infinite loop")
            if len(removed_gene) > 0:
                i = np.random.choice(range(len(removed_gene)))
                if removed_gene[i] not in gene and tuple(reversed(removed_gene[i])) not in gene:
                    whitelist.append(removed_gene[i])
                del removed_gene[i]
            else:
                # Small risk for an infinite loop!
                e = random.choice(list(self.edges.keys()))
                c1 = e not in gene and tuple(reversed(e)) not in gene
                c2 = e not in whitelist and tuple(reversed(e)) not in whitelist
                if c1 and c2:
                    whitelist.append(e)

        # Create the child
        child = gene + whitelist
        assert len(child) == self.number_of_locations

        # Remove edges from rich nodes and give them to poor ones
        child = self._robin_hood(child)
        return child

    def _reconnect(self, individual, neg, pos, visits, ind):
        """
        Try to fix the individual, by removing excess connections and
        increasing defect connections
        :param individual: The set of edges constituting the individual
        :param neg: Set of not visited locations
        :param pos: Set of locations visited too much
        :param visits: {location: [#incoming, #outgoing]}
        :param ind: [0,1] the index under consideration
        :return:
        """
        assert sum([v for k, v in neg.items()]) == -sum([v for k, v in pos.items()])
        unsafe = False
        for key in neg.keys():
            if len(pos) == 0:
                raise ValueError('No excess edges but some defect ones?')
            # Scan the locations with extra connections.
            # To find one that suffices
            for lcandidate in random.sample(list(pos.keys()),len(list(pos.keys()))):
                rcandidates = [e[ind-1] for e in individual
                               if e[ind] == lcandidate]
                if ind == 0:
                    candidates = [(key, e) for e in rcandidates
                                  if ((key, e) in self.edges.keys()
                                  and (e, key) not in individual)]
                if ind == 1:
                    candidates = [(e, key) for e in rcandidates
                                  if ((e, key) in self.edges.keys()
                                  and (key, e) not in individual)]

                if len(candidates) > 0:
                    winner = random.choice(candidates)
                    if ind == 0:
                        del individual[individual.index((lcandidate, winner[1]))]
                    else:
                        del individual[individual.index((winner[0], lcandidate))]
                    individual.append(winner)
                    neg[key] += 1
                    pos[lcandidate] -= 1
                    if pos[lcandidate] == 0:
                        del pos[lcandidate]

                    break
            if neg[key] != 0:
                # Probably the location has only a few connections.
                # Then a connection must be forced

                # This is not going to be efficient:
                # Delete any of the connections:
                for conn in np.random.permutation(individual):
                    connection = tuple(conn)
                    if connection[ind] in pos.keys():
                        pos[connection[ind]] -= 1
                        if pos[connection[ind]] == 0:
                            del pos[connection[ind]]
                    del individual[individual.index(connection)]
                    break

                candidates = self.graph.get_vertex(key).get_connections(retrieve_id=True)
                candidates = [c for c in candidates
                              if (c, key) not in individual
                              and (key, c) not in individual]
                # Force one of these connections
                if ind == 0:
                    individual.append((key, np.random.choice(candidates)))
                elif ind == 1:
                    individual.append((np.random.choice(candidates), key))
                neg[key] += 1
                unsafe = True
        return neg, pos, individual, unsafe

    def _robin_hood(self, individual):
        """
        Can fix an individual by recursively reconnectinig locations
        requiring that the locations with too many visits give to the ones
        with too less visits
        :param individual: individual to be fixed
        :return: fixed individual
        """
        unsafe  = True
        iterations = 0
        # Computationally this is very inefficient
        # visits = {key: val for key, val in visits.items() if val != [0, 0]}
        while unsafe:
            visits = self._visited_locations(individual)
            lneg, lpos, rneg, rpos = _add_pos_neg(visits)
            lneg, lpos, individual, u1 = self._reconnect(individual, lneg, lpos, visits, 0)
            rneg, rpos, individual, u2 = self._reconnect(individual, rneg, rpos, visits, 1)
            unsafe = u1 or u2
            iterations += 1
            if iterations >= int(1e6):
                raise Exception("Could not fix the individual")
        for checking in [lneg,lpos,rneg,rpos]:
            if sum([val for key,val in checking.items()]) != 0:
                raise Exception('Error in {}'.format(checking))
        return individual

    def _breed_population(self, mating_pool):
        """
        mating_pool(list): [individual, fitness]
        """
        children_population = [mating_pool[i][0] for i in range(self.elite_size)]

        for i in range(self.elite_size, self.population_size):
            [i1, i2] = random.sample(range(len(mating_pool)), 2)
            children_population.append(self._breed(mating_pool[i1], mating_pool[i2]))

        return children_population

    def _cycle_ordering(self, individual):
        new_individual = [individual[0]]
        already_visited = [new_individual[0][0]]
        while len(already_visited) < self.number_of_locations:
            next_location = new_individual[-1][1] # Pick the last visited location
            if next_location not in already_visited:
                already_visited.append(next_location)
            else:
                return individual # If something is broken do not change individual
            connection_edge = [e for e in individual if e[0] == next_location]
            if len(connection_edge) != 1:
                return individual # Also here, the routine cannot fixed something already broken
            new_individual.append(connection_edge[0])
        assert len(new_individual) == len(individual)
        return new_individual

    def _mutate(self, individual, mutation_rate):
        """
        Only consistent mutations will be allowed.
        This will force the algo towards some beam
        simulated annealing
        """
        number_of_mutations = np.random.binomial(
            self.number_of_locations, mutation_rate)

        # Slow down mutations but guarantee that they are order preserving
        # (this should be the case only for individuals that already preserve the order)
        individual = self._cycle_ordering(individual)

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
        return individual

    def _sort_individual(self, individual):
        individual = {edge: self.edge_sorter[edge] for edge in individual}
        return sorted(individual, key=individual.get)

    def _sort_population(self, population):
        return [self._sort_individual(individual) for individual in population]

    def _mutate_population(self, population,mutant_rate):
        return [self._mutate(individual, mutant_rate) for individual in population]

    def _next_generation(self, current_generation,mutant_rate):
        """
        :param current_generation: population being evolved
        :return: next generation after breeding and mutating
        """
        ranked_routes = self._rank_routes(current_generation)
        # Keep track of the progress
        self.progress.append(1./ranked_routes[0][1])
        mating_pool = self._mating_pool_selection(
            ranked_routes, current_generation)
        children_generation = self._breed_population(mating_pool)
        next_generation = self._mutate_population(children_generation, mutant_rate)
        next_generation = self._sort_population(next_generation)
        return next_generation

    def solve(self, plot=False):
        """
        Solve the problem instance, with the set of parameters given at instantiation.
        Args:
            plot: if True at the end will provide a plot of the
                best fitness along generations
            pr: if True will print some output
        """
        self.mutations = 0
        pop = self._initialize_population()
        start = time.time()
        for i in range(1,self.generations+1):
            if i % self.print_every == 0:
                # self._plot_and_print(pop)
                print("Time per generation={}".format((time.time()-start)/self.print_every))
                start = time.time()
            pop = self._next_generation(pop, self._mutant_schedule(i))

        rr = self._rank_routes(pop)
        best_route_index = rr[0][0]
        best_route = pop[best_route_index]
        sol = self.graph.build_graph_solution_from_edges(best_route)
        self.best_route = best_route
        if plot:
            plt.plot(self.progress, '-o')
        return sol

    def _plot_and_print(self, pop):
        rr = self._rank_routes(pop)
        best_route_index = rr[0][0]
        best_route = pop[best_route_index]
        sol = self.graph.build_graph_solution_from_edges(best_route)
        print(algos.evaluate_solution(sol), rr[0][1])
        print('Violations:', self._location_mismatch_cost(best_route))
        print('Connected sets:', self._count_connected_sets(best_route))
        sol.render()
        return True

    def _initialize_population(self):
        if self.hybrid:
            pre_solver = SimAnneal(self.locations, self.edges,
                                   alpha=0.9995,
                                   stopping_iter=int(1e5),
                                   get_lucky=True)
        else:
            pre_solver = ConstraintSatisfaction(self.graph, lucky=True, luck_limit=int(1e5))
        pre_solution = pre_solver.solve()
        d = []
        for i in range(-1,len(pre_solution.adding_order)-1):
            d.append((
                     pre_solution.adding_order[i],
                     pre_solution.adding_order[i+1]))
        return [self._mutate(
                d, self.super_mutant_rate) for i in range(self.population_size)]

    def _count_connected_sets(self,individual):
        visited = {l: False for l in self.location_list}
        connected_sets = 0
        all_visited = False
        while not all_visited:
            start = [key for key,val in visited.items() if val is False][0]
            self._transversal(start, visited, individual)
            all_visited = all(val for key, val in visited.items())
            connected_sets += 1
        return connected_sets

    def _transversal(self, node, visited, individual):
        visited[node] = True
        waiting_list = [edge[1] for edge in individual if edge[0] == node]
        for item in waiting_list:
            if not visited[item]:
                return self._transversal(item, visited, individual)
        return


def _add_pos_neg(visits):
    """
    Create the positive/negative dictionaries
    """
    negatives = [{}, {}]
    positives = [{}, {}]
    for key, val in visits.items():
        for i,v in enumerate(val):
            if v > 0:
                positives[i][key] = v
            elif v < 0:
                negatives[i][key] = v
    return negatives[0], positives[0], negatives[1], positives[1]



