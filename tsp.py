import yaml
import torch
import pylab as plt
import pickle
import time
import argparse
import sys

from algorithms.continuum import Continuum
from algorithms.genetic import Genetic
from algorithms.search import TravelingSalesman
from algorithms.simulated_anneling import SimAnneal
from algorithms.constraint_satisfaction import ConstraintSatisfaction
from algorithms.algos import evaluate_solution
from problems.map_generator import random_graph_layout
from objects.graph import Graph


SOLVER = dict()
SOLVER['search'] = TravelingSalesman
SOLVER['continuum'] = Continuum
SOLVER['genetic'] = Genetic
SOLVER['annealing'] = SimAnneal

OPTIMIZER = dict()
OPTIMIZER['Adam'] = torch.optim.Adam
OPTIMIZER['SGD'] = torch.optim.SGD

parser = argparse.ArgumentParser(description='Solve an instance of TSP.')

# Optional arguments
parser.add_argument('--to_file', dest='dump_filename', default=None,
                    help='Pickle the generated instance of graph to the file "DUMP_FILENAME_info.p"')

parser.add_argument('--from_file', dest='load_filename', default=None,
                    help='Use the graph pickled at LOAD_FILENAME.')

parser.add_argument('--config_file', dest='config_filename', default='config.yml',
                    help='Use CONFIG_FILENAME as config file (instead of "config.yml"). Must be yaml formatted.')

parser.add_argument('--number_of_locations', dest='number_of_locations', type=int,
                    default=20, help='Number of collection points.')

parser.add_argument('--number_of_deposits', dest='number_of_deposits', type=int,
                    default=2, help='Number_of_deposits.')

parser.add_argument('--fraction_of_edges', dest='number_of_arcs', type=float,
                    default=.5, help='Fraction of active edges.')

parser.add_argument('--traffic', dest='traffic', type=float,
                    default=.75, help='Traffic constant: if 0, euclidean distance, if 1 fully random.')

parser.add_argument('--solver', dest='solver', type=str,
                    default=None, help='Type of the solver. If this is set it overrides the solver in the configfile.')

args = parser.parse_args()

with open(args.config_filename, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Define the map layout, and relative graph
if args.load_filename is None:
    locations, edges, weights = random_graph_layout(args.number_of_locations,
                                                    args.number_of_deposits,
                                                    number_of_arcs=args.number_of_arcs,
                                                    traffic=args.traffic)
    graph = Graph(locations=locations, weights=weights)
    if args.dump_filename is not None:
        dump_filename = args.dump_filename
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        infostring = '_solution_locations{}_edges{}_traffic{}_time{}.p'.format(
            graph.num_vertices, len(graph.get_edges(get_all=True)), args.traffic, timestamp)
        dump_filename = dump_filename.split('.')[0]
        dump_filename = 'data/' + dump_filename + infostring
        pickle.dump(graph, open(dump_filename, 'wb'))
else:
    graph = pickle.load(open(args.load_filename, 'rb'))
    if not isinstance(graph, Graph):
        raise TypeError('Loaded type {} is not corrected, expected Graph.'.format(type(graph)))

# Chose a solver
solver_name = args.solver or cfg['solver'].lower()
if solver_name not in SOLVER.keys():
    raise KeyError('Solver of type {} not available. Choose between {}'.format(solver_name, SOLVER.keys()))
else:
    print('Solving with: {}'.format(solver_name))

solver = SOLVER[solver_name](graph, **cfg['solver_initialization'][solver_name])

# A couple of extra instructions for continuum solver
if solver_name == 'continuum':
    try:
        cfg['solver_execution']['continuum']['torch_optimizer'] =\
            OPTIMIZER[cfg['solver_execution']['continuum']['torch_optimizer']]
    except KeyError:
        cfg['solver_execution']['continuum']['torch_optimizer'] = torch.optim.Adam

    if cfg['solver_execution']['continuum']['dtype'] == 64:
        cfg['solver_execution']['continuum']['dtype'] = torch.float64

    if cfg['solver_execution']['continuum']['device'] == 'cpu':
        cfg['solver_execution']['continuum']['device'] = torch.device('cpu')
    elif cfg['solver_execution']['continuum']['device'] == 'cuda':
        cfg['solver_execution']['continuum']['device'] = torch.device('cpu')
    else:
        print('Setting device to standard device')
        cfg['solver_execution']['continuum']['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Execute solution
start = time.time()
solution = solver.solve(**cfg['solver_execution'][solver_name])
print('Solution found in {} seconds'.format(time.time()-start))

# Evaluate solution
print('Solution length {}'.format(evaluate_solution(solution)))

# Plot solution
solution.render()
plt.show()

# Pickle solution
t = time.localtime()
timestamp = time.strftime('%m-%d-%Y_%H%M', t)
pickle.dump(solution, open("./data/solution_locations{}_edges{}_traffic{}_time{}.p".format(
    graph.num_vertices, len(graph.get_edges(get_all=True)), args.traffic, timestamp), "wb"))
