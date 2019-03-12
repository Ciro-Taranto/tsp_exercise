import argparse
import pickle
from objects.graph import Graph
from observer.human import render

parser = argparse.ArgumentParser(description='Compare several solutions of TSP, stored as pickle')

# Optional arguments
parser.add_argument('filenames',  default=str, nargs='+',
                    help='List the names of the pickled files for which you want to render')

args = parser.parse_args()
solutions = [pickle.load(open(i, 'rb')) for i in args.filenames]
render(*solutions)
