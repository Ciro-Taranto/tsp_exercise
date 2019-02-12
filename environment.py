import pylab as plt
from matplotlib import collections as mc
import map_generator
from graph import Graph
import numpy as np


class Problem():
    """
    Abstract class for a search problem."""

    def __init__(self, initial=None, goals=(), **additional_keywords):
        """Provide an initial state and optional goal states.
        A subclass can have additional keyword arguments."""
        self.initial = initial  # The initial state of the problem.
        self.goals = goals      # A collection of possible goal states.
        self.__dict__.update(**additional_keywords)

    def actions(self, state):
        "Return a list of actions executable in this state."
        raise NotImplementedError  # Override this!

    def result(self, state, action):
        "The state that results from executing this action in this state."
        raise NotImplementedError  # Override this!

    def is_goal(self, state):
        "True if the state is a goal."
        return state in self.goals  # Optionally override this!

    def step_cost(self, state, action, result=None):
        "The cost of taking this action from this state."
        return 1  # Override this if actions have different costs


class Environment():
    """
    Class defining the environment
    Inspire the class to the env of openAI gym  
    """

    def __init__(self, number_of_points, number_of_deposits):
        """
        Initialize the instance of the map
        Args:
            number_of_points (int): number of collection points
            number_of_deposits(int): number fo deposits
        Returns:
            -a list of 2d coordinates for the collection points.
            -a list of 2d coordinates for the deposits.
            -a list with the load to be collected from each stop.
            -a   (number_of_points+number_of_deposits)
                x(number_of_points+number_of_deposits)
                array with the distance between the points 
        """

        self.problem_instance = map_generator.garbage_collection_points_layout(
            number_of_points,
            number_of_deposits)
        self.load_per_point = self.problem_instance[1]

        return None

    def actions(self, state):
        "Return a list of actions executable in this state."
        raise NotImplementedError

    def result(self, state, action):
        """
        Given an action (or a query) of the agent
        returns the state of the env. 
        For the moment it returns the full instance of the problem. 
        With partial observability the env will return the proper load. 
        """
        return self.problem_instance

    def render(self, g):
        """
        function to plot the graph g 
        """
        vertices = g.get_vertices()
        locations = []
        deposits = []
        #s = []
        for vert in vertices:
            location = g.get_vertex(vert).get_location()
            if vert.startswith('l'):
                locations.append(location)
                # s.append(self.load_per_point[vert])
            elif vert.startswith('d'):
                deposits.append(location)
        locations = np.array(locations)
        deposits = np.array(deposits)
        fig, ax = plt.subplots()
        ax.scatter(
            locations[:, 0], locations[:, 1],
            color='blue')  # , s=5*s)
        ax.scatter(
            deposits[:, 0], deposits[:, 1],
            color='red'
        )
        edges = g.get_edges()
        edges = [[g.get_vertex(k[0]).get_location(),
                  g.get_vertex(k[1]).get_location()]
                 for k in edges.keys()]
        lines = mc.LineCollection(edges, linewidths=1)
        ax.add_collection(lines)
        return edges
