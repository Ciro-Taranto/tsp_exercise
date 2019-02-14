from problems import map_generator
from objects.graph import Graph
from problems.abstract_problems import Problem, Node
import numpy as np


class Environment(object):
    """
    Abstract class for the environment.
    Still to understand the relation with Problem
    and Node classes
    """

    def actions(self, state):
        """
        Return a list of actions executable in this state.
        Override this function
        """

        # Should invoke corresponding function of Problem or of Node
        raise NotImplementedError

    def result(self, state, action):
        """
        Return the result of a given action
        Override this function.
        """

        # Should invoke corresponing function of Problem or of Node
        return self.problem_instance


class VeihcleEnvironment(Environment):
    """
    Class defining the environment. 
    This should be transformed into an abstract class,
    which in turns should invoke abstract Problem and Node
    to serve the actions to the agent.
    """

    def __init__(self, number_of_points, number_of_deposits):
        """
        Initialize the instance of the map
        Args:
            number_of_points (int): number of collection points
            number_of_deposits(int): number fo deposits
        """
        self.problem_instance = map_generator.garbage_collection_points_layout(
            number_of_points,
            number_of_deposits)
        self.load_per_point = self.problem_instance[1]
        return None
