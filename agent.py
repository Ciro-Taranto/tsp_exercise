from environment import Environment
from algorithms.simulated_anneling import SimAnneal
from algorithms.genetic import Genetic
from algorithms.search import TravelingSalesman
from objects.solver import Solver


class agent():
    """
    Class to instantiate the agent.
    It should receive the problem from an environment 
    and have methods to solve it and return a solution 
    """

    def __init__(self):
        """
        Nothin to do for initialization
        """
        self.solver = Solver()

    def chose_solver(self, solver_class):
        """
        solver_class must be the class inheriting from Solver,
        e.g. TravelingSalesman, Genetic, or SimAnneal,
        """
        self.solver = solver_type

    def instantiate_solver(self, problem_instance):
        self.solver = self.solver(problem_instance)

    def receive_instance(self, problem_instance):
        self.problem_insance = problem_instance

    def provide_solution(self):
        return self.solver.solve()

    def respond_with_action(self, problem_state):
        raise NotImplementedError
