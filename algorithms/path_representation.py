from objects.graph import Graph
import numpy as np
import torch
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
from torch.nn import functional as nnf


def static_weights(iterations):
    return np.array([[1./3., 1./3., 1./3.]] * iterations)


def alternating_weights(iterations, residual=0.1):
    """
    Function that alternates the leading weight in the loss function.
    Order: 0 -> rows and cols violations; 1 -> entropy violations; 2 -> path length
    :param iterations:int, total number of iterations
    :param residual: float, how much the 'other components' should contribute to the loss
    :return: np.array, iterations * 3, where each column gives the weight of one component at each iteration step
    """
    return np.array([[1 if i % iterations == 0 else residual,
                      1 if i % iterations == 1 else residual,
                      1 if i % iterations == 2 else residual]
                     for i in range(iterations)])

class Continuum:
    """
    Class to approach the TSP using a generalization in the continuum
    A state is represented as a n*n array whose entries P_{ij}
    is (loosely) the probability of visiting location j at step i.

    Associated to each state there is a cost, accounting for violations
    (e.g.: city visited twice, entropy of the continuous representation...)
    and for the expectation value of the path length.

    The cost is minimized using PyTorch, which instantiates a model from the class
    PathRepresentationModel, where all the numbers are kept track of.

    Arguments:
    ----------
    graph: Graph, the graph (custom class!) associated to the problem
    device: string, the torch.device to use. Default: cuda if available
    """
    def __init__(self, graph, device=None, **kwargs):
        if not isinstance(graph, Graph):
            raise ValueError('The class must be initialized with a Graph. {} given'.format(type(graph)))
        self.graph = graph

        try:
            self.device = torch.device(device)
        except:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        # To keep track of the progress
        self.progress = torch.tensor([], device=self.device, dtype=self.dtype)
        self.snapshots = []

        # For the torch optimization
        self.variable_tensor = self.initialize_variable_tensor(self.device, self.dtype)
        self.working_tensor = nnf.softmax(self.variable_tensor, dim=1)
        self.complete_tensor = None
        self.weight_tensor = None

    def initialize_variable_tensor(self, device, dtype):
        propagation = torch.tensor(self.graph.adjacency_matrix, device=device, dtype=dtype)
        propagation = propagation / propagation.sum(dim=1, keepdim=True)
        path_matrix = torch.zeros_like(propagation)
        path_matrix[0, 0] = 1.
        for i in range(1, path_matrix.shape[0]):
            path_matrix[i, :] = path_matrix[i-1, :].matmul(propagation)
        return Variable(torch.log(path_matrix + 1e-7), requires_grad=True)

    def initialize_modified_weight_tensor(self, worse_edge_factor, device, dtype):
        weights = self.graph.weight_matrix
        worse_edge = np.max(weights)
        weights[np.where(weights == 0)] = worse_edge * worse_edge_factor
        return torch.tensor(weights, device=device, dtype=dtype)

    def tensor_path_cost(self):
        ew = self.working_tensor[:-1, :].mm(self.weight_tensor.mm(self.working_tensor[1:, :].t())).trace()
        # ew = torch.einsum('ti,ij,jt->t', self.working_tensor[:-1, :],
        #                   self.weight_tensor, self.working_tensor[1:, :].t()).sum() # slightly slower!
        return ew + self.working_tensor[-1:, :].matmul(self.weight_tensor.matmul(self.working_tensor[0, :]))[0]

    def tensor_entropy(self, epsilon=1e-7):
        return -torch.sum(torch.log(self.working_tensor + epsilon) * self.working_tensor)

    def tensor_violations(self):
        return torch.sum((self.working_tensor.sum(dim=0)-1) ** 2)

    def loss(self, s, memory=True):
        """
        Loss function for optimization.
        :param s: torch.tensor, [violations_cost, entropy_cost, path_cost]
        :param memory: bool, if True, will append to the a proper tensor
        :return: loss at the iteration
        """
        violations = self.tensor_violations()
        entropy = self.tensor_entropy()
        path_cost = self.tensor_path_cost()

        if memory:
            self.progress = torch.cat((self.progress, torch.tensor([[violations, entropy, path_cost]],
                                                                   device=self.device, dtype=self.dtype)))
        return s[0] * violations + s[1] * entropy + s[2] * path_cost

    def solve(self, violations_cost=1., entropy_cost=1.,
              path_cost=1., snapshots=10,
              torch_optimizer=torch.optim.Adam, lr=0.005,
              iterations=50000, worse_edge_factor=3.,
              schedule=static_weights,
              from_fresh=False, memory=True,
              optimizer_params={},
              ):
        """
        'solve' the TSP instance associated to the graph by minimizing the cost of the representation tensor
        :param violations_cost:float, multiplicative parameter for the loss function
        :param path_cost:float, multiplicative parameter for the loss function
        :param entropy_cost:float,multiplicative parameter for the loss function
        :param snapshots:int, how many snapshots of the state to take
        :param torch_optimizer: optimizer from torch
        :param lr: float, learning rate
        :param iterations: int, number of iterations
        :param worse_edge_factor: float, how much to penalize the 'use' of nonexisting links
        :param schedule: np.array or callable, define the evolution of the weights in the training
        :param from_fresh: bool, continue training or use a brand new tensor
        :param memory: bool, if True the loss will be appended in a tensor
        :param optimizer_params: parameters to pass to the torch optimizer
        :return: the solution graph
        """

        scheduled_params = schedule(iterations)
        scheduled_params = torch.tensor([violations_cost, entropy_cost, path_cost]
                                        * scheduled_params, dtype=self.dtype, device=self.device)
        assert scheduled_params.shape == torch.Size([iterations, 3])

        if from_fresh:
            self.variable_tensor = self.initialize_variable_tensor(self.device, self.dtype)
        self.weight_tensor = self.initialize_modified_weight_tensor(worse_edge_factor, self.device, self.dtype)

        print(f'Initial path cost= {self.tensor_path_cost()}')
        print(f'Expected path cost = {self.weight_tensor.mean() * self.graph.num_vertices}')
        print(f'Ratio={self.tensor_path_cost()/(self.weight_tensor.mean() * self.graph.num_vertices)}')
        print(f'Initial entropy cost={self.tensor_entropy()}')
        print(f'Initial violations cost={self.tensor_violations()}')

        self.snapshots = self.snapshots or []
        optimizer = torch_optimizer([self.variable_tensor], lr=lr, **optimizer_params)
        every = max(iterations // snapshots, 1)

        for i in torch.arange(iterations, device=self.device):
            optimizer.zero_grad()
            self.working_tensor = nnf.softmax(self.variable_tensor, dim=1)
            loss = self.loss(scheduled_params[i, :], memory=memory)
            if i % every == 0:
                # Syncronization step! You might not want it...
                self.snapshots.append(nnf.softmax(self.variable_tensor, dim=1))
            loss.backward()
            optimizer.step()

        print(f'Final path cost= {self.tensor_path_cost()}')
        print(f'Final entropy cost={self.tensor_entropy()}')
        print(f'Final violations cost={self.tensor_violations()}')

        self.complete_tensor = nnf.softmax(self.variable_tensor, dim=1)
        return self.complete_tensor

    def path_length_first(self, iterations, temperature=.1, residual=0.1):
        """
        First: try to reduce the path cost, then reduce the violations, then kill the entropy
        :param iterations: int, total number of iterations
        :param temperature:float, how sharp should the change be
        :param residual:float, constant part of the schedule, to avoid huge increase in the other aspects
        :return:np.array, iterations * 3, each column gives the weight of one component @ one iteration step
        """
        f = lambda x, location: 1. / (1 + np.exp((float(x / iterations) - location) / temperature))
        step_location = 1. / 3.
        sum_violations = np.array([residual + (1. - f(i, step_location)) *
                                   f(i, 2 * step_location)
                                   for i in range(iterations)])
        entropy_violations = np.array([residual + (1. - f(i, 2 * step_location))
                                       for i in range(iterations)])
        path_cost = np.array([residual + f(i, step_location) for i in range(iterations)])
        schedule = np.column_stack((sum_violations, entropy_violations, path_cost))
        return (schedule.transpose() / np.sum(schedule, axis=1)).transpose()

        ################################################
        #             Visualization functions          #
        ################################################

    def get_solution_graph(self):
        """
        Build the solution graph from the complete tensor.

        The following aspects must be considered:
        - The loss compromises between errors and path cost;
        - Until one does not penalize using a non-existing link these might be used;
        - Sometimes a location can appear in multiple locations, this is to be avoided;
        :return: solution graph
        """
        arr = np.array(self.complete_tensor.cpu().data)
        order = [np.argmax(arr[i,:]) for i in range(arr.shape[0])]
        order = [self.graph.ind_vert[i] for i in order]
        try:
            return self.graph.build_graph_solution(order)
        except KeyError:
            return order

    def show_evolution(self, cmap='Blues', persist=False, sleep=.5, disordered=False):
        arr = self.complete_tensor.cpu().detach().numpy()
        order = np.array([np.argmax(arr[:, i]) for i in range(arr.shape[0])])
        if disordered:
            order = np.arange(arr.shape[1])

        for state in self.snapshots:
            if persist is False:
                clear_output(wait=True)
            sns.heatmap(state.cpu().data.numpy()[order], cmap=cmap)
            display(plt.show())
            time.sleep(sleep)
        return

    def plot_progress(self, keys=['loss'], xlim=None, ylim=None):
        """
        Simple util to print the progress after self.solve() has been executed
        :param keys: which components of the loss should be printed
        :param xlim: either None or a list of dimension two
        :param ylim: either None or a list of dimension two
        """
        summary = self.progress.cpu().data.numpy()
        columns = {'violations':0, 'entropy':1, 'path':2}

        if len(self.progress) == 0:
            print('No data to plot... yet')
            return
        for key in keys:
            try:
                plt.plot(summary[:, columns[key]], '-o', label=key)
            except KeyError:
                print('Key {} not found.'.format(key))
                print('Available keys: {}'.format(columns.keys()))
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        return True



