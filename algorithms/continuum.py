from objects.graph import Graph
import numpy as np
import torch
import time
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display, clear_output


def column_normalize(array):
    norm = np.sum(array, axis=0)
    array[:, np.nonzero(norm)] = array[:, np.nonzero(norm)] / norm[np.nonzero(norm)]
    return array


def entropy(array):
    where = np.nonzero(array)
    return -np.sum(array[where] * np.log(array[where]))


def static_weights(iterations):
    return np.array([[1./3., 1./3., 1./3.]] * iterations  )


class PathRepresentationModel(nn.Module):
    """
    Class to define a model for the optimization of the TSP problem in a
    Path representation formalism
    :param worse_edge_factor:float, how much the use of non-existing links should be penalized
    :param sigma:float, for the softmax in the evaluation of the path cost, variance of the gaussian
    :param device:torch.device, usually gpu or cpu. Default cuda (gpu)
    :param dtype: datatype for torch default torch.float32
    """

    def __init__(self, graph, worse_edge_factor=3, sigma=0.05,
                 device=None, dtype=None):

        # For device compatibility
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
            print('Setting device {}'.format(self.device))
        if isinstance(dtype, torch.dtype):
            self.dtype = dtype
        else:
            self.dtype = torch.float32
            print('Setting dtype to {}'.format(self.dtype))
        # To use as a model (instance of nn.Module)
        super().__init__()

        # About the graph to ssolve
        if not isinstance(graph, Graph):
            raise ValueError('Graph type expected. {} given.'.format(type(graph)))
        self.graph = graph
        epsilon = 1e-6
        self.propagation_matrix = graph.adjacency_matrix / (np.sum(graph.adjacency_matrix, axis=0)+epsilon)
        # For the pytorch instance
        self.sigma = sigma
        self.modified_weight_tensor = self._get_modified_weight_tensor(worse_edge_factor)
        self.complete_tensor = None

        self.variable_tensor = nn.Parameter(self.initialize_variable_tensor())
        self.summary = None

        self.maxent = torch.tensor(np.log(self.graph.num_vertices), device=self.device, dtype=self.dtype)

    def forward(self):
        return self.get_working_tensor(self.variable_tensor)

    def initialize_variable_tensor(self, start=None):
        """
        Initialization of the Variable tensor.
        It is initialized with the sqrt of the initial path matrix.
        This is to work with the square of the tensor and avoid negative entries.
        TODO maybe abs is better?
        :param start: from which location to start
        :return: torch.tensor, with requires_grad=True.
        """
        path_matrix = np.sqrt(self.initialize_path_matrix(start=start))
        return Variable(torch.tensor(path_matrix, device=self.device, requires_grad=True, dtype=self.dtype))

    def reset_variable_tensor(self):
        self.variable_tensor = nn.Parameter(self.initialize_variable_tensor())
        return True

    def get_working_tensor(self, variable_tensor):
        """
        Square of the variable tensor, to guarantee that entries are >=0.
        :param variable_tensor: torch.tensor, with requires_grad=True
        :return: torch.tensor
        """
        return variable_tensor ** 2

    def _get_modified_weight_tensor(self, worse_edge_factor):
        worse_edge = np.max(self.graph.weight_matrix)
        modified_weight_matrix = self.graph.weight_matrix
        modified_weight_matrix[np.where(self.graph.weight_matrix == 0)] = worse_edge_factor * worse_edge
        return torch.tensor(modified_weight_matrix, device=self.device, dtype=self.dtype)

    def tensor_average_path_cost(self, path_tensor, epsilon=1e-6):
        """
        Path cost associated to a state, represented as path tensor.
        Evaluate the average Path cost, under the following assumptions:
        - The probability of using a link is proportional to the occupations of the source and destination
        - The probability of using any of the links is 1 (normalization)
        - The probability of using a link is independent on the weight of the link
        - Using a non-existing link is associated to a cost > cost of the cost of the most expensive link
        :param path_tensor: torch.tensor, representing the present state and requires_grad=True
        :return: expected weight of the path considered, according only to route length
        """
        ew = torch.tensor(0, device=self.device, dtype=self.dtype)
        for i in torch.arange(-1, path_tensor.shape[1] - 1):
            from_column = path_tensor[:, i]
            to_column = path_tensor[:, i + 1]
            link_probabilities = torch.ger(from_column, to_column)
            link_probabilities = link_probabilities / (torch.sum(link_probabilities)+epsilon)
            ew += torch.sum(link_probabilities * self.modified_weight_tensor)
        return ew

    def tensor_extra_cost(self, path_tensor, epsilon=0.):
        """
        There cost function has a problem in the fact that often some
        'expensive' link is used.
        In other words: here we use probabilities, but when we want to go back
        to the discrete situation almost all the paths do not matter (we do not consider
        a quantum version of tsp).
        To include this in the calculation we could force the cost of the path that goes
        from the argmax of one column to the argmax of the next.
        But argmax is not differentiable! So we cannot use autograd.
        max is differentiable instead, and this is what we will use.
        """
        # return self.tensor_average_path_cost(path_tensor)
        softmax = lambda t: torch.exp(-(t - torch.max(t)) ** 2. / self.sigma)
        ew = torch.tensor(0, device=self.device, dtype=self.dtype)
        for i in torch.arange(-1, path_tensor.shape[1] - 1):
            from_column = softmax(path_tensor[:, i])
            to_column = softmax(path_tensor[:, i + 1])
            link_probabilities = torch.ger(from_column, to_column)
            link_probabilities = link_probabilities / (torch.sum(link_probabilities) + epsilon)
            ew += torch.sum(link_probabilities * self.modified_weight_tensor)
        return ew

    def tensor_entropy(self, path_tensor, epsilon=1e-6):
        """
        Relative (normalized to number of locations) entropy associated to a path tensor
        :param path_tensor: torch.tensor representing the state
        :param epsilon:float, small parameter to avoid log(0) [good old physics memories]
        :return: torch.tensor, scalar containing the relative entropy of the path tensor
        """
        epsilon = epsilon * torch.ones(path_tensor.shape, device=self.device, dtype=self.dtype)
        total_entropy = -torch.sum(torch.log(path_tensor + epsilon) * path_tensor)
        return total_entropy / path_tensor.shape[0]

    def tensor_violations(self, path_tensor, dim=1):
        """
        Each city must be visited once: each row must sum to one
        If not, there is a price to pay
        :param path_tensor: torch.tensor representing the state
        :param dim: either 1 or 0 1 for rows, 0 for columns
        :return: \sum_i |1-\sum_j P_{ij}|, with P the path matrix (for rows, for columns invert ij)
        """
        if dim not in [0, 1]:
            raise ValueError('dim must be 0 or 1. {} given'.format(dim))
        return torch.sum(torch.abs(torch.sum(path_tensor, dim=dim)
                                   - torch.ones(path_tensor.shape[0], device=self.device, dtype=self.dtype)))

    def loss(self, col_violations_cost=1., row_violations_cost=1.,
             length_cost=1., entropy_cost=1., append=True):
        """
        Loss function for pytorch tensor.
        Also appends entropy, violation and path cost to the summary list
        :param col_violations_cost: float, cost of columns not summing to one
        :param row_violations_cost: float, cost of rows not summing to one
        :param length_cost: float, cost of having a longer route
        :param entropy_cost: float, cost of having fractional entries instead of ~0 or ~1
        :param append: bool, if the history of the 'training' should be recorded
        :return: float, cost associated to the matrix
        """
        if self.summary is None:
            self.summary = torch.tensor([], dtype=self.dtype, device=self.device)
        path_tensor = self.forward()
        row_evaluated_violations = self.tensor_violations(path_tensor, dim=1)
        column_evaluated_violations = self.tensor_violations(path_tensor, dim=0)
        path_evaluated_cost = self.tensor_average_path_cost(path_tensor)
        path_evaluated_extra_cost = self.tensor_extra_cost(path_tensor)
        entropy_evaluated_cost = self.tensor_entropy(path_tensor)
        lo = col_violations_cost * column_evaluated_violations
        lo += row_violations_cost * row_evaluated_violations
        lo += length_cost * (entropy_evaluated_cost/self.maxent * path_evaluated_cost
                             + (1.-entropy_evaluated_cost/self.maxent) * path_evaluated_extra_cost)
        lo += entropy_cost * entropy_evaluated_cost
        if append:
            self.summary = torch.cat((self.summary, torch.tensor([
                row_evaluated_violations, column_evaluated_violations,
                path_evaluated_cost, path_evaluated_extra_cost,
                entropy_evaluated_cost], dtype=self.dtype, device=self.device)))
        return lo

    ################################################
    #         Functions working with numpy         #
    ################################################

    def initialize_path_matrix(self, start=None):
        """
        Initialize the path matrix representing a "solution" of the TSP to be optimized

        :param start: string/int, key of the start location, or corresponding id
        :return: a path n*n matrix, n is the number of locations, obtained by applying recursively the operation
            P_{j,i+1} = \sum M_{j,k} P_{k,i}
            where P_{k,0} is \delta_{k, start} and M_{j,k} is the column normalized adjacency matrix
        """
        initialization_vector = np.zeros(self.graph.num_vertices)
        if isinstance(start, str):
            initialization_vector[self.graph.vert_ind[start]] = 1
        elif isinstance(start, int):
            initialization_vector[start] = 1
        elif start is None:
            initialization_vector[0] = 1
        else:
            raise ValueError('Start either from a string or from an integer. {} given'.format(type(start)))
        path_matrix = self._forward_initialization(initialization_vector)
        return path_matrix
        path_matrix = self._backward_initialization(path_matrix)
        return path_matrix

    def _forward_initialization(self, initialization_vector):
        path_matrix = np.zeros((len(initialization_vector), len(initialization_vector)))
        path_matrix[:, 0] = initialization_vector
        for i in range(1, path_matrix.shape[1]):
            path_matrix[:, i] = np.dot(self.propagation_matrix, path_matrix[:, i - 1])
        return path_matrix

    def _backward_initialization(self, path_matrix, epsilon=1e-6):
        """
        Breaking the simmetry for the first city also implies that this is the last city.
        As a consequence, we can impose that only cities from which one can reach the initial city are allowed in
        the column N-1. We can then propagate, this information backwards.
        :param path_matrix:
        :return: updated path matrix
        """
        forward = self.propagation_matrix
        initial_location = np.where(path_matrix[:, 0] == 1)[0][0]
        # Initialize the reconstructed matrix
        reconstructed_matrix = np.zeros(path_matrix.shape)
        reconstructed_matrix[initial_location, 0] = 1

        # Find locations allowed for the last visited location
        adj = self.graph.adjacency_matrix
        allowed_last_visited = np.nonzero(adj[initial_location, :])
        # Initialize the last location
        reconstructed_matrix[allowed_last_visited, -1] = path_matrix[allowed_last_visited, -1]
        reconstructed_matrix[:, -1] = reconstructed_matrix[:, -1] / (np.sum(reconstructed_matrix[:, -1]) + epsilon)

        for i in np.arange(-2, -self.graph.num_vertices, -1):
            backward = np.transpose(forward) * path_matrix[:, i]
            backward = column_normalize(backward)
            reconstructed_matrix[:, i] = np.dot(backward, reconstructed_matrix[:, i + 1])
            reconstructed_matrix[:, i] = reconstructed_matrix[:, i] / (np.sum(reconstructed_matrix[:, i]) + epsilon)
        return reconstructed_matrix


class Continuum:
    """
    Class to approach the TSP using a generalization in the continuum
    A state is represented as a n*n array whose entries M_{ij}
    is (loosely) the probability of visiting location i at step j.

    Associated to each state there is a cost, accounting for violations
    (e.g.: city visited twice, entropy of the continuous representation...)
    and for the expectation value of the path length.

    The cost is minimized using PyTorch, which instantiates a model from the class
    PathRepresentationModel, where all the numbers are kept track of.

    Arguments:
    ----------
    graph: Graph, the graph (custom class!) associated to the problem
    """

    # TODO add docstrings
    # TODO implement a routine that does the training internally
    # TODO implement visualization
    def __init__(self, graph):
        if not isinstance(graph, Graph):
            raise ValueError('The class must be initialized with a Graph. {} given'.format(type(graph)))
        self.graph = graph

        # To keep track of the progress
        self.summary = None
        self.snapshots = None

        # to avoid instantiating again the model
        self.model = None

        self.scheduled_params = None
        self.complete_tensor = None

    def get_model(self, device, dtype, worse_edge_factor=3., sigma=0.05):
        return PathRepresentationModel(self.graph, worse_edge_factor=worse_edge_factor, sigma=sigma,
                                       device=device, dtype=dtype)

    def alternating_weights(self, iterations, residual=0.1):
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

    def path_length_first(self, iterations, temperature=.1, residual=0.1):
        """
        First: try to reduce the path cost, then reduce the violations, then kill the entropy
        :param iterations: int, total number of iterations
        :param temperature:float, how sharp should the change be
        :param residual:float, constant part of the schedule, to avoid huge increase in the other aspects
        :return:np.array, iterations * 3, each column gives the weight of one component @ one iteration step
        """
        f = lambda x, location: 1./(1+np.exp((float(x/iterations) - location)/temperature))
        step_location = 1. / 3.
        sum_violations = np.array([residual + (1. - f(i, step_location)) *
                                   f(i, 2 * step_location)
                                   for i in range(iterations)])
        entropy_violations = np.array([residual + (1. - f(i, 2* step_location))
                                       for i in range(iterations)])
        path_cost = np.array([residual + f(i, step_location) for i in range(iterations)])
        schedule = np.column_stack((sum_violations, entropy_violations, path_cost))
        return (schedule.transpose() / np.sum(schedule, axis=1)).transpose()

    def scheduled_solve(self, col_violations_cost=1., row_violations_cost=1.,
                        length_cost=1., entropy_cost=1., snapshots=10,
                        device=torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu')),
                        dtype=torch.float32,
                        torch_optimizer=torch.optim.Adam, lr=0.005,
                        iterations=50000, worse_edge_factor=3., sigma=0.05,
                        schedule=static_weights,
                        from_fresh=False, append=True,
                        **kwargs):
        """
        'solve' the TSP instance associated to the graph by minimizing the cost of the representation tensor
        :param col_violations_cost:float, multiplicative parameter for the loss function
        :param row_violations_cost:float, multiplicative parameter for the loss function
        :param length_cost:float, multiplicative parameter for the loss function
        :param entropy_cost:float,multiplicative parameter for the loss function
        :param snapshots:int, how many snapshots of the state to take
        :param device: torch.device, specify if 'cuda' or 'cpu'. Default: 'cuda', if available.
        :param dtype: torch.dtype, default torch.float32.
        :param torch_optimizer: optimizer from torch
        :param lr: float, learning rate
        :param iterations: int, number of iterations
        :param worse_edge_factor: float, how much to penalize the 'use' of nonexisting links
        :param sigma: float, variance of the gaussian of softmax, used in evaluate_extra_cost
        :param schedule: np.array or callable, define the evolution of the weights in the training
        :param from_fresh: bool, continue training or use a brand new tensor
        :param append: bool, if True the loss will be appended in a tensor
        :param kwargs: parameters to pass to the torch optimizer
        :return: the solution graph
        """

        if callable(schedule):
            scheduled_params = schedule(iterations)
        elif isinstance(schedule, list):
            scheduled_params = np.array(schedule)
        elif isinstance(schedule, np.ndarray):
            scheduled_params = schedule
        else:
            raise TypeError('schedule must be callable or list or np array. {} given.'.format(type(schedule)))

        scheduled_params = torch.tensor(scheduled_params, dtype=dtype, device=device)
        if scheduled_params.shape != torch.Size([iterations, 3]):
            raise ValueError('Wrong schedule shape. Given {}. Expected {}'.format(self.scheduled_params.shape,
                                                                                  torch.Size([iterations, 3])))
        if (self.model is None) or (self.model.device != device) or (self.model.dtype != dtype):
            print('Instantiate a new model')
            self.model = self.get_model(worse_edge_factor=worse_edge_factor, sigma=sigma,
                                        device=device, dtype=dtype)

        if from_fresh:
            self.model.reset_variable_tensor()

        self.snapshots = self.snapshots or []
        optimizer = torch_optimizer(self.model.parameters(), lr=lr, **kwargs)
        every = max(iterations // snapshots, 1)
        for i in torch.arange(iterations, device=device):
            optimizer.zero_grad()
            loss = self.model.loss(col_violations_cost=col_violations_cost * scheduled_params[i, 0],
                                   row_violations_cost=row_violations_cost * scheduled_params[i, 0],
                                   length_cost=length_cost * scheduled_params[i, 2],
                                   entropy_cost=entropy_cost * scheduled_params[i, 1], append=append)
            if i % every == 0:
                # Syncronization step! You might not want it...
                self.snapshots.append(self.model().data)
            loss.backward()
            optimizer.step()
        self.complete_tensor = self.model()
        self.summary = self.model.summary
        # if self.model.summary.data.shape[0] > 0:
        #     self.summary = pd.DataFrame(np.array(self.model.summary.cpu().data),
        #                                 columns=['rows', 'cols', 'path_smooth', 'path_sharp', 'entropy'])
        return self.complete_tensor

    def solve(self, col_violations_cost=1., row_violations_cost=1.,
              length_cost=1., entropy_cost=1., snapshots=10,
              device=torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu')),
              dtype=torch.float32,
              torch_optimizer=torch.optim.Adam, lr=0.005,
              iterations=50000, worse_edge_factor=3., sigma=0.05,
              from_fresh=False, append=True,
              **kwargs):
        """
        'solve' the TSP instance associated to the graph by minimizing the cost of the representation tensor
        :param col_violations_cost:float, multiplicative parameter for the loss function
        :param row_violations_cost:float, multiplicative parameter for the loss function
        :param length_cost:float, multiplicative parameter for the loss function
        :param entropy_cost:float,multiplicative parameter for the loss function
        :param snapshots:int, how many snapshots of the state to take
        :param device: torch.device, specify if 'cuda' or 'cpu'. Default: 'cuda', if available.
        :param dtype: torch.dtype, default torch.float32.
        :param torch_optimizer: optimizer from torch
        :param lr: float, learning rate
        :param iterations: int, number of iterations
        :param worse_edge_factor: float, how much to penalize the 'use' of nonexisting links
        :param sigma: float, variance of the gaussian of softmax, used in evaluate_extra_cost
        :param from_fresh: bool, continue training or use a brand new tensor
        :param append: bool, if True the loss will be appended in a tensor
        :param kwargs: parameters to pass to the torch optimizer
        :return: the solution graph
        """

        if (self.model is None) or (self.model.device != device) or (self.model.dtype != dtype):
            print('Instantiate a new model')
            self.model = self.get_model(worse_edge_factor=worse_edge_factor, sigma=sigma,
                                        device=device, dtype=dtype)

        if from_fresh:
            self.model.reset_variable_tensor()

        self.snapshots = self.snapshots or []
        optimizer = torch_optimizer(self.model.parameters(), lr=lr, **kwargs)
        every = max(iterations // snapshots, 1)
        for i in range(iterations):
            optimizer.zero_grad()
            loss = self.model.loss(col_violations_cost=col_violations_cost,
                                   row_violations_cost=row_violations_cost,
                                   length_cost=length_cost,
                                   entropy_cost=entropy_cost, append=append)
            if i % every == 0:
                # Syncronization step! You might not want it...
                self.snapshots.append(self.model().data)
            loss.backward()
            optimizer.step()
        self.complete_tensor = self.model()
        self.summary = self.model.summary
        # if self.model.summary.data.shape[0] > 0:
        #     self.summary = pd.DataFrame(np.array(self.model.summary.cpu().data),
        #                                 columns=['rows', 'cols', 'path_smooth', 'path_sharp', 'entropy'])
        return self.complete_tensor

    def _get_used_locations(self, complete_tensor):
        # This function is not going to be pretty: I have no clue how to write it
        # In principle here some choice would have to be made!
        arr = np.array(complete_tensor.data)
        orders = np.zeros(arr.shape, dtype=int)
        for i in range(arr.shape[1]):
            orders[:, i] = np.argsort(arr[:, i])[::-1]

        used_locations = [self.graph.ind_vert[orders[0, 0]]]
        for i in range(1, orders.shape[1]):
            departure = used_locations[-1]
            for j in range(orders.shape[0]):
                arrival = self.graph.ind_vert[orders[j, i]]
                c1 = (departure, arrival) in self.graph.get_edges(get_all=True).keys()
                c2 = arrival not in used_locations
                if c1 and c2:
                    used_locations.append(arrival)
                    break
        if len(used_locations) != self.graph.num_vertices:
            raise ValueError('Did not get the correct order')
        return used_locations

    def get_solution_graph(self, complete_tensor):
        """
        Build the solution graph from the complete tensor.

        The following aspects must be considered:
        - The loss compromises between errors and path cost;
        - Until one does not penalize using a non-existing link these might be used;
        - Sometimes a location can appear in multiple locations, this is to be avoided;
        :param complete_tensor:
        :return: solution graph
        """
        used_locations = self._get_used_locations(complete_tensor)
        # order = [np.argmax(arr[:,i]) for i in range(arr.shape[1])]
        # order = [self.graph.ind_vert[i] for i in order]
        try:
            return self.graph.build_graph_solution(used_locations)
        except KeyError:
            return used_locations

    def show_evolution(self, cmap='Blues', persist=False, sleep=.5):
        arr = np.array(self.complete_tensor.data)
        order = np.array([np.argmax(arr[:, i]) for i in range(arr.shape[1])])

        # order = np.array([self.graph.vert_ind[v] for v in self._get_used_locations(self.complete_tensor)], dtype=int)
        for state in self.snapshots:
            if persist is False:
                clear_output(wait=True)
            sns.heatmap(state[order], cmap=cmap)
            display(plt.show())
            time.sleep(sleep)
        return

    def plot_progress(self, key='loss', xlim=None, ylim=None):
        """
        Simple util to print the progress after self.solve() has been executed
        :param key: which components of the loss should be printed
        :param xlim: either None or a list of dimension two
        :param ylim: either None or a list of dimension two
        """
        if len(self.summary) == 0:
            print('No data to plot.. yet')
            return
        if key not in self.summary[0].keys():
            y = [sum([val for k, val in i.items()]) for i in self.summary]
        else:
            y = [i[key] for i in self.summary]
        plt.plot(y, '-o')
        plt.xlim(xlim)
        plt.ylim(ylim)
        return True
