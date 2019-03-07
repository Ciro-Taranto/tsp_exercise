from objects.graph import Graph
import numpy as np
import torch
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output


def column_normalize(array):
    norm = np.sum(array, axis=0)
    array[:, np.nonzero(norm)] = array[:, np.nonzero(norm)]/norm[np.nonzero(norm)]
    return array


def entropy(array):
    where = np.nonzero(array)
    return -np.sum(array[where]*np.log(array[where]))


class Continuum:
    """
    Class to approach the TSP using a generalization in the continuum
    A state is represented as a n*n array whose entries M_{ij}
    is (loosely) the probability of visiting location i at step j.

    Associated to each state there is a cost, accounting for violations
    (e.g.: city visited twice, entropy of the continuous representation...)
    and for the expectation value of the path length.

    The cost is minimized using PyTorch

    Arguments:
    ----------
    graph: Graph, the graph (custom class!) associated to the problem
    worse_edge_factor: float, in evaluating the expectation value of the route length,
            associate worse_edge_factor * longest_link to the non-existing connections
    """
    # TODO add docstrings
    # TODO implement a routine that does the training internally
    # TODO implement visualization
    def __init__(self, graph, worse_edge_factor=3.):
        if not isinstance(graph,Graph):
            raise ValueError('The class must be initialized with a Graph. {} given'.format(type(graph)))
        self.graph = graph
        self.propagation_matrix = self.column_normalized_adj_matrix()

        # For the pytorch instance
        weight_matrix = self.graph.weight_matrix
        worse_edge = np.max(weight_matrix)
        modified_weight_matrix = weight_matrix
        modified_weight_matrix[np.where(weight_matrix==0)] = worse_edge_factor * worse_edge
        self.modified_weight_tensor = torch.tensor(modified_weight_matrix)
        self.modified_weight_matrix= modified_weight_matrix
        self.variable_tensor = None
        self.complete_tensor = None

        # To keep track of the progress
        self.summary = []
        self.snapshots = []

    def column_normalized_adj_matrix(self):
        adj = self.graph.adjacency_matrix
        return adj/np.sum(adj,axis=0)

    def _forward_initialization(self,initialization_vector):
        path_matrix = np.zeros((len(initialization_vector), len(initialization_vector)))
        path_matrix[:,0] = initialization_vector
        for i in range(1,path_matrix.shape[1]):
            path_matrix[:,i] = np.dot(self.propagation_matrix, path_matrix[:,i-1])
        return path_matrix

    def _backward_initialization(self,path_matrix):
        """
        Breaking the simmetry for the first city also implies that this is the last city.
        As a consequence, we can impose that only cities from which one can reach the initial city are allowed in
        the column N-1. We can then propagate, this information backwards.
        :param path_matrix:
        :return: updated path matrix
        """
        forward = self.propagation_matrix
        initial_location = np.where(path_matrix[:,0] == 1)[0][0]
        # Initialize the reconstructed matrix
        reconstructed_matrix = np.zeros(path_matrix.shape)
        reconstructed_matrix[initial_location, 0] = 1

        # Find locations allowed for the last visited location
        adj = self.graph.adjacency_matrix
        allowed_last_visited = np.nonzero(adj[initial_location, :])
        # Initialize the last location
        reconstructed_matrix[allowed_last_visited, -1] = path_matrix[allowed_last_visited, -1]
        reconstructed_matrix[:, -1] = reconstructed_matrix[:, -1]/np.sum(reconstructed_matrix[:, -1])

        for i in np.arange(-2, -self.graph.num_vertices, -1):
            backward = np.transpose(forward) * path_matrix[:, i]
            backward = column_normalize(backward)
            reconstructed_matrix[:, i] = np.dot(backward, reconstructed_matrix[:, i+1])
            reconstructed_matrix[:, i] = reconstructed_matrix[:, i]/np.sum(reconstructed_matrix[:, i])
        return reconstructed_matrix

    def initialize_path_matrix(self,start=None):
        """
        Initialize the path matrix representing a "solution" of the TSP to be optimized

        :param start: string/int, key of the start location, or corresponding id
        :return: a path n*n matrix, n is the number of locations, obtained by applying recursively the operation
            P_{j,i+1} = \sum M_{j,k} P_{k,i}
            where P_{k,0} is \delta_{k, start} and M_{j,k} is the column normalized adjacency matrix
        """
        initialization_vector = np.zeros(self.graph.num_vertices)
        if isinstance(start,str):
            initialization_vector[self.graph.vert_ind[start]] = 1
        elif isinstance(start,int):
            initialization_vector[start] = 1
        elif start is None:
            initialization_vector[0] = 1
        else:
            raise ValueError('Start either from a string or from an integer. {} given'.format(type(start)))
        path_matrix = self._forward_initialization(initialization_vector)
        return path_matrix
        path_matrix = self._backward_initialization(path_matrix)
        return path_matrix

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
        return Variable(torch.tensor(path_matrix), requires_grad=True)

    def get_working_tensor(self, variable_tensor):
        """
        Square of the variable tensor, to guarantee that entries are >=0.
        :param variable_tensor: torch.tensor, with requires_grad=True
        :return: torch.tensor
        """
        return variable_tensor**2

    def tensor_average_path_cost(self, path_tensor):
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
        ew = torch.tensor(0, dtype=torch.float64)
        for i in torch.arange(-1, path_tensor.shape[1]-1):
            from_column = path_tensor[:, i]
            to_column = path_tensor[:, i+1]
            link_probabilities = torch.ger(from_column, to_column)
            link_probabilities = link_probabilities/torch.sum(link_probabilities)
            ew += torch.sum(link_probabilities * self.modified_weight_tensor)
        return ew

    def evaluate_average_path_cost(self, path_matrix):
        """
        Path cost associated to a state, represented as path matrix.
        Evaluate the average Path cost, under the following assumptions:
        - The probability of using a link is proportional to the occupations of the source and destination
        - The probability of using any of the links is 1 (normalization)
        - The probability of using a link is independent on the weight of the link
        - Using a non-existing link is associated to a cost > cost of the cost of the most expensive link
        :param path_matrix: np.arrat, representing the present state and requires_grad=True
        :return: expected weight of the path considered, according only to route length
        """
        expected_weight = 0.
        for i in range(path_matrix.shape[1]-1):
            from_column = path_matrix[:, i]
            to_column = path_matrix[:, i+1]
            link_probabilities = np.outer(from_column, to_column)
            norm = np.sum(link_probabilities)
            if norm == 0:
                raise ValueError('Cannot have two columns completely disconnected')
            link_probabilities = link_probabilities/norm
            expected_weight += np.sum(link_probabilities * self.modified_weight_matrix)
        return expected_weight

    def tensor_extra_cost(self, path_tensor):
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
        softmax = lambda t: torch.exp(-(t-torch.max(t))**2./0.05)
        ew = torch.tensor(0, dtype=torch.float64)
        for i in torch.arange(-1, path_tensor.shape[1]-1):
            from_column = softmax(path_tensor[:, i])
            to_column = softmax(path_tensor[:, i+1 ])
            link_probabilities = torch.ger(from_column, to_column)
            link_probabilities = link_probabilities / torch.sum(link_probabilities)
            ew += torch.sum(link_probabilities * self.modified_weight_tensor)
        return ew

    def evaluate_relative_entropy(self, path_matrix):
        """
        Evaluate relative entropy of the path matrix
        :param path_matrix:
        :return: relative entropy
        """
        return entropy(path_matrix)/path_matrix.shape[0]

    def tensor_entropy(self, path_tensor):
        """
        Relative (normalized to number of locations) entropy associated to a path tensor
        :param path_tensor: torch.tensor representing the state
        :return: torch.tensor, scalar containing the relative entropy of the path tensor
        """
        # Good old memories of physics...
        epsilon = 1e-6*torch.ones(path_tensor.shape, dtype=torch.float64)
        total_entropy = -torch.sum( torch.log(path_tensor + epsilon) * path_tensor)
        return total_entropy/path_tensor.shape[0]

    def evaluate_violations(self,path_matrix, axis=1):
        """
        Each city must be visited once: each row must sum to one
        If not, there is a price to pay
        :param path_matrix: path matrix to evaluate violations
        :param axis: either 1 or 0 1 for rows, 0 for columns
        :return: \sum_i |1-\sum_j P_{ij}|, with P the path matrix (for rows, for columns invert ij)
        """
        if axis not in [0, 1]:
            raise ValueError('Axis in evaluate violations must be 0 or 1. {} give'.format(axis))
        return np.sum(np.abs(np.sum(path_matrix,axis=axis)-np.ones(path_matrix.shape[0])))

    def tensor_violations(self, path_tensor, dim=1):
        """
        Each city must be visited once: each row must sum to one
        If not, there is a price to pay
        :param path_tensor: torch.tensor representing the state
        :param dim: either 1 or 0 1 for rows, 0 for columns
        :return: \sum_i |1-\sum_j P_{ij}|, with P the path matrix (for rows, for columns invert ij)
        """
        if dim not in [0,1]:
            raise ValueError('dim must be 0 or 1. {} given'.format(dim))
        return torch.sum(torch.abs(torch.sum(path_tensor, dim=dim)
                                   - torch.ones(path_tensor.shape[0], dtype=torch.float64)))

    def cost(self, path_matrix,col_violations_cost=1., row_violations_cost=1.,
             length_cost=1., entropy_cost=1.):
        """
        Evaluate the cost using numpy. Use this to benchmark Pytorch loss implementation
        :param path_matrix: np.array, matrix to evaluate;
        :param col_violations_cost: float, cost of columns not summing to one
        :param row_violations_cost: float, cost of rows not summing to one
        :param length_cost: float, cost of having a longer route
        :param entropy_cost: float, cost of having fractional entries instead of ~0 or ~1
        :return: float, cost associated to the matrix
        """
        coefficients = np.array([col_violations_cost, row_violations_cost, length_cost, entropy_cost])
        costs = np.array([self.evaluate_violations(path_matrix, axis=0),
                          self.evaluate_violations(path_matrix),
                          self.evaluate_average_path_cost(path_matrix),
                          self.evaluate_relative_entropy(path_matrix)])
        return coefficients.dot(costs)

    def loss(self, path_tensor, col_violations_cost=1., row_violations_cost=1.,
             length_cost=1., entropy_cost=1.):
        """
        Loss function for pytorch tensor.
        Also appends entropy, violation and path cost to the summary list
        :param path_tensor: torch.tensor, tensor to evaluate. path_tensor.requires_grad must be True
        :param col_violations_cost: float, cost of columns not summing to one
        :param row_violations_cost: float, cost of rows not summing to one
        :param length_cost: float, cost of having a longer route
        :param entropy_cost: float, cost of having fractional entries instead of ~0 or ~1
        :return: float, cost associated to the matrix
        """
        row_evaluated_violations = self.tensor_violations(path_tensor, dim=1)
        column_evaluated_violations = self.tensor_violations(path_tensor, dim=0)
        path_evaluated_cost = self.tensor_average_path_cost(path_tensor)
        path_evaluated_extra_cost = self.tensor_extra_cost(path_tensor)
        entropy_evaluated_cost = self.tensor_entropy(path_tensor)
        lo = torch.tensor(col_violations_cost, dtype=torch.float64) * column_evaluated_violations
        lo += torch.tensor(row_violations_cost, dtype=torch.float64) * row_evaluated_violations
        lo += torch.tensor(length_cost, dtype=torch.float64) * (path_evaluated_cost + path_evaluated_extra_cost) * .5
        lo += torch.tensor(entropy_cost, dtype=torch.float64) * entropy_evaluated_cost
        self.summary.append({'row_violations': row_evaluated_violations,
                             'col_violations': column_evaluated_violations,
                             'path_cost': path_evaluated_cost,
                             'extra_cost': path_evaluated_extra_cost,
                             'entropy': entropy_evaluated_cost})
        return lo

    def solve(self, col_violations_cost=1., row_violations_cost=1.,
              length_cost=1., entropy_cost=1., snapshots=20,
              torch_optimizer=torch.optim.Adam, lr=0.005,
              iterations=50000,
              from_fresh=False,
              **kwargs):
        """
        'solve' the TSP instance associated to the graph by minimizing the cost of the representation tensor
        :param col_violations_cost:float, multiplicative parameter for the loss function
        :param row_violations_cost:float, multiplicative parameter for the loss function
        :param length_cost:float, multiplicative parameter for the loss function
        :param entropy_cost:float,multiplicative parameter for the loss function
        :param snapshots:int, how many snapshots of the state to take
        :param torch_optimizer: optimizer from torch
        :param lr: float, learning rate
        :param iterations: int, number of iterations
        :param kwargs: parameters to pass to the torch optimizer
        :return: the solution graph
        """
        if self.variable_tensor is None or from_fresh:
            self.variable_tensor = self.initialize_variable_tensor()

        optimizer = torch_optimizer([self.variable_tensor], lr=lr, **kwargs)
        every = iterations//snapshots
        for i in range(iterations):
            optimizer.zero_grad()
            complete_tensor = self.get_working_tensor(self.variable_tensor)
            loss = self.loss(complete_tensor, col_violations_cost=col_violations_cost,
                             row_violations_cost=row_violations_cost, length_cost=length_cost,
                             entropy_cost=entropy_cost)
            if i % every == 0:
                self.snapshots.append(np.array(complete_tensor.data))
            loss.backward()
            optimizer.step()
        self.complete_tensor = complete_tensor
        return complete_tensor

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
        order = np.array([np.argmax(arr[:,i]) for i in range(arr.shape[1])])

        # order = np.array([self.graph.vert_ind[v] for v in self._get_used_locations(self.complete_tensor)], dtype=int)
        for state in self.snapshots:
            if persist is False:
                clear_output(wait=True)
            sns.heatmap(state[order],  cmap=cmap)
            display(plt.show())
            time.sleep(sleep)

        return
                            
    def plot_progress(self, key='loss',xlim=None, ylim=None):
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
        plt.plot(y,'-o')
        plt.xlim(xlim)
        plt.ylim(ylim)
        return True

