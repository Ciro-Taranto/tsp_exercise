from observer.human import render
from collections import OrderedDict
import numpy as np
import operator


class Vertex():
    def __init__(self, node, location=(0, 0)):
        self.id = node
        self.location = location
        self.adjacent = {}

    def __str__(self):
        return "{} adjacent:".format({self.id})+",".join(
            [str(x.id) for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self, sort=False, retrieve_id=False):
        if not sort:
            if retrieve_id:
                return [k.get_id() for k in self.adjacent.keys()]
            else:
                return self.adjacent.keys()
        else:
            return sorted(self.adjacent, key=self.adjacent.get)

    def get_nearest(self):
        return min(self.adjacent, key=self.adjacent.get)

    def get_location(self):
        return self.location

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def delete_edge(self, neighbor):
        if neighbor in self.adjacent.keys():
            del self.adjacent[neighbor]
            return True
        else:
            return False


class Graph:
    def __init__(self, **kwargs):
        self.vert_dict = OrderedDict()
        self.num_vertices = 0
        self.adding_order = []
        # Obsolete, the different constructor should be given as class method
        if "locations" in kwargs.keys() and "weights" in kwargs.keys():
            locations = kwargs['locations']
            weights = kwargs['weights']
            self.import_graph(locations, weights)
        self.vert_ind = {vert_id: i for i, vert_id in enumerate(self.vert_dict.keys())}
        self.ind_vert = {v: k for k, v in self.vert_ind.items()}

    def import_graph(self, locations, weights):
        """
        Import a graph given the location of the vertexes
        and the edges weight.
        Args:
            -location(dict): {key: l}
            -weights(dict): {(from,to):weight}
        """
        assert isinstance(locations,dict) and isinstance(weights,dict)
        for i, l in locations.items():
            self.add_vertex(i, l)
        for key, weight in weights.items():
            self.add_edge(key[0], key[1], weight)
        return True

    @property
    def adjacency_matrix(self):
        if hasattr(self,'adj'):
            return self.adj
        else:
            self.adj = self._compute_adjacency_matrix()
            return self.adj

    @property
    def weight_matrix(self):
        if hasattr(self,'weight'):
            return self.weight
        else:
            self.weight = self._compute_adjacency_matrix(weight=True)
            return self.weight

    @classmethod
    def from_locations_eges(cls, locations, weights):
        """
        Construct from locations and weights
        """
        if not (isinstance(locations,dict) and isinstance(weights,dict)):
            raise ValueError('Locations and weights must be dictionaries')
        graph = cls()
        for i, l in locations.items():
            graph.add_vertex(i, l)
        for key, weight in weights.items():
            graph.add_edge(key[0], key[1], weight)
        graph.vert_ind = {vert_id: i for i, vert_id in enumerate(graph.vert_dict.keys())}
        return graph

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node, location):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node, location)
        self.vert_dict[node] = new_vertex
        self.adding_order.append(node)
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, weight=0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], weight)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], weight)
        return True

    def get_vertices(self):
        return self.vert_dict.keys()

    def get_vertices_connections(self, sorted=False):

        all_connections = {key: {n.get_id(): n.get_weight(val)
                                 for n in val.get_connections()}
                           for key, val in self.vert_dict.items()}
        if sorted:
            all_connections = {k: sorted(v.items(), key=operator.itemgetter(1))
                               for k, v in all_connections.items()}
        return all_connections

    def delete_vertex(self, vert_id):
        """
        To delete a vertex, first delete all the edges
        in which the vertex is included.
        This implementation requires symmetric graphs.
        """
        vert = self.get_vertex(vert_id)
        if vert is None:
            return False
        else:
            for neighbor in vert.get_connections():
                neighbor.delete_edge(vert)
            del self.vert_dict[vert_id]
            self.num_vertices -= 1
            return True

    def get_edge_weight(self, n1, n2):
        v1 = self.get_vertex(n1)
        v2 = self.get_vertex(n2)
        weight = v1.get_weight(v2)
        return weight

    def get_edges(self, get_all=False):
        """
        Returns the edges arranged in a dictionary
        Note: Each edge is counted just once.
        """
        edges = {}
        for vert in iter(self):
            vert_id = vert.get_id()
            for neighbor in vert.get_connections():
                neighbor_id = neighbor.get_id()
                if vert_id < neighbor_id or get_all:
                    edges[(vert_id, neighbor_id)
                          ] = vert.get_weight(neighbor)
        return edges

    def _compute_adjacency_matrix(self, weight=False):
        """
        Gives the adjacency matrix associated to the graph.
        For this scope the weights are not used.
        The graph is supposed to be directed.

        We want to use it as acting on a column vector.
        v_i^{t+1} = \sum_j M_{ij} v_j^t
        Meaning that if v_i^t are the location at step t, at step t+1
        I can go to v_i^{t+1}.

        To obtain this if (from,to) is in the list of edges it must be set M_{to,from}!=0
        Normalization and probabilites are left aside for the moment
        :return: np.array, adjacency matrix
        """
        print('Computing adj...')
        adjacency = np.zeros((len(self.vert_dict),len(self.vert_dict)))
        for edge, w in self.get_edges(get_all=True).items():
            adjacency[self.vert_ind[edge[1]], self.vert_ind[edge[0]]]= w if weight else 1
        return adjacency



    def build_graph_solution(self, order):
        g = Graph()
        start = order[0]
        l = self.get_vertex(start).get_location()
        g.add_vertex(start, l)
        for i in range(1, len(order)):
            # Add vertex
            n = order[i]
            loc = self.get_vertex(n).get_location()
            g.add_vertex(n, loc)
            # Add relative edges
            p = order[i-1]
            w = self.get_edge_weight(n, p)
            g.add_edge(n, p, w)
            g.add_edge(p, n, w)
        w = self.get_edge_weight(order[0], order[-1])
        g.add_edge(order[0], order[-1], w)
        g.add_edge(order[-1], order[0], w)
        return g

    def build_graph_solution_from_edges(self, solution_edges):
        all_edges = self.get_edges(get_all=True)
        solution_edges = {edge: all_edges[edge] for edge in solution_edges}
        locations = {v: self.get_vertex(v).get_location()
                     for v in self.get_vertices()}
        g = Graph(locations=locations, weights=solution_edges)
        return g

    def render(self):
        render(self)
        return True
