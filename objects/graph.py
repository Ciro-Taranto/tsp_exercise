"""
Adapted from:
https://www.bogotobogo.com/python/python_graph_data_structures.php
"""
from observer.human import render
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

    def get_connections(self, sort=False):
        if not sort:
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


class Graph():
    def __init__(self, **kwargs):
        self.vert_dict = {}
        self.num_vertices = 0
        self.adding_order = []
        if "locations" in kwargs.keys() and "weights" in kwargs.keys():
            locations = kwargs['locations']
            weights = kwargs['weights']
            self.import_graph(locations, weights)

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

    def import_graph(self, locations, weights):
        """
        Import a graph given the location of the vertexes
        and the edges weight. 
        Args: 
            -location(dict): {key: loc}
            -weights(dict): {(from,to):weight}
        """
        assert type(locations) == dict
        assert type(weights) == dict
        for i, l in locations.items():
            self.add_vertex(i, l)
        for key, weight in weights.items():
            self.add_edge(key[0], key[1], weight)
        return True

    def build_graph_solution(self, order):
        g = Graph()
        start = order[0]
        l = self.get_vertex(start).get_location()
        g.add_vertex(start, l)
        for i in range(1, len(order)):
            # vAdd vertex
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

    def render(self):
        render(self)
        return True
