"""
Adapted from: 
https://www.bogotobogo.com/python/python_graph_data_structures.php
"""


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

    def get_connections(self):
        return self.adjacent.keys()

    def get_location(self):
        return self.location

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]


class Graph:
    def __init__(self, **kwargs):
        self.vert_dict = {}
        self.num_vertices = 0
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

    def get_vertices(self):
        return self.vert_dict.keys()

    def get_edges(self):
        """
        Returns the edges arranged in a dictionary
        """
        edges = {}
        for vert in iter(self):
            vert_id = vert.get_id()
            for neighbor in vert.get_connections():
                neighbor_id = neighbor.get_id()
                if vert_id < neighbor_id:
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
