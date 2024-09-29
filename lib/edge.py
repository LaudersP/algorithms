from lib.vertex import Vertex

class Edge:
    def __init__(self, start_vertex, end_vertex, weight=None, directed=False):
        """Construct a new edge"""
        self._start = start_vertex
        self._end = end_vertex
        self._weight = weight
        self._directed = directed
    
    @property
    def start_vertex(self):
        """
        Get the start vertex of the edge
        :return: The start vertex object
        """
        return self._start
    
    @property
    def end_vertex(self):
        """
        Get the end vertex of the edge
        :return: The end vertex object
        """
        return self._end
    
    @property
    def weight(self):
        """
        Get the weight of the edge
        :return: The edge weight
        """
        return self._weight
    
    @weight.setter
    def weight(self, value):
        """
        Set the weight of the edge
        :param value: The new edge weight
        """
        self._weight = value
    
    @property
    def directed(self):
        """
        Check if the edge is directed
        :return: True if the edge is directed, False otherwise
        """
        return self._directed
    
    def __eq__(self, other):
        """
        Check if this edge is equal to another edge
        :param other: The other edge to compare
        :return: True if edges are equal, False otherwise
        """
        if not isinstance(other, Edge):
            return NotImplemented
        return (self.start_vertex == other.start_vertex and
                self.end_vertex == other.end_vertex and
                self.weight == other.weight and
                self.directed == other.directed)
    
    def __hash__(self):
        """
        Compute the hash of the edge
        :return: The hash value
        """
        return hash((self.start_vertex, self.end_vertex, self.weight, self.directed))
    
    def __repr__(self) -> str:
        """
        Return the string representation of the edge
        :return: A string representing the edge
        """
        direction = "->" if self.directed else "--"
        if self.weight is None:
            return f"Edge({self.start_vertex.id}{direction}{self.end_vertex.id})"
        else:
            return f"Edge({self.start_vertex.id}{direction}{self.end_vertex.id}) [{self.weight}]"
