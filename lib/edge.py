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
        Get the start of an edge
        :return: The start vertex object
        """
        return self._start
    
    @property
    def end_vertex(self):
        """
        Get the end of an edge
        :return: The end vertex object
        """
        return self._end
    
    @property
    def weight(self):
        """
        Get the weight of an edge
        :return: The edge weight
        """
        return self._weight
    
    @weight.setter
    def weight(self, value):
        """
        Set the end of an edge
        :param value: The edge weight
        """
        self._weight = value
    
    @property
    def directed(self):
        """
        Set the end of an edge
        :param vertex: The vertex object
        """
        return self._directed
    
    def __repr__(self) -> str:
        if self.weight is None:
            return f"Edge({self.start_vertex.id}->{self.end_vertex.id})"
        else:
            return f"Edge({self.start_vertex.id}->{self.end_vertex.id}) [{self.weight}]"