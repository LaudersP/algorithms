class Vertex:
    """Class used to handle vertex id and weight holding"""
    
    def __init__(self, id, weight=None):
        """Construct a new vertex"""
        self._id = id
        self._weight = weight
        
    @property
    def id(self):
        """
        Get the id of a vertex
        :return: The label ID
        """
        return self._id
    
    @property
    def weight(self):
        """
        Get the weight of a vertex
        :return: The weight value
        """
        return self._weight
    
    @weight.setter
    def weight(self, value):
        """
        Set the weight of a vertex
        :param value: The weight of the vertex
        """
        self._weight = value
        
    def __repr__(self) -> str:
        """
        Get a string representation of the edge
        :return: A representation as a sting
        """
        if self.weight is None:
            return f"Vertex({self.id})"
        else:
            return f"Vertex({self.id}) [{self.weight}]"

    def __lt__(self, other) -> bool:
        """
        < operator
        :param other: The vertex to compare against
        :return: True if this vertex's id is less than the `other` vertex's id
        """
        if not isinstance(other, Vertex):
            return NotImplemented
        return self.id < other.id
    
    def __eq__(self, other):
        """
        == operator
        :param other: The vertex to compare against
        :return: True if this vertex is equal to the `other` vertex
        """
        if not isinstance(other, Vertex):
            return NotImplemented
        return self.id == other.id and self.weight == other.weight
        
    def __hash__(self):
        """Allows Vertex to be used as a key in dictionaries and stored in sets."""
        return hash((self.id, self.weight))
