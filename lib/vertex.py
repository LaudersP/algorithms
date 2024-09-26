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