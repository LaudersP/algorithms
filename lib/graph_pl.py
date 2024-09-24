import lib.vector as vector
import lib.matrix as matrix

class Graph(object):
    """This class contains graph operations"""
            
    @property
    def adj_list(self) -> vector:
        """
        Getter for the adjacency list
        :return: Adjacency list
        """
        return self._adj_list
    
    @adj_list.setter
    def adj_list(self, adj_list) -> None:
        """
        Setter for the adjacency list
        :param adj_list: The adjacency list to set
        """
        self._adj_list = adj_list
        
    @property
    def adj_matrix(self) -> matrix.Matrix:
        """
        Getter for the adjacency matrix
        :return: Adjacency matrix
        """
        return self._adj_matrix
    
    @adj_matrix.setter
    def adj_matrix(self, adj_matrix):
        """
        Setter for the adjacency matrix
        :param adj_matrix: The adjacency matrix to set
        """
        self._adj_matrix = adj_matrix
        
    @property
    def order(self) -> int:
        """
        Getter for the graph order
        :return: The order of the graph
        """
        return self._order
    
    @order.setter
    def order(self, value) -> None:
        """
        Setter for the graph order
        :param value: The desired order of the graph
        """
        self._order = value
        
    @property
    def size(self) -> int:
        """
        Getter for the graph size
        :return: The size of the graph
        """
        return self._size
    
    @size.setter
    def size(self, value) -> None:
        """
        Setter for the graph size
        :param value: The desired size of the graph
        """
        self._size = value
        
    def __init__(self, graph_input=None) -> None:
        """
        The constructor for the Graph class
        :param *args: A adjacency list or matrix
        """
        self._adj_list = None
        self._adj_matrix = None
        self._order = 0
        self._size = 0
        
        # Check if a adjacency list was supplied
        if isinstance(graph_input, dict): 
            # Set the adjacency list
            self._adj_list = graph_input
            
            # Get the order
            self._get_order() 
            
            # Convert the list to matrix
            self._adj_list_to_matrix()
            
            # Get the size of the graph
            self._get_size()
        # Check if a adjacency matrix was supplied
        elif isinstance(graph_input, matrix.Matrix):
            # Set the adjacency matrix
            self._adj_matrix = graph_input
            
            # Get the order of the graph
            self._get_order() 
            
            # Convert the matrix to a list
            self._adj_matrix_to_list()

            # Get the size of the graph
            self._get_size()
            
    def _adj_list_to_matrix(self) -> None:
        """
        Convert the adjacency matrix to an adjacency list
        """
        # Check for an adjacency list
        if not self.adj_list:
            raise Exception("No adjacency list to convert from!")
        
        # Create a list to hold the edges
        matrix_blueprint = []
        
        # Initialize an empty matrix
        for _ in range(self._order):
            column_data = [0 for _ in range(self._order)]
            row_data = vector.Vector(*column_data)
            matrix_blueprint.append(row_data)
            
        # Create the matrix
        self._adj_matrix = matrix.Matrix(*matrix_blueprint)
        
        # Convert the adjacency list to matrix
        for key, data in self._adj_list.items():
            for element in range(self._order - 1):
                self._adj_matrix.__setitem__([key - 1, int(data.__getitem__(element)) - 1], 1)
    
    def _adj_matrix_to_list(self) -> None:
        """
        Convert the adjacency list to an adjacency matrix
        """
        # Check for an adjacency matrix
        if not self.adj_matrix:
            raise Exception("No adjacency matrix to convert from!")
        
        # Create a dict to hold the edges
        adj_list_blueprint = {}
        
        # Break down the matrix
        for row in range(self._order):
            edge_data = []
            for column in range(self._order):
                # Check if there is a edge at the current matrix position
                if self._adj_matrix.__getitem__([row, column]):
                    edge_data.append(column + 1)
                    
            # Add the edge to the list
            adj_list_blueprint[row + 1] = vector.Vector(*edge_data)
            
        # Create the list
        self._adj_list = adj_list_blueprint
    
    def _get_order(self) -> None:
        """
        Calculates the order of a graph
        """
        # Check if there is a list
        if self.adj_list:
            self._order = len(list(self.adj_list.keys()))
        # Check if there is a matrix
        elif self.adj_matrix:
            self._order = self._adj_matrix.num_rows
        else:
            raise Exception("Unable to get the order due to no adjacency list or matrix being found!")
    
    def _get_size(self) -> None:
        """
        Calculates the size of a graph
        """
        # Check if there is a list
        if self.adj_list:
            for _, edges in self._adj_list.items():
                self._size += len(edges)
                
            self._size = self._size / 2
        else:
            raise Exception("Unable to get the size due to no adjacency list being found!")
    
    def get_vertex_weight(self):
        """
        Gets the weight of a vertex
        :param *: The desired vertex
        :return: The weight of the vertex
        """
        pass
    
    def set_vertex_weight(self):
        """
        Sets the weight of a vertex
        :param *: The desired vertex
        :param weight: The desired weight
        """
        pass
    
    def get_edge_weight(self):
        """
        Gets the weight of a edge
        :param *: The desired edge
        :return: The weight of the edge
        """
        pass
    
    def set_edge_weight(self):
        """
        Sets the weight of a edge
        :param *: The desired edge
        :param weight: The desired weight
        """
        pass
    
    def add_vertex(self):
        """
        Adds a new vertex to the graph
        :param *: The new vertex
        """
        pass
    
    def del_vertex(self):
        """
        Deletes a vertex from the graph and all connecting edges
        :param *: The desired vertex
        """
        pass
    
    def add_edge(self):
        """
        Adds a new edge to the graph
        :param start: Starting vertex
        :param end: Ending vertex
        """
        pass
    
    def del_edge(self):
        """
        Deletes a vertex from the graph
        :param start: Starting vertex
        :param end: Ending vertex
        """
        pass
    
    def is_directed(self) -> bool:
        """
        Checks if the graph is directed
        :return: If the graph is directed
        """
        pass
    
    def is_connected(self) -> bool:
        """
        Checks if the graph is connected
        :return: If the graph is connected
        """
        pass
    
    def is_unilaterally_connected(self) -> bool:
        pass
    
    def is_weakly_connected(self) -> bool:
        pass
    
    def is_tree(self) -> bool:
        """
        Checks if the graph is a tree
        :return: If the graph is a tree
        """
        pass
    
    def components(self) -> int:
        """
        Gets the number of components in the graph
        :return: The number of components
        """
        pass
    
    def girth(self) -> int:
        """
        Gets the length of the shortest cycle in the graph
        :return: The length of the shortest cycle
        """
        pass
    
    def circumference(self) -> int:
        """
        Gets the length of the longest cycle in the graph
        :return: The length of the longest cycle
        """
        pass