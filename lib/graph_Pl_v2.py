from lib.vertex import Vertex
from lib.edge import Edge

class GraphV2(object):
    """This class handles creation and operations of graph related code"""
    
    def __init__ (self, graph_input=None) -> None:
        """
        Create a new instance of the Graph class
        :param graph_input: A adjacency list or matrix for a graph
        """
        self._adj_list = None
        self._adj_matrix = None
        
        # Check if the input is a adjacency list
        if isinstance(graph_input, dict):
            # Iterate through to thoroughly check the list
            for key, values in graph_input.items():
                # Check the key
                if not isinstance(key, Vertex):
                    raise Exception("Adjacency list key must be of type 'Vertex'")
                
                # Check the edge values
                for value in values:
                    # Check the value
                    if not isinstance(value, Edge):
                        raise Exception("Adjacency list value must be of type 'Edge'")
                    
            # Set the adjacency list
            self._adj_list = graph_input
            
            # Convert to adjacency matrix
            self.__adj_list_to_matrix()

        # Check if the input is a adjacency matrix      
        elif isinstance(graph_input, list):
            # Iterate through to thoroughly check the matrix
            for row in graph_input:
                # Check that the row is a list
                if not isinstance(row, list):
                    raise Exception("Matrix row must be of type 'list'")
                
                # Iterate through column values
                for value in row:
                    # Check that the value is edge
                    if value is not None and not isinstance(value, Edge):
                        raise Exception("Matrix column value must be of type 'Edge'")
                    
            # Set the adjacency matrix
            self._adj_matrix = graph_input

        # No graph was supplied
        self._adj_list = {}
        self.adj_matrix = []
                    
    @property
    def adj_list(self) -> dict:
        """
        Getter for the adjacency list
        :return: Current adjacency list
        """
        return self._adj_list
                    
    @property
    def adj_matrix(self) -> list:
        """
        Getter for the adjacency matrix
        :return: Current adjacency matrix
        """
        return self._adj_matrix
    
    @property
    def order(self) -> int:
        """
        Getter for the graph order
        :return: The order of the graph
        """
        # Check if there is a adjacency list
        if self.adj_list:
            return len(list(self.adj_list.keys()))
        # Check if there is a matrix
        elif self.adj_matrix:
            return len(self.adj_matrix)
        
    @property
    def size(self) -> int:
        """
        Getter for the graph size
        :return: The size of the graph
        """
        visited = []
        
        # Check if there is a list
        if self.adj_list:
            # Iterate through the list
            for _, edges in self.adj_list.items():
                # Iterate through the edges
                for edge in edges:
                    # Check if the edge has been counted
                    if visited.count(edge) == 0:
                        visited.append(edge)
            
        # Check if there is a matrix
        elif self.adj_matrix:
            # Iterate through the rows
            for row in self.adj_matrix:
                # Iterate through the column values
                for value in row:
                    # Check for an edge and see if it has been counted for
                    if value is not None and visited.count(value) == 0:
                        visited.append(value)
                        
        # Return the size value
        return len(visited)
    
    def __adj_list_to_matrix(self) -> None:
        """
        Convert the adjacency matrix to an adjacency list
        """
        # Check for an adjacency list
        if not self.adj_list:
            raise Exception("No adjacency list to convert from!")
        
        # Initialize a fresh matrix
        self._adj_matrix = []
        
        # Initialize an empty matrix
        for _ in range(self.order):
            column_data = [None for _ in range(self.order)]
            self._adj_matrix.append(column_data)
        
        # Get a list of vertices
        vertices = list(self._adj_list.keys())
        
        # Set the matrix values
        for row, vertex in enumerate(vertices):
            # Iterate through the edges
            for edge in self.adj_list[vertex]:
                # Find the column based on ending vertex
                for column, value in enumerate(vertices):
                    # Check if the edge has no direction (start end order does not matter if true)
                    if edge.directed == False:
                        # Check if start is vertex and end is value
                        if edge.start_vertex == vertex and edge.end_vertex == value:
                            self._adj_matrix[row][column] = edge
                        # Check if end is vertex and start is value
                        elif edge.end_vertex == vertex and edge.start_vertex == value:
                            self._adj_matrix[row][column] = edge
                            
                    # Start end order does matter
                    else:
                        if edge.end_vertex == value:
                            self._adj_matrix[row][column] = edge
                
    def __adj_matrix_to_list(self) -> None:
        """
        Convert the adjacency list to an adjacency matrix
        """
        pass
    
    def get_vertex_weight(self, vertex):
        """
        Gets the weight of a vertex
        :param vertex: The desired vertex
        :return: The weight of the vertex
        """
        # Check that the vertex is valid
        if not isinstance(vertex, Vertex):
            raise Exception("`vertex` must be of type 'Vertex'!")
        
        return vertex.weight
    
    def set_vertex_weight(self, vertex, weight):
        """
        Sets the weight of a vertex
        :param vertex: The desired vertex
        :param weight: The desired weight
        """
        # Check that the vertex is valid
        if not isinstance(vertex, Vertex):
            raise Exception("`vertex` must be of type 'Vertex'!")
        
        vertex.weight = weight
    
    def get_edge_weight(self, edge):
        """
        Gets the weight of a edge
        :param vertex: The desired edge
        :return: The weight of the edge
        """
        # Check that the edge is valid
        if not isinstance(edge, Edge):
            raise Exception("`edge` must be of type 'Edge'!")
        
        return Edge.weight
    
    def set_edge_weight(self, edge, weight):
        """
        Sets the weight of a edge
        :param *: The desired edge
        :param weight: The desired weight
        """
        # Check that the edge is valid
        if not isinstance(edge, Vertex):
            raise Exception("`edge` must be of type 'Edge'!")
        
        edge.weight = weight
    
    def add_vertex(self, vertex):
        """
        Adds a new vertex to the graph
        :param vertex: An instance of the vertex class
        """
        # Check that the vertex is valid
        if not isinstance(vertex, Vertex):
            raise Exception("`vertex` must be of type 'Vertex'!")
        
        # Initialize adj_list entry
        self._adj_list[vertex] = []

        # Update matrix
        self.__adj_list_to_matrix()
    
    def del_vertex(self, vertex):
        """
        Deletes a vertex from the graph and all connecting edges
        :param vertex: The desired vertex
        """
        # Check that the vertex is valid
        if not isinstance(vertex, Vertex):
            raise Exception("`vertex` must be of type 'Vertex'!")

        # Go to the adj_list and get its edges
        edges = self.adj_list[vertex]

        # Iterate and delete all edges
        for edge in edges:
            del edge

        # Delete vertex from adj_list
        del vertex

        # Update matrix
        self.__adj_list_to_matrix()
    
    def add_edge(self, edge):
        """
        Adds a new edge to the graph
        :param edge: An instance of the edge class
        """
        # Check that the edge is valid
        if not isinstance(edge, Edge):
            raise Exception("`edge` must be of type 'Edge'!")
        
        # Check if the edge is directed
        if edge.directed:
            # Insert into the adj_list for the first node
            self._adj_list[edge.start_vertex] += edge
        else:
            # Insert into the adj_list for both nodes
            self._adj_list[edge.start_vertex] += edge
            self._adj_list[edge.end_vertex] += edge
        
        # Update the matrix
        self.__adj_list_to_matrix()

    def del_edge(self, edge):
        """
        Deletes a vertex from the graph
        :param edge: Starting vertex
        """
        # Check that the edge is valid
        if not isinstance(edge, Edge):
            raise Exception("`edge` must be of type 'Edge'!")
        
        # Check if the edge is directed
        if edge.directed:
            # Remove the edge from the starting vertex edge list
            pass
        else:
            # Remove the edge from both vertex edge lists
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
        