from lib.vertex import Vertex
from lib.edge import Edge
from collections import deque
import numpy as np
import math

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
            
            # Convert to adjacency list
            self.__adj_matrix_to_list()

        # No graph was supplied
        else:
            self._adj_list = {}
            self._adj_matrix = []
                    
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
                    if edge not in visited:
                        visited.append(edge)
            
        # Check if there is a matrix
        elif self.adj_matrix:
            # Iterate through the rows
            for row in self.adj_matrix:
                # Iterate through the column values
                for value in row:
                    # Check for an edge and see if it has been counted for
                    if value is not None and value not in visited:
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
        # Check for an adjacency matrix
        if not self.adj_matrix:
            raise Exception("No adjacency matrix to convert from!")
        
        # Initialize a fresh adj_list
        self._adj_list = {}
        
        # Iterate through the matrix and get all vertices
        for vertex in self.adj_matrix:
            for edge in vertex:
                # Check if the entry is not None
                if edge is not None:
                    # Add the vertices
                    if edge.start_vertex not in self.adj_list:
                        self._adj_list[edge.start_vertex] = []
                        
                    if edge.end_vertex not in self.adj_list:
                        self._adj_list[edge.end_vertex] = []
                        
        # Add in the edges
        for vertex in self.adj_matrix:
            for edge in vertex:
                if edge is not None:
                    # Check if the edge is directed
                    if edge.directed and edge not in self.adj_list[edge.start_vertex]:
                        # Add to only the start vertex
                        self._adj_list[edge.start_vertex].append(edge)
                    else:
                        # Add to both vertex
                        if edge not in self._adj_list[edge.start_vertex]:
                            self._adj_list[edge.start_vertex].append(edge)
                            
                        if edge not in self._adj_list[edge.end_vertex]:
                            self._adj_list[edge.end_vertex].append(edge)
                            
    def adj_matrix_to_2D_numpy_array(self) -> np.ndarray:
        """Convert the adjacency matrix to a 2D NumPy array with 1s and 0s.
        :return: A 2D NumPy array of the graph's adjacency matrix with binary values.
        """
        # Check if there is an adjacency matrix to convert
        if not self.adj_matrix and not self.adj_list:
            raise Exception("Adjacency matrix is not defined!")

        # Check if there is an adjacency list to firstly convert
        if not self.adj_matrix and self.adj_list:
            self.__adj_list_to_matrix()

        # Initialize an empty 2D list
        binary_matrix = []

        # Iterate through each row in the adjacency matrix
        for row in self.adj_matrix:
            # Initialize an empty list for the current row
            binary_row = []

            # Iterate through each column in the row
            for column in row:
                # Check if the column has an edge instance
                if column is not None:
                    binary_row.append(1)
                else:
                    binary_row.append(0)

            # Append the processed row to the binary matrix
            binary_matrix.append(binary_row)

        # Convert the binary matrix to a NumPy array
        array = np.array(binary_matrix)

        return array

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
        
        return edge.weight
    
    def set_edge_weight(self, edge, weight):
        """
        Sets the weight of a edge
        :param *: The desired edge
        :param weight: The desired weight
        """
        # Check that the edge is valid
        if not isinstance(edge, Edge):
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
        
        # Get the adj_list keys
        vertices = list(self._adj_list.keys())
        
        # Get the keys ids in order
        vertices_id = sorted([v.id for v in vertices])
            
        # Create a new adj_list
        new_adj_list = {}
        for vertex_id in vertices_id:
            for vertex in vertices:
                if vertex.id == vertex_id:
                    new_adj_list[vertex] = self._adj_list[vertex]
                    break
                
        # Update the adj_list
        self._adj_list = new_adj_list

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
        
        # Iterate through the dict
        for vertex_edges in self.adj_list:
            for edge in self.adj_list[vertex_edges]:
                # Check if the start or end vertex of a edge is linked with `vertex`
                if edge.start_vertex is vertex:
                    self.adj_list[vertex_edges].remove(edge)
                    
                if edge.end_vertex is vertex:
                    self.adj_list[vertex_edges].remove(edge)

        # Delete vertex from adj_list
        self._adj_list.pop(vertex)

        # Update matrix
        self.__adj_list_to_matrix()
        
    def add_edge(self, edge, allow_multiple_edges=False):
        """
        Adds a new edge to the graph
        :param edge: An instance of the edge class
        :param allow_multiple_edges: Optional parameter to allow loops or multiple edges between two vertices
        """
        # Check that the edge is valid
        if not isinstance(edge, Edge):
            raise Exception("`edge` must be of type 'Edge'!")
        
        # Check for loops if not allowed
        if not allow_multiple_edges and edge.start_vertex == edge.end_vertex:
            raise Exception("Loops are not allowed unless 'allow_multiple_edges' is True!")
        
        # Check if the edge is directed
        if edge.directed:
            # Check for multiple edges if not allowed
            if not allow_multiple_edges and edge in self._adj_list[edge.start_vertex]:
                raise Exception("Multiple edges between the same vertices are not allowed unless 'allow_multiple_edges' is True!")
            
            # Insert into the adj_list for the start vertex
            self._adj_list[edge.start_vertex].append(edge)
        else:
            # Check for multiple edges if not allowed
            if not allow_multiple_edges:
                # Check if the edge already exists in either vertex's adjacency list
                if edge in self._adj_list[edge.start_vertex] or edge in self._adj_list[edge.end_vertex]:
                    raise Exception("Multiple edges between the same vertices are not allowed unless 'allow_multiple_edges' is True!")
            
            # Check if the edge is a loop
            if edge.start_vertex != edge.end_vertex:
                self._adj_list[edge.start_vertex].append(edge)
                self._adj_list[edge.end_vertex].append(edge)
            else:
                self._adj_list[edge.start_vertex].append(edge)
        
        # Update the adjacency matrix
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
            self._adj_list[edge.start_vertex].remove(edge)
        else:
            self._adj_list[edge.start_vertex].remove(edge)
            self._adj_list[edge.end_vertex].remove(edge)
            
        # Update the matrix
        self.__adj_list_to_matrix()
    
    def is_directed(self) -> bool:
        """
        Checks if the graph is directed
        :return: If the graph is directed
        """
        # Loop through the matrix 
        for i in range(self.order):
            for j in range(i, self.order):
                # Compare the symmetry
                if self.adj_matrix[i][j] != self.adj_matrix[j][i]:
                    return True # Directed
         
        return False # Not directed
            
    def is_connected(self) -> bool:
        """
        Checks if the graph is connected.
        For undirected graphs, returns True if the graph is connected.
        For directed graphs, returns True if the graph is strongly connected.
        :return: True if the graph is connected (undirected) or strongly connected (directed), False otherwise.
        """
        # Check if the graph is directed
        if self.is_directed():
            return self.is_strongly_connected()
        else:
            # Get a list of vertices
            vertices = list(self.adj_list.keys())
            
            # Check that the graph has vertices
            if not vertices:
                return True
            
            # Get the traversal order
            traversal_order = self.bfs(vertices[0])
            
            # Check if all vertices were visited
            return len(traversal_order) == len(vertices)
    
    def is_unilaterally_connected(self) -> bool:
        """
        checks if there is a directed path between every pair of vertices in one direction
        :return: If the graph is unilaterally connected
        """
        # Get the vertices
        vertices = list(self.adj_list.keys())
        
        # Check for vertices
        if not vertices:
            return True
        
        # Create the results list
        vertices_bfs_results = {}
        for vertex in vertices:
            vertices_bfs_results[vertex] = self.bfs(vertex)
            
        # Check every pair of vertices (u, v)
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u = vertices[i]
                v = vertices[j]
                
                # Check if there is a path form u -> v or v -> u
                if v not in vertices_bfs_results[u] and u not in vertices_bfs_results[v]:
                    return False
                
        return True
    
    def is_strongly_connected(self) -> bool:
        """
        Checks if there is a directed path between every pair of vertices in both directions
        :return: If the graph is strongly connected
        """
        # Get the vertices
        vertices = list(self.adj_list.keys())
        
        # Check for vertices
        if not vertices:
            return True
        
        # Pick a start vertex
        start_vertex = vertices[0]
        
        # Perform a BFS search to see if all vertices to see are reachable
        original_reachable = self.bfs(start_vertex)
        if len(original_reachable) != len(vertices):
            return False
        
        # Transpose the graph
        transposed_graph = self.transpose()
        
        # Perform a BFS from start_vertex on the transposed graph
        transposed_reachable = transposed_graph.bfs(start_vertex)
        if len(transposed_reachable) != len(vertices):
            return False
            
        return True
    
    def is_weakly_connected(self) -> bool:
        """
        Checks if the underlying graph is connected if direction is ignored
        :return: If the graph is weakly connected
        """
        # Make a new graph
        underlying_graph = GraphV2()
        
        # Get the vertices
        vertices = list(self.adj_list.keys())
        
        # Add all of the vertices
        for vertex in vertices:
            underlying_graph.add_vertex(vertex)
            
        # Add all of the edges
        for vertex in vertices:
            for edge in self.adj_list[vertex]:
                underlying_graph.add_edge(Edge(edge.start_vertex, edge.end_vertex, directed=False))
                
        # Perform a BFS
        traversal_order = self.bfs(vertices[0])
        
        # Check if all vertices were visited
        return len(vertices) == len(traversal_order)
    
    def is_tree(self) -> bool:
        """
        Checks if the graph is a tree
        :return: If the graph is a tree
        """
        # Check to see if the size is appropriate for a tree
        if self.size != (self.order - 1):
            return False
        
        # Check if the graph is connected
        if not self.is_connected():
            return False
        
        return True
    
    def components(self) -> int:
        """
        Gets the number of components in the graph
        :return: The number of components
        """
        # Get the vertices
        vertices = list(self.adj_list.keys())
        
        # Check for vertices
        if not vertices:
            return 0
        
        # Initialize the visited list and component counter
        visited = set()
        components = 0
        
        # Traverse each vertex
        for vertex in vertices:
            # Check if the vertex has been visited
            if vertex not in visited:
                # Perform a BFS
                visited.update(self.bfs(vertex))
                        
                # Increment component count
                components += 1
                
        return components
    
    def get_cycle_length(self, start_vertex, compare_function=None) -> int:
        """
        Find the length of a cycle based on a starting vertex.
        :param start_vertex: Starting vertex of the cycle.
        :param compare_function: Function to compare cycle lengths (e.g., min or max).
        :return: The length of the cycle or None if no cycle exists.
        """
        if self.is_directed():
            # Setup DFS
            visited = set()
            rec_stack = {}
            cycle_lengths = []

            # DFS recursive helper function
            def dfs(current, depth):
                # Add the current node for tracking
                visited.add(current)
                rec_stack[current] = depth

                # Iterate through the edges for the current vertex
                for edge in self.adj_list[current]:
                    # Get the neighboring vertex
                    neighbor = edge.end_vertex

                    # Check if the neighbor vertex has been visited
                    if neighbor not in visited:
                        dfs(neighbor, depth + 1)
                    # Check if the neighbor vertex is in the recursion stack
                    elif neighbor in rec_stack:
                        # Cycle detected
                        cycle_len = depth - rec_stack[neighbor] + 1
                        cycle_lengths.append(cycle_len)

                # Remove the current vertex from the recursion stack
                rec_stack.pop(current)

            # Start the recursive DFS
            dfs(start_vertex, 0)

            # Determine what to return
            if cycle_lengths:
                # Determine if there is a compare function
                if compare_function:
                    return compare_function(cycle_lengths)
                else:
                    return cycle_lengths[0]
            else:
                return None
        else:
            # Setup for BFS
            visited = set()
            parent = {}
            depths = {}
            queue = deque()

            # Add the starting vertex
            visited.add(start_vertex)
            parent[start_vertex] = None
            depths[start_vertex] = 0
            queue.append(start_vertex)

            cycle_lengths = []

            # Perform the BFS
            while queue:
                # Get the next vertex
                current = queue.popleft()

                # Iterate through the current vertex's edges
                for edge in self.adj_list[current]:
                    # Determine the neighbor vertex
                    if edge.start_vertex == current:
                        neighbor = edge.end_vertex
                    else:
                        neighbor = edge.start_vertex

                    # Check if the neighbor vertex has been visited
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        depths[neighbor] = depths[current] + 1
                        queue.append(neighbor)
                    elif parent[current] != neighbor:
                        # Cycle detected
                        cycle_len = depths[current] + depths[neighbor] + 1
                        cycle_lengths.append(cycle_len)

            # Determine what to return
            if cycle_lengths:
                # Determine if there is a compare function
                if compare_function:
                    return compare_function(cycle_lengths)
                else:
                    return cycle_lengths[0]
            else:
                return None

    def girth(self) -> int:
        """
        Gets the length of the shortest cycle in the graph.
        :return: The length of the shortest cycle or None if no cycle exists.
        """
        shortest_cycle = None
        
        # Iterate through the vertices
        for vertex in self.adj_list:
            # Get the cycle length
            cycle_length = self.get_cycle_length(vertex, compare_function=min)
            
            # Check that there is a cycle
            if cycle_length is not None:
                # Check if the shortest cycle needs updated
                if shortest_cycle is None or cycle_length < shortest_cycle:
                    shortest_cycle = cycle_length
                    
        return shortest_cycle

    def circumference(self) -> int:
        """
        Gets the length of the longest cycle in the graph.
        :return: The length of the longest cycle or None if no cycle exists.
        """
        longest_cycle = None
        
        # Iterate through the vertices
        for vertex in self.adj_list:
            # Get the cycle length
            cycle_length = self.get_cycle_length(vertex, compare_function=max)
            
            # Check that there is a cycle
            if cycle_length is not None:
                # Check if the longest cycle needs updated
                if longest_cycle is None or cycle_length > longest_cycle:
                    longest_cycle = cycle_length
                    
        return longest_cycle

    def bfs(self, start_vertex):
        """
        Perform a breadth-first search on the graph
        :param start_vertex: The starting vertex
        :returns: The traversal order of the search
        """
        # Check that start_vertex is valid
        if not isinstance(start_vertex, Vertex):
            raise Exception("`start_vertex` must be of type `Vertex`!")
         
        # Storing traversal order
        visited = []
        
        # Create the queue
        queue = deque([start_vertex])
        
        # Perform the BFS while the queue is not empty
        while queue:
            # Get the front vertex
            vertex = queue.popleft()
            
            # Check if the vertex has been visited
            if vertex not in visited:
                # Add to the visited list
                visited.append(vertex)
                
                # Iterate trough all the edges for the vertex 
                for edge in self.adj_list[vertex]:
                    # Check if the edge is directed
                    if edge.directed and edge.end_vertex not in visited:
                        # Add the ending vertex
                        queue.append(edge.end_vertex)
                    else:
                        # Add the start vertex if not the current vertex
                        if edge.start_vertex != vertex and edge.start_vertex not in visited:
                            queue.append(edge.start_vertex)
                        # Add the end vertex if not the current vertex
                        if edge.end_vertex != vertex and edge.end_vertex not in visited:
                            queue.append(edge.end_vertex)
                           
        return visited
        
    def transpose(self):
        """
        Creates and returns the transpose of the graph (u -> v) reversed to (v -> u)
        :return: A new GraphV2 object that is transpose of the current graph
        """
        # Initialize an empty adjacency list
        transpose_adj_list = {}
        for vertex in self.adj_list:
            transpose_adj_list[vertex] = []
            
        # Get the edges
        for vertex in self.adj_list:
            for edge in self.adj_list[vertex]:
                # Add reversed edge to the transpose adjacency list
                transpose_adj_list[edge.end_vertex].append(Edge(edge.end_vertex, edge.start_vertex, edge.weight, edge.directed))
                
        # Return the transpose graph
        return GraphV2(transpose_adj_list)
    
    def get_binary_adj_matrix(self) -> list:
        """
        Get a binary adjacancey matrix based on the current graph
        :return: A binary adjacecncy matrix
        """
        # Check that there is a adjacency matrix
        if self.adj_matrix is None and self.adj_list:
            self.__adj_list_to_matrix()
        
        # Initialize a matrix
        binary_matrix = []

        # Iterate through the rows in the matrix
        for row in self.adj_matrix:
            # Matrix row list
            row_data = []

            # Iterate through the columns in the matrix
            for column in row:
                # Check if there is an edge
                if column is not None:
                    row_data.append(1)
                else:
                    row_data.append(0)

            # Insert row into matrix
            binary_matrix.append(row_data)

        return binary_matrix
    
    def get_weight_adj_matrix(self) -> list:
        """
        Get a weight adjacency matrix based on the current graph.
        :return: A weight adjacency matrix
        """
        # Check that there is an adjacency matrix
        if self.adj_matrix is None and self.adj_list:
            self.__adj_list_to_matrix()

        # Initialize a matrix with zeros
        weight_matrix = []

        # Iterate through the rows in the matrix
        for row in self.adj_matrix:
            # Matrix row list
            row_data = []

            # Iterate through the columns in the matrix
            for column in row:
                # Check if there is an edge
                if column is not None:
                    # Use the edge weight if specified, otherwise default to 1
                    row_data.append(column.weight if column.weight is not None else 1)
                else:
                    row_data.append(0)

            # Insert row into matrix
            weight_matrix.append(row_data)

        return weight_matrix
    
    def get_degree_matrix(self):
        """
        Get a degree matrix for the current graph.
        :return: A degree matrix
        """
        # Initialize a matrix with zeros
        degree_matrix = []

        # Fill the matrix with 0's
        for row in range(self.order):
            row_data = []

            for column in range(self.order):
                row_data.append(0)

            degree_matrix.append(row_data)

        # Get the adjacency matrix
        adjacency_matrix = self.get_binary_adj_matrix()

        # Calculate the degree of each node
        for i in range(self.order):
            # Sum the connections for the node (row) in the adjacency matrix
            degree = sum(adjacency_matrix[i])
            degree_matrix[i][i] = degree

        return degree_matrix

    def get_laplacian_matrix(self):
        """
        Get a Laplacian matrix for the current graph
        :return: The Laplacian matrix
        """
        # Initialize a empty matrix
        matrix = []

        for row in range(self.order):
            row_data = []

            for column in range(self.order):
                row_data.append(0)

            matrix.append(row_data)

        # Get the degree matrix
        degree_matrix = self.get_degree_matrix()

        # Get the Adjacency matrix
        adjacency_matrix = self.get_binary_adj_matrix()

        # Get the resulting matrix
        for row in range(self.order):
            for column in range(self.order):
                matrix[row][column] = degree_matrix[row][column] - adjacency_matrix[row][column]

        return matrix
    
    def dijkstras_algorithm(self, start_vertex, output_paths=False):
        """
        Finds the shortest path from the start_vertex to 
        all other vertices using the Dijkstra's Algorithm
        :param start_vertex: The vertex at which the algorithm should start
        :param output_paths: Defaulted to False, if True the function will return the shorestest path to each vertex from the `start_vertex`
        :return: A dictionary of the distances to each vertex, A list of the paths to each vertex from the `start_vertex`
        """
        # Check for valid arguments
        if not isinstance(start_vertex, Vertex):
            raise Exception("`start_vertex` must be of type `Vertex`!")
        
        # Initialize the distances to all the vertices by setting them to infinity
        distances = {key: float('inf') for key in self.adj_list}
        distances[start_vertex] = 0
        
        # Initialize a list to store the visited vertices
        visited = []

        # Initialize a dictionary to store the paths
        if output_paths:
            paths = {}
        
        # Create the queue by adding the starting node
        queue = deque([(start_vertex, 0, [start_vertex])])
        
        # Iterate while there are vertices in the queue
        while queue:
            # Sort the queue using the distance
            queue = deque(sorted(queue, key=lambda x: x[1])) 

            # Get the smallest distant vertex
            current_vertex, current_distance, traversal_list = queue.popleft()
            
            if current_vertex in visited:
                continue
            
            # Add the `current_vertex` to the visited list
            visited.append(current_vertex)
            
            # Iterate through the `current_vertex` edge list
            for edge in self.adj_list[current_vertex]:
                # Get the connecting vertex
                if edge.start_vertex == current_vertex:
                    connecting_vertex = edge.end_vertex
                else:
                    connecting_vertex = edge.start_vertex
                    
                new_distance = current_distance + edge.weight
                
                # Check if the new distance is less than the distance of the last calculation; greedy algorithm
                if new_distance < distances[connecting_vertex]:
                    distances[connecting_vertex] = new_distance

                    updated_traversal_list = traversal_list + [connecting_vertex]

                    if output_paths:
                        paths[connecting_vertex] = updated_traversal_list
                    
                    # Add the connecting vertex to the queue
                    queue.append((connecting_vertex, new_distance, updated_traversal_list))  

        if output_paths:
            return distances, paths
        
        return distances
    
    def a_star(self, start_vertex, end_vertex, heuristic_function, output_paths=False):
        """
        Finds the shortest path from the start_vertex to
        the end_vertex using the A* Algorithm
        :param start_vertex: The vertex at which the algorithm should start
        :param end_vertex: The vertex at which the algorithm is searching for
        :param heuristic_function: The function used to get the estimated weight
        :param output_paths: Defaulted to False, if True the function will return the shorestest path to each vertex from the `start_vertex`
        :return: The distance of the path, A list containing the visited vertices on the path
        """
        # Check for valid arguments
        if not isinstance(start_vertex, Vertex):
            raise Exception("`start_vertex` must be of type `Vertex`!")
        if not isinstance(end_vertex, Vertex):
            raise Exception("`end_vertex` must be of type `Vertex`!")
        if not callable(heuristic_function):
            raise Exception("`heuristic_function` must be callable!")
        
        # Initialize the distances to all the vertices by setting them to infinity
        known_distances = {key: float('inf') for key in self.adj_list}
        known_distances[start_vertex] = 0
        estimated_distances = {key: float('inf') for key in self.adj_list}
        estimated_distances[start_vertex] = heuristic_function(start_vertex, end_vertex)

        # Initialize a dictionary to store the paths
        if output_paths:
            paths = {}

        # Create the queue by adding the starting node
        queue = deque([(start_vertex, 0, [start_vertex])])

        # Iterate while there are vertices in the queue
        while queue:
            # Sort the queue using the distance
            queue = deque(sorted(queue, key=lambda x: x[1] + heuristic_function(x, end_vertex)))

            # Get the smallest `estimated_distant` vertex
            current_vertex, current_distance, traversal_list = queue.popleft()

            if current_vertex == end_vertex:
                if output_paths:
                    return current_distance, traversal_list
                else:
                    return current_distance
            
            # Iterate through the `current_vertex` edge list
            for edge in self.adj_list[current_vertex]:
                # Get the connecting vertex
                if edge.start_vertex == current_vertex:
                    connecting_vertex = edge.end_vertex
                else:
                    connecting_vertex = edge.start_vertex

                new_known_distance = current_distance + edge.weight

                # Check if the new known distance is less than the distance of the last known distance; greedy algorithm
                if new_known_distance < known_distances[connecting_vertex]:
                    known_distances[connecting_vertex] = new_known_distance
                    
                    # f(n) = g(n) + h(n)
                    estimated_distances[connecting_vertex] = new_known_distance + heuristic_function(connecting_vertex, end_vertex)

                    updated_traversal_list = traversal_list + [connecting_vertex]

                    # Add the connecting vertex to the queue
                    queue.append((connecting_vertex, new_known_distance, updated_traversal_list)) 

        return float('inf'), []

    def floyd_warshall_algorithm(self, start_vertex):
        """
        Finds the shortest path from the start_vertex to 
        all other vertices using the Floyd-Warshall Algorithm
        :param start_vertex: The vertex at which the algorithm should start
        :return: A matrix of the distances to each vertex
        """
        # Check for valid arguments
        if not isinstance(start_vertex, Vertex):
            raise Exception("`start_vertex` must be of type `Vertex`!")
        
        # Initialize the matrix
        matrix = []
        for row in range(self.order):
            column_data = []
            for column in range(self.order):
                if row == column:
                    column_data.append(0)   # Vertex distance to itself is 0
                else:
                    column_data.append(float('inf'))    # Initialize to infinity
                
            matrix.append(column_data)
            
        # Get a list of vertices
        vertices = list(self._adj_list.keys())
        
        # Create D(0)
        for row, vertex in enumerate(vertices):
            # Iterate through the edges
            for edge in self.adj_list[vertex]:
                # Find the column based on ending vertex
                for column, value in enumerate(vertices):
                    # Check if start is vertex and end is value
                    if edge.start_vertex == vertex and edge.end_vertex == value:
                        matrix[row][column] = edge.weight
                        
                    # Check if end is vertex and start is value
                    elif edge.end_vertex == vertex and edge.start_vertex == value:
                        matrix[row][column] = edge.weight
                                
        # Iterate once for each vertex           
        for k in range(self.order):
            # Store the old matrix
            last_matrix = matrix
            
            # Iterate through the rows in the matrix
            for i in range(self.order):
                # Iterate through the columns in the matrix
                for j in range(self.order):
                    # Set the value of d(k)ij to min(d(k-1)ij, d(k-1)ik + d(k-1kj))
                    matrix[i][j] = min(last_matrix[i][j], (last_matrix[i][k] + last_matrix[k][j]))
        
        return matrix