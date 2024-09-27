import unittest
from lib.vertex import Vertex
from lib.edge import Edge
from lib.graph_Pl_v2 import GraphV2

class TestGraphInit(unittest.TestCase):
    def test_valid_adj_list_nondirectional(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        AC = Edge(A, C)
        
        # Create the adjacency list
        adj_list = {
            A: [AB, AC],
            B: [AB, BC],
            C: [AC, BC]
        }
        
        # Test initializing the graph via a valid adjacency list
        graph = GraphV2(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Test the order of the graph
        self.assertEqual(len(list(adj_list.keys())), graph.order)
        
        # Test the size of the graph
        self.assertEqual((graph.order * (graph.order - 1)) / 2, graph.size)
        
        # Test the conversion
        adj_matrix = [
            [None, AB, AC],
            [AB, None, BC],
            [AC, BC, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
    def test_valid_adj_list_directional(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CA = Edge(C, A, directed=True)
        
        # Create the adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CA]
        }
        
        # Test initializing the graph via a valid adjacency list
        graph = GraphV2(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Test the order of the graph
        self.assertEqual(len(list(adj_list.keys())), graph.order)
        
        # Test the size of the graph
        self.assertEqual(graph.order * (graph.order - 1) / 2, graph.size)
        
        # Test the conversion
        adj_matrix = [
            [None, AB, None],
            [None, None, BC],
            [CA, None, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
    def test_valid_adj_matrix(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        AC = Edge(A, C)
        
        # Create the adjacency list
        adj_matrix = [
            [None, AB, AC],
            [AB, None, BC],
            [AB, BC, None]
        ]
        
        # Test initializing the graph via a valid adjacency list
        graph = GraphV2(adj_matrix)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the order size of the graph
        self.assertEqual(len(adj_matrix), graph.order)
        
        # Test the size of the graph
        self.assertEqual(graph.order * (graph.order - 1) / 2, graph.size)
        
class TestGraphMethods(unittest.TestCase):
    def test_get_vertex_weight(self):
        """Test getting the weight of a vertex via the graph class"""
        # Vertices
        A = Vertex('A', 1)
        Z = Vertex('Z', 26)
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test initialization
        adj_matrix = [
            [None,None],
            [None,None]
        ]
        self.assertEqual(adj_list, graph.adj_list)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the weight getter
        self.assertEqual(A.weight, graph.get_vertex_weight(A))
        self.assertEqual(Z.weight, graph.get_vertex_weight(Z))
        
    def test_invalid_get_vertex_weight(self):
        """Test getting the weight of a vertex via the graph class"""
        # Vertices
        A = Vertex('A', 1)
        Z = Vertex('Z', 26)
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test initialization
        adj_matrix = [
            [None,None],
            [None,None]
        ]
        self.assertEqual(adj_list, graph.adj_list)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the weight getter
        self.assertRaises(Exception, graph.get_vertex_weight, "A")
        self.assertRaises(Exception, graph.get_vertex_weight, "Z")

        
    def test_set_vertex_weight(self):
        """Test getting the weight of a vertex via the graph class"""
        # Vertices
        A = Vertex('A', 1)
        Z = Vertex('Z', 26)
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test initialization
        adj_matrix = [
            [None,None],
            [None,None]
        ]
        self.assertEqual(adj_list, graph.adj_list)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        self.assertEqual(A.weight, graph.get_vertex_weight(A))
        self.assertEqual(Z.weight, graph.get_vertex_weight(Z))
        
        # Test the setter
        graph.set_vertex_weight(A, 26)
        graph.set_vertex_weight(Z, 1)
        self.assertEqual(A.weight, graph.get_vertex_weight(A))
        self.assertEqual(Z.weight, graph.get_vertex_weight(Z))
        
    def test_invalid_set_vertex_weight(self):
        """Test getting the weight of a vertex via the graph class"""
        # Vertices
        A = Vertex('A', 1)
        Z = Vertex('Z', 26)
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test initialization
        adj_matrix = [
            [None,None],
            [None,None]
        ]
        self.assertEqual(adj_list, graph.adj_list)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the weight getter
        self.assertRaises(Exception, graph.set_vertex_weight, "A", "FAILURE")
        self.assertRaises(Exception, graph.set_vertex_weight, "Z", "FAILURE")