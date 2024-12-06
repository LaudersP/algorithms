import unittest
from lib.vertex import Vertex
from lib.edge import Edge
from lib.graph_Pl_v2 import GraphV2
import numpy as np
import math

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
        AB = Edge(A, B)
        BC = Edge(B, C)
        CA = Edge(C, A)
        
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
        
    def test_valid_adj_matrix_nondirectional(self):
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
        self.assertEqual((graph.order * (graph.order - 1)) / 2, graph.size)
        
        # Test the conversion
        adj_list = {
            A: [AB, AC],
            B: [AB, BC],
            C: [AC, BC]
        }
        self.assertEqual(adj_list, graph.adj_list)
        
    def test_valid_adj_matrix_directional(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CA = Edge(C, A, directed=True)
        
        # Create the adjacency list
        adj_matrix = [
            [None, AB, None],
            [None, None, BC],
            [CA, None, None]
        ]
        
        # Test initializing the graph via a valid adjacency list
        graph = GraphV2(adj_matrix)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the order size of the graph
        self.assertEqual(len(adj_matrix), graph.order)
        
        # Test the size of the graph
        self.assertEqual((graph.order * (graph.order - 1)) / 2, graph.size)
        
        # Test the conversion
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CA]
        }
        self.assertEqual(adj_list, graph.adj_list)
        
    def test_invalid_init(self):
        self.assertRaises(Exception, GraphV2("adj_list"))
        
class TestGraphVertexMethods(unittest.TestCase):
    def test_get_vertex_weight_none(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the weight getter
        self.assertIsNone(graph.get_vertex_weight(A))
        self.assertIsNone(graph.get_vertex_weight(Z))
        
    def test_invalid_get_vertex_weight(self):
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
        
        # Test the weight getter
        self.assertRaises(Exception, graph.get_vertex_weight, "A")
        self.assertRaises(Exception, graph.get_vertex_weight, "Z")
        
    def test_get_vertex_weight(self):
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
        
        # Test the weight getter
        self.assertEqual(A.weight, graph.get_vertex_weight(A))
        self.assertEqual(Z.weight, graph.get_vertex_weight(Z))
        
    def test_invalid_set_vertex_weight(self):
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
        
        # Test the weight getter
        self.assertRaises(Exception, graph.set_vertex_weight, "A", "FAILURE")
        self.assertRaises(Exception, graph.set_vertex_weight, "Z", "FAILURE")
        
    def test_set_vertex_weight(self):
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
        
        # Test the setter
        graph.set_vertex_weight(A, 26)
        graph.set_vertex_weight(Z, 1)
        self.assertEqual(A.weight, graph.get_vertex_weight(A))
        self.assertEqual(Z.weight, graph.get_vertex_weight(Z))
        
    def test_add_vertex(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Add vertex
        P = Vertex('P')
        graph.add_vertex(P)
        
        # Test the results
        adj_list = {
            A: [AZ],
            P: [],
            Z: [AZ]
        }
        self.assertEqual(adj_list, graph.adj_list)
        
        adj_matrix = [
            [None,None,AZ],
            [None,None,None],
            [AZ,None,None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
    def test_add_weighted_vertex(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Add vertex
        P = Vertex('P', 3.12)
        graph.add_vertex(P)
        
        # Test the results
        adj_list = {
            A: [AZ],
            P: [],
            Z: [AZ]
        }
        self.assertEqual(adj_list, graph.adj_list)
        
        adj_matrix = [
            [None,None,AZ],
            [None,None,None],
            [AZ,None,None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
    def test_add_invalid_vertex(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Add vertex
        P = Vertex('P', 3.12)
        self.assertRaises(Exception, graph.add_vertex, 'P')
        
    def test_del_vertex(self):
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
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Delete vertex
        graph.del_vertex(C)
        
        # Test the results
        adj_list = {
            A: [AB],
            B: [AB]
        }
        self.assertEqual(adj_list, graph.adj_list)
        adj_matrix = [
            [None, AB],
            [AB, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
    def test_del_weighted_vertex(self):
        # Vertices
        A = Vertex('A', 1)
        B = Vertex('B', 2)
        C = Vertex('C', 3)
        
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
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Delete vertex
        graph.del_vertex(C)
        
        # Test the results
        adj_list = {
            A: [AB],
            B: [AB]
        }
        self.assertEqual(adj_list, graph.adj_list)
        adj_matrix = [
            [None, AB],
            [AB, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
    def test_del_invalid_vertex(self):
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
        
        # Remove vertex
        self.assertRaises(Exception, graph.del_vertex, 'Z')
        
class TestGraphEdgeMethods(unittest.TestCase):
    def test_get_edge_weight_none(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the weight getter
        self.assertIsNone(graph.get_edge_weight(AZ))
    
    def test_invalid_get_edge_weight(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z, 26)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the weight getter
        self.assertRaises(Exception, graph.get_edge_weight, "AZ")
    
    def test_get_edge_weight(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z, 26)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the weight getter
        self.assertEqual(AZ.weight, graph.get_edge_weight(AZ))
    
    def test_invalid_set_edge_weight(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z, 26)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the weight getter
        self.assertRaises(Exception, graph.set_edge_weight, "AZ", 0)
    
    def test_set_edge_weight(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z, 0)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the weight getter
        graph.set_edge_weight(AZ, 26)
        self.assertEqual(AZ.weight, graph.get_edge_weight(AZ))
    
    def test_add_edge(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Add edge
        graph.add_edge(AZ)
        
        # Test the results
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        self.assertEqual(adj_list, graph.adj_list)
        adj_matrix = [
            [None, AZ],
            [AZ, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
    
    def test_add_invalid_edge(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Add edge
        self.assertRaises(Exception, graph.add_edge, "AZ")
    
    def test_add_directed_edge(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z, directed=True)
        
        # Create the adjacency list
        adj_list = {
            A: [],
            Z: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Add edge
        graph.add_edge(AZ)
        
        # Test the results
        adj_list = {
            A: [AZ],
            Z: []
        }
        self.assertEqual(adj_list, graph.adj_list)
        adj_matrix = [
            [None, AZ],
            [None, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
    
    def test_del_edge(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Del edge
        graph.del_edge(AZ)
        
        # Test the results
        adj_list = {
            A: [],
            Z: []
        }
        self.assertEqual(adj_list, graph.adj_list)
        adj_matrix = [
            [None, None],
            [None, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
    
    def test_del_directed_edge(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z, directed=True)
        ZA = Edge(Z, A, directed=True)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [ZA]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Del edges
        graph.del_edge(AZ)
        graph.del_edge(ZA)
        
        # Test the results
        adj_list = {
            A: [],
            Z: []
        }
        self.assertEqual(adj_list, graph.adj_list)
        adj_matrix = [
            [None, None],
            [None, None]
        ]
        self.assertEqual(adj_matrix, graph.adj_matrix)
    
    def test_del_invalid_edge(self):
        # Vertices
        A = Vertex('A')
        Z = Vertex('Z')
        
        # Edges
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
            Z: [AZ]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Delete edge
        self.assertRaises(Exception, graph.del_edge, "AZ")
        
    def test_add_edge_disallow_multiple_edges(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')

        # Edges
        edge1 = Edge(A, B)
        edge2 = Edge(A, B)  # Another edge between A and B

        # Create the adjacency list
        adj_list = {
            A: [edge1],
            B: [edge1]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Attempt to add another edge between A and B without allowing multiple edges
        with self.assertRaises(Exception) as context:
            graph.add_edge(edge2)

        # Check the exception message
        self.assertEqual(
            str(context.exception),
            "Multiple edges between the same vertices are not allowed unless 'allow_multiple_edges' is True!"
        )

    def test_add_edge_allow_multiple_edges(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')

        # Edges
        edge1 = Edge(A, B)
        edge2 = Edge(A, B)  # Another edge between A and B

        # Create the adjacency list
        adj_list = {
            A: [edge1],
            B: [edge1]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Add another edge between A and B allowing multiple edges
        graph.add_edge(edge2, allow_multiple_edges=True)

        # Test the results
        adj_list = {
            A: [edge1, edge2],
            B: [edge1, edge2]
        }
        self.assertEqual(adj_list, graph.adj_list)

    def test_add_loop_disallow_loops(self):
        # Vertices
        A = Vertex('A')

        # Edge (loop)
        loop_edge = Edge(A, A)

        # Create the adjacency list
        adj_list = {
            A: []
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Attempt to add a loop without allowing loops
        with self.assertRaises(Exception) as context:
            graph.add_edge(loop_edge)

        # Check the exception message
        self.assertEqual(
            str(context.exception),
            "Loops are not allowed unless 'allow_multiple_edges' is True!"
        )

    def test_add_loop_allow_loops(self):
        # Vertices
        A = Vertex('A')

        # Edge (loop)
        loop_edge = Edge(A, A)

        # Create the adjacency list
        adj_list = {
            A: []
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Add a loop allowing loops
        graph.add_edge(loop_edge, allow_multiple_edges=True)

        # Test the results
        adj_list = {
            A: [loop_edge]
        }
        self.assertEqual(adj_list, graph.adj_list)
        
class TestGraphMethods(unittest.TestCase):
    def test_is_directed_not_directed(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        AC = Edge(A, C)
        
        # Adjacency matrix
        adj_matrix = [
            [None, AB, AC],
            [AB, None, BC],
            [AC, BC, None]
        ]
        
        # Initialize the graph
        graph = GraphV2(adj_matrix)
        
        # Test for direction
        self.assertFalse(graph.is_directed())
    
    def test_is_directed_directed(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        CA = Edge(A, C)
        
        # Adjacency matrix
        adj_matrix = [
            [None, AB, None],
            [None, None, BC],
            [CA, None, None]
        ]
        
        # Initialize the graph
        graph = GraphV2(adj_matrix)
        
        # Test for direction
        self.assertTrue(graph.is_directed())
        
    def test_bfs(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        F = Vertex('F')
        G = Vertex('G')
        H = Vertex('H')
        I = Vertex('I')

        # Edges
        AB = Edge(A, B)  
        AC = Edge(A, C)  
        BD = Edge(B, D)  
        CE = Edge(C, E)  
        DF = Edge(D, F)  
        EF = Edge(E, F)  
        EG = Edge(E, G)  
        FH = Edge(F, H)  
        GI = Edge(G, I)  
        HC = Edge(H, C)  
        GB = Edge(G, B)  

        # Adjacency list
        adj_list = {
            A: [AB, AC],  
            B: [BD],      
            C: [CE],      
            D: [DF],      
            E: [EF, EG],  
            F: [FH],      
            G: [GI, GB],  
            H: [HC],      
            I: []         
        }

        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Perform the BFS
        desired_results = [A, B, C, D, E, F, G, H, I]
        self.assertEqual(desired_results, graph.bfs(A))
    
    def test_bfs_directional(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        F = Vertex('F')
        G = Vertex('G')
        H = Vertex('H')
        I = Vertex('I')

        # Edges
        AB = Edge(A, B, directed=True)
        AC = Edge(A, C, directed=True)
        BD = Edge(B, D, directed=True)
        CE = Edge(C, E, directed=True)
        DF = Edge(D, F, directed=True)
        EF = Edge(E, F, directed=True)
        EG = Edge(E, G, directed=True)
        FH = Edge(F, H, directed=True)
        GI = Edge(G, I, directed=True)
        HC = Edge(H, C, directed=True)
        GB = Edge(G, B, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB, AC],  
            B: [BD],      
            C: [CE],      
            D: [DF],      
            E: [EF, EG],  
            F: [FH],      
            G: [GI, GB],  
            H: [HC],      
            I: []         
        }

        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Perform the BFS
        desired_output = [D, F, H, C, E, G, I, B]
        self.assertEqual(desired_output, graph.bfs(D))
    
    def test_is_connected_not_connected(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')

        # Edges
        AB = Edge(A, B, directed=True)
        CD = Edge(C, D, directed=True)
        DE = Edge(D, E, directed=True)
        
        # Adjacency list
        adj_list = {
            A: [AB],  
            B: [],      
            C: [CD],      
            D: [DE],      
            E: []         
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the connectivity
        self.assertFalse(graph.is_connected())
    
    def test_is_connected_connected(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        F = Vertex('F')
        G = Vertex('G')
        H = Vertex('H')
        I = Vertex('I')

        # Edges
        AB = Edge(A, B, directed=True)
        AC = Edge(A, C, directed=True)
        BD = Edge(B, D, directed=True)
        CE = Edge(C, E, directed=True)
        DF = Edge(D, F, directed=True)
        DA = Edge(D, A, directed=True)
        EF = Edge(E, F, directed=True)
        EG = Edge(E, G, directed=True)
        FE = Edge(F, E, directed=True)
        FH = Edge(F, H, directed=True)
        GI = Edge(G, I, directed=True)
        GB = Edge(G, B, directed=True)
        IA = Edge(I, A, directed=True)
        HC = Edge(H, C, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB, AC],
            B: [BD],
            C: [CE],
            D: [DF, DA],
            E: [EF, EG],
            F: [FH, FE],
            G: [GI, GB],
            H: [HC],
            I: [IA]     
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Test the connectivity
        self.assertTrue(graph.is_connected())

        
    def test_is_tree_disconnected_graph(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        
        # Edges
        AB = Edge(A, B)
        CD = Edge(C, D)
        
        # Adjacency list
        adj_list = {
            A: [AB],
            B: [AB],
            C: [CD],
            D: [CD]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check if the graph is a tree
        self.assertFalse(graph.is_tree())
        
    def test_is_tree_cycle_graph(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        CA = Edge(C, A)
        
        # Create adjacency list
        adj_list = {
            A: [AB, CA],
            B: [AB, BC],
            C: [BC, CA]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check if the graph is a tree
        self.assertFalse(graph.is_tree())
    
    def test_is_tree_tree(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        
        # Edges
        AB = Edge(A, B, directed=False)
        AC = Edge(A, C, directed=False)
        BD = Edge(B, D, directed=False)
        BE = Edge(B, E, directed=False)

        # Adjacency list
        adj_list = {
            A: [AB, AC],
            B: [AB, BD, BE],
            C: [AC],
            D: [BD],
            E: [BE]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check if the graph is a tree
        self.assertTrue(graph.is_tree())
        
    def test_is_unilaterally_connected_not(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        
        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CD = Edge(C, D, directed=True)
        
        # Adjacency list
        adj_list = {
            A: [AB],   
            B: [BC],   
            C: [CD],   
            D: [],     
            E: []      
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list) 
        
        # Check if the graph is not unilaterally connected
        self.assertFalse(graph.is_unilaterally_connected())  
    
    def test_is_unilaterally_connected(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        
        # Edges 
        AB = Edge(A, B, directed=True)  
        BC = Edge(B, C, directed=True)  
        CD = Edge(C, D, directed=True)  
        DE = Edge(D, E, directed=True)  
        EA = Edge(E, A, directed=True)  

        # Adjacency list 
        adj_list = {
            A: [AB],  
            B: [BC],  
            C: [CD],  
            D: [DE],  
            E: [EA]   
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check if the graph is unilaterally connected
        self.assertTrue(graph.is_unilaterally_connected())
        
    def test_transpose(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')

        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        AC = Edge(A, C, directed=True)

        # Adjacency list 
        adj_list = {
            A: [AB, AC],
            B: [BC],    
            C: [],     
        }

        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Transpose graph
        transposed_graph = graph.transpose()
        
        # Test the results
        BA = Edge(B, A, directed=True)
        CB = Edge(C, B, directed=True)
        CA = Edge(C, A, directed=True)
        transposed_adj_list = {
            A: [],
            B: [BA],
            C: [CA, CB]
        }
        self.assertEqual(transposed_adj_list, transposed_graph.adj_list)
        
    def test_strongly_connected_not(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        
        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check if it is strongly connected
        self.assertFalse(graph.is_strongly_connected())
    
    def test_strongly_connected(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CA = Edge(C, A, directed=True)
        
        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CA]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check if it is strongly connected
        self.assertTrue(graph.is_strongly_connected())
        
    def test_components(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        F = Vertex('F')
        
        # Edge
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CA = Edge(C, A, directed=True)
        DE = Edge(D, E)
        
        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CA],
            D: [DE],
            E: [DE],
            F: []
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Get the number of components
        self.assertEqual(3, graph.components())
        
    def test_is_weakly_connected_not(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')

        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [],
            D: []  
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check if the graph is weakly connected
        self.assertFalse(graph.is_weakly_connected())

    def test_is_weakly_connected(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')

        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CD = Edge(C, D, directed=True)
        DA = Edge(D, A, directed=True)  

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CD],
            D: [DA]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check if the graph is weakly connected
        self.assertTrue(graph.is_weakly_connected())
        
    def test_girth_directed_multiple_cycles(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')

        # Edges (creating multiple directed cycles)
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CD = Edge(C, D, directed=True)
        DA = Edge(D, A, directed=True)
        CE = Edge(C, E, directed=True)
        EC = Edge(E, C, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CD, CE],
            D: [DA],
            E: [EC]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check the graph's girth (shortest cycle should be C -> E -> C, length 2)
        self.assertEqual(2, graph.girth())

    def test_girth_undirected_multiple_cycles(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')

        # Edges (creating multiple undirected cycles)
        AB = Edge(A, B)
        BC = Edge(B, C)
        CD = Edge(C, D)
        DA = Edge(D, A)
        CE = Edge(C, E)

        # Adjacency list
        adj_list = {
            A: [AB, DA],
            B: [AB, BC],
            C: [BC, CD, CE],
            D: [CD, DA],
            E: [CE]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check the graph's girth (shortest cycle should be A -> B -> C -> D -> A, length 4)
        self.assertEqual(4, graph.girth())

    def test_girth_directed_no_cycles(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')

        # Edges (no cycles)
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CD = Edge(C, D, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CD],
            D: []
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check the graph's girth (no cycles, should be None)
        self.assertIsNone(graph.girth())

    def test_circumference_directed_multiple_cycles(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')

        # Edges
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CD = Edge(C, D, directed=True)
        DA = Edge(D, A, directed=True)
        CE = Edge(C, E, directed=True)
        EC = Edge(E, C, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CD, CE],
            D: [DA],
            E: [EC]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check the graph's circumference (longest cycle should be A -> B -> C -> D -> A, length 4)
        self.assertEqual(4, graph.circumference())

    def test_circumference_undirected_multiple_cycles(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')

        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        CD = Edge(C, D)
        DA = Edge(D, A)
        CE = Edge(C, E)

        # Adjacency list
        adj_list = {
            A: [AB, DA],
            B: [AB, BC],
            C: [BC, CD, CE],
            D: [CD, DA],
            E: [CE]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check the graph's circumference (longest cycle should be A -> B -> C -> D -> A, length 4)
        self.assertEqual(6, graph.circumference())

    def test_circumference_directed_no_cycles(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')

        # Edges (no cycles)
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CD = Edge(C, D, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CD],
            D: []
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check the graph's circumference (no cycles, should be None)
        self.assertIsNone(graph.circumference())

    # Tests for get_cycle_length

    def test_get_cycle_length(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')

        # Edges (directed cycle A -> B -> C -> D -> A)
        AB = Edge(A, B, directed=True)
        BC = Edge(B, C, directed=True)
        CD = Edge(C, D, directed=True)
        DA = Edge(D, A, directed=True)

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [BC],
            C: [CD],
            D: [DA]
        }

        # Initialize the graph
        graph = GraphV2(adj_list)

        # Check the length of the cycle starting from vertex A
        self.assertEqual(4, graph.get_cycle_length(A))

        # Check the length of the cycle starting from vertex C (should also be 4)
        self.assertEqual(4, graph.get_cycle_length(C))

class TestGraphMatrixMethods(unittest.TestCase):
    def setUp(self):
        # Vertices
        self.A = Vertex('A')
        self.B = Vertex('B')
        self.C = Vertex('C')
        self.D = Vertex('D')

        # Edges
        self.AB = Edge(self.A, self.B)
        self.BC = Edge(self.B, self.C)
        self.CD = Edge(self.C, self.D)
        self.DA = Edge(self.D, self.A)
        self.AC = Edge(self.A, self.C, weight=5)

        # Create adjacency matrix
        self.adj_matrix = [
            [None, self.AB, self.AC, None],
            [self.AB, None, self.BC, None],
            [self.AC, self.BC, None, self.CD],
            [None, None, self.CD, None]
        ]

        # Initialize the graph with the adjacency matrix
        self.graph = GraphV2(self.adj_matrix)

    def test_get_binary_adj_matrix(self):
        # Get the binary matrix
        binary_matrix = self.graph.get_binary_adj_matrix()

        # Create expected matrix
        expected_binary_matrix = [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ]

        # Compare
        self.assertEqual(binary_matrix, expected_binary_matrix)

    def test_get_weight_adj_matrix(self):
        # Get the weight matrix
        weight_matrix = self.graph.get_weight_adj_matrix()

        # Create expected matrix
        expected_weight_matrix = [
            [0, 1, 5, 0],
            [1, 0, 1, 0],
            [5, 1, 0, 1],
            [0, 0, 1, 0]
        ]

        # Compare
        self.assertEqual(weight_matrix, expected_weight_matrix)

    def test_get_degree_matrix(self):
        # Get the degree matrix
        degree_matrix = self.graph.get_degree_matrix()

        # Create expected matrix
        expected_degree_matrix = [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 1]
        ]

        # Compare
        self.assertEqual(degree_matrix, expected_degree_matrix)

    def test_get_laplacian_matrix(self):
        # Get the laplacian matrix
        laplacian_matrix = self.graph.get_laplacian_matrix()

        # Create expected matrix
        expected_laplacian_matrix = [
            [2, -1, -1, 0],
            [-1, 2, -1, 0],
            [-1, -1, 3, -1],
            [0, 0, -1, 1]
        ]

        # Compare
        self.assertEqual(laplacian_matrix, expected_laplacian_matrix)
    
    def test_adj_matrix_to_2D_numpy_array(self):
        # Get the numpy array
        numpy_array = self.graph.adj_matrix_to_2D_numpy_array()
        
        # Create expected array
        expected_numpy_array = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])

        # Compare
        self.assertTrue((numpy_array == expected_numpy_array).all())
        
class TestDijkstraAlgorithm(unittest.TestCase):
    def test_dijkstra_not_directed(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, weight=1)
        AC = Edge(A, C, weight=4)
        BC = Edge(B, C, weight=2)
        
        # Create the adjacency list
        adj_list = {
            A: [AB, AC],
            B: [AB, BC],
            C: [AC, BC]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Run Dijkstra
        distances = graph.dijkstras_algorithm(A)
        
        # Expected distances
        expected = {
            A: 0,
            B: 1,
            C: 3
        }
        self.assertEqual(distances, expected)
        
    def test_dijkstra_directed(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        
        # Edges
        AD = Edge(A, D, weight=2, directed=True)
        BA = Edge(B, A, weight=1, directed=True)
        BC = Edge(B, C, weight=3, directed=True)
        CA = Edge(C, A, weight=4, directed=True)
        DC = Edge(D, C, weight=5, directed=True)
        
        # Create the adjacency list
        adj_matrix = [
            [None, None, None, AD],
            [BA, None, BC, None],
            [CA, None, None, None],
            [None, None, DC, None]
        ]
        
        # Initialize the graph
        graph = GraphV2(adj_matrix)
        
        # Run Dijkstra
        distances = graph.dijkstras_algorithm(A)
        
        # Expected distances
        expected = {
            A: 0,
            B: float('inf'),
            C: 7,
            D: 2
        }
        self.assertEqual(distances, expected)
        
    def test_dijkstra_invalid_start_vertex(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        
        # Edges
        AB = Edge(A, B, weight=1)
        
        # Create the adjacency list
        adj_list = {
            A: [AB],
            B: [AB]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check invalid start_vertex (not a Vertex object)
        with self.assertRaises(Exception) as context:
            graph.dijkstras_algorithm("Invalid Start")
        self.assertEqual(
            str(context.exception),
            "`start_vertex` must be of type `Vertex`!"
        )

class TestAStarAlgorithm(unittest.TestCase):
    def test_a_star_not_directed(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        
        # Edges
        AB = Edge(A, B, weight=1)
        BC = Edge(B, C, weight=2)
        AD = Edge(A, D, weight=4)
        CD = Edge(C, D, weight=3)
        
        # Create the adjacency list
        adj_list = {
            A: [AB, AD],
            B: [AB, BC],
            C: [BC, CD],
            D: [AD, CD]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Heuristics
        def heuristic(start_vertex, end_vertex):
            heuristics = {
                A: 0,
                B: 1,
                C: 2,
                D: 3
            }
            
            return heuristics[end_vertex]
        
        # Run A* Algorithm
        cost, path = graph.a_star(A, D, heuristic, output_paths=True)
        
        # Expected cost and path
        self.assertEqual(cost, 4)
        self.assertEqual(path, [A, D])
     
    def test_a_star_directed(self):   
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        
        # Edges
        AD = Edge(A, D, weight=2, directed=True)
        BA = Edge(B, A, weight=1, directed=True)
        BC = Edge(B, C, weight=3, directed=True)
        CA = Edge(C, A, weight=4, directed=True)
        DC = Edge(D, C, weight=5, directed=True)
        
        # Create the adjacency list
        adj_matrix = [
            [None, None, None, AD],
            [BA, None, BC, None],
            [CA, None, None, None],
            [None, None, DC, None]
        ]
        
        # Initialize the graph
        graph = GraphV2(adj_matrix)
        
        # Heuristics
        def heuristic(start_vertex, end_vertex):
            heuristics = {
                A: 0,
                B: float('inf'),
                C: 2,
                D: 1
            }
            
            return heuristics[end_vertex]
        
        # Run Dijkstra
        cost, path = graph.a_star(A, C, heuristic, output_paths=True)
        
        # Expected cost and path
        self.assertEqual(cost, 7)
        self.assertEqual(path, [A, D, C])
        
    def test_a_star_invalid_start_vertex(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, weight=1)
        BC = Edge(B, C, weight=2)
        
        # Heuristics
        heuristics = {
            A: 3,
            B: 2,
            C: 1
        }
        
        # Create the adjacency list
        adj_list = {
            A: [AB],
            B: [AB, BC],
            C: [BC]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check invalid start_vertex (not a Vertex object)
        with self.assertRaises(Exception) as context:
            graph.a_star("Invalid Start", C, heuristics)
        self.assertEqual(
            str(context.exception),
            "`start_vertex` must be of type `Vertex`!"
        )
        
    def test_a_star_invalid_end_vertex(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, weight=1)
        BC = Edge(B, C, weight=2)
        
        # Heuristics
        heuristics = {
            A: 3,
            B: 2,
            C: 1
        }
        
        # Create the adjacency list
        adj_list = {
            A: [AB],
            B: [AB, BC],
            C: [BC]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check invalid start_vertex (not a Vertex object)
        with self.assertRaises(Exception) as context:
            graph.a_star(A, "Invalid Start", heuristics)
        self.assertEqual(
            str(context.exception),
            "`end_vertex` must be of type `Vertex`!"
        )
    
    def test_a_star_invalid_heuristics(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, weight=1)
        BC = Edge(B, C, weight=2)
        
        # Heuristics
        heuristics = "Invalid Heuristics"  # Not a dictionary
        
        # Create the adjacency list
        adj_list = {
            A: [AB],
            B: [AB, BC],
            C: [BC]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check invalid heuristics (not a dictionary)
        with self.assertRaises(Exception) as context:
            graph.a_star(A, C, heuristics)
        self.assertEqual(
            str(context.exception),
            "`heuristic_function` must be callable!"
        )

class TestFloydWarshallAlgorithm(unittest.TestCase):
    def test_floyd_warshall_not_directed(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, weight=1)
        BC = Edge(B, C, weight=2)
        AC = Edge(A, C, weight=4)
        
        # Create the adjacency list
        adj_list = {
            A: [AB, AC],
            B: [AB, BC],
            C: [AC, BC]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Run Floyd-Warshall
        dist_matrix = graph.floyd_warshall_algorithm(A)
        
        # Expected distance matrix
        expected_dist_matrix = [
            [0, 1, 3],
            [1, 0, 2],
            [3, 2, 0]
        ]
        
        # Compare the matrices
        self.assertEqual(dist_matrix, expected_dist_matrix)

    def test_floyd_warshall_invalid_start_vertex(self):
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        
        # Edges
        AB = Edge(A, B, weight=1)
        BC = Edge(B, C, weight=2)
        
        # Create the adjacency list
        adj_list = {
            A: [AB],
            B: [AB, BC],
            C: [BC]
        }
        
        # Initialize the graph
        graph = GraphV2(adj_list)
        
        # Check invalid start_vertex (not a Vertex object)
        with self.assertRaises(Exception) as context:
            graph.floyd_warshall_algorithm("Invalid Start")
        self.assertEqual(
            str(context.exception),
            "`start_vertex` must be of type `Vertex`!"
        )