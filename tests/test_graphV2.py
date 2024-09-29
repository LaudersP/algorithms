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
        AB = Edge(A, B)
        BC = Edge(B, C)
        CA = Edge(C, A)
        
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
        AZ = Edge(A, Z)
        
        # Create the adjacency list
        adj_list = {
            A: [AZ],
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
        AZ = Edge(A, Z)
        ZA = Edge(Z, A)
        
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
        
        # Test the connectivity
        self.assertTrue(graph.is_connected())