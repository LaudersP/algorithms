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