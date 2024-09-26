import unittest
from lib.vertex import Vertex
from lib.edge import Edge

class TestEdge(unittest.TestCase):
    def test_init_no_weight_no_direction(self):
        # Test initializing the edge
        A = Vertex('A', 3.14)
        B = Vertex('B', 3.14)
        
        AB = Edge(A, B)
        self.assertEqual(AB.start_vertex, A)
        self.assertEqual(AB.end_vertex , B)
        self.assertIsNone(AB.weight)
        self.assertEqual(AB.directed, False)
        
    def test_init_weighted_no_direction(self):
        # Test initializing the edge with weight
        A = Vertex('A', 3.14)
        B = Vertex('B', 3.14)
        
        AB = Edge(A, B, "PI")
        self.assertEqual(AB.start_vertex, A)
        self.assertEqual(AB.end_vertex , B)
        self.assertEqual(AB.weight , "PI")
        self.assertEqual(AB.directed, False)
        
    def test_init_no_weight_directional(self):
        # Test initializing the edge with direction
        A = Vertex('A', 3.14)
        B = Vertex('B', 3.14)
        
        AB = Edge(A, B, directed=True)
        self.assertEqual(AB.start_vertex, A)
        self.assertEqual(AB.end_vertex , B)
        self.assertIsNone(AB.weight)
        self.assertEqual(AB.directed, True)
        
    def test_init_weighted_directional(self):
        # Test initializing the edge with weight and direction
        A = Vertex('A', 3.14)
        B = Vertex('B', 3.14)
        
        AB = Edge(A, B, "PI", True)
        self.assertEqual(AB.start_vertex, A)
        self.assertEqual(AB.end_vertex , B)
        self.assertEqual(AB.weight , "PI")
        self.assertEqual(AB.directed, True)
        
    def test_changing_weights(self):
        # Test initializing the edge with weight and direction
        A = Vertex('A', 3.14)
        B = Vertex('B', 3.14)
        
        AB = Edge(A, B, "PI", True)
        self.assertEqual(AB.start_vertex, A)
        self.assertEqual(AB.end_vertex , B)
        self.assertEqual(AB.weight , "PI")
        self.assertEqual(AB.directed, True)
        
        AB.weight = "Pumpkin Pie!"
        self.assertEqual(AB.weight, "Pumpkin Pie!")