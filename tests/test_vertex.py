import unittest
from lib.vertex import Vertex

class TestVertex(unittest.TestCase):
    def test_init_no_weight(self):
        # Test initializing the vertex
        A = Vertex('A')
        self.assertEqual(A.id, 'A')
        self.assertIsNone(A.weight)
        
    def test_init_weighted(self):
        # Test initializing the vertex with weight
        A = Vertex('A', 5.0)
        self.assertEqual(A.id, 'A')
        self.assertEqual(A.weight, 5.0)
        self.assertIsInstance(A.weight, float)
    
    def test_changing_weights(self):
        # Test changing the weight of a vertex
        A = Vertex('A', "Bob")
        self.assertEqual(A.id, 'A')
        self.assertEqual(A.weight, "Bob")
        
        A.weight = "Alice"
        self.assertEqual(A.id, 'A')
        self.assertEqual(A.weight, "Alice")