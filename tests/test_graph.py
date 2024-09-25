import unittest
from lib.vector import Vector
from lib.matrix import Matrix
from lib.graph_pl import Graph

class TestGraphInit(unittest.TestCase):
    def test_valid_adj_list(self):
        # Test initializing the graph via a valid adjacency list
        adj_list = {
            1: Vector(2, 3, 4),
            2: Vector(1, 3, 4),
            3: Vector(1, 2, 4),
            4: Vector(1, 2, 3)
        }
        
        graph = Graph(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Test the order size
        self.assertEqual(len(list(adj_list.keys())), graph.order)
        
        # Test the converted matrix
        adj_matrix = Matrix(
            Vector(0, 1, 1, 1),
            Vector(1, 0, 1, 1),
            Vector(1, 1, 0, 1),
            Vector(1, 1, 1, 0)
        )
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the size
        self.assertEqual(graph.order * (graph.order - 1) / 2, graph.size)
        
    def test_invalid_adj_list(self):
        # Test initializing the graph via a invalid adjacency list
        adj_list = {
            1: [2, 3, 4],
            2: [1, 3, 4],
            3: [1, 2, 4],
            4: [1, 2, 3]
        }
        
        self.assertRaises(Exception, Graph(adj_list))
    
    def test_valid_adj_matrix(self):
        # Test initializing the graph via a valid adjacency matrix
        adj_matrix = Matrix(
            Vector(0, 1, 1, 1),
            Vector(1, 0, 1, 1),
            Vector(1, 1, 0, 1),
            Vector(1, 1, 1, 0)
        )
        
        graph = Graph(adj_matrix)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the order size
        self.assertEqual(adj_matrix.num_rows, graph.order)
        
        # Test the converted list
        adj_list = {
            1: Vector(2, 3, 4),
            2: Vector(1, 3, 4),
            3: Vector(1, 2, 4),
            4: Vector(1, 2, 3)
        }
        self.assertEqual(adj_list, graph.adj_list)
        
        # Test the size
        self.assertEqual(graph.order * (graph.order - 1) / 2, graph.size)
        
    def test_invalid_adj_matrix(self):
        # Test initializing the graph via a invalid adjacency list
        adj_matrix = [
            [0,1,1,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,0]
        ]
        
        self.assertRaises(Exception, Graph(adj_matrix))
        
    def test_not_passing_arguments(self):
        # Test initializing the graph with no preexisting graph made
        graph = Graph()
        
        self.assertIsNone(graph.adj_list)
        self.assertIsNone(graph.adj_matrix)
        self.assertEqual(0, graph.order)
        self.assertEqual(0, graph.size)

if __name__ == '__main__':
    unittest.main()