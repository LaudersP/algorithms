import unittest
from lib.vector import Vector
from lib.matrix import Matrix
from lib.graph_pl import Graph

class TestGraph(unittest.TestCase):
    def test_adj_list_init(self):
        # Test initializing the graph via an adjacency list
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
        
    def test_adj_matrix_init(self):
        # Test initializing the graph via an adjacency matrix
        adj_matrix = Matrix(
            Vector(0, 1, 1, 1),
            Vector(1, 0, 1, 1),
            Vector(1, 1, 0, 1),
            Vector(1, 1, 1, 0)
        )
        
        graph = Graph(adj_matrix)
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the order size
        

if __name__ == '__main__':
    unittest.main()