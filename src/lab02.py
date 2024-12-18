# Parker Lauders
# Lab02

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.graph_Pl_v2 import GraphV2
from lib.vertex import Vertex
from lib.edge import Edge

def main():
    # Create the map
    # Cities
    Oradea = Vertex("Oradea")
    Zerind = Vertex("Zerind")
    Arad = Vertex("Arad")
    Sibiu = Vertex("Sibiu")
    Fagaras = Vertex("Fagaras")
    Rimnicu_Vilcea = Vertex("Rimnicu Vilcea")
    Timisoara = Vertex("Timisoara")
    Lugoj = Vertex("Lugoj")
    Pitesti = Vertex("Pitesti")
    Mehadia = Vertex("Mehadia")
    Drobeta = Vertex("Drobeta")
    Craiova = Vertex("Craiova")
    Bucharest = Vertex("Bucharest")
    Giurgiu = Vertex("Giurgiu")
    Urziceni = Vertex("Urziceni")
    Hirsova = Vertex("Hirsova")
    Eforie = Vertex("Eforie")
    Vaslui = Vertex("Vaslui")
    Iasi = Vertex("Iasi")
    Neamt = Vertex("Neamt")

    # Roads
    Oradea_Zerind = Edge(Oradea, Zerind, 71)
    Oradea_Sibiu = Edge(Oradea, Sibiu, 151)
    Zerind_Arad = Edge(Zerind, Arad, 75)
    Arad_Sibiu = Edge(Arad, Sibiu, 140)
    Arad_Timisoara = Edge(Arad, Timisoara, 118)
    Sibiu_Fagaras = Edge(Sibiu, Fagaras, 99)
    Sibiu_Rimnicu_Vilcea = Edge(Sibiu, Rimnicu_Vilcea, 80)
    Timisoara_Lugoj = Edge(Timisoara, Lugoj, 111)
    Lugoj_Mehadia = Edge(Lugoj, Mehadia, 70)
    Mehadia_Drobeta = Edge(Mehadia, Drobeta, 75)
    Drobeta_Craiova = Edge(Drobeta, Craiova, 120)
    Fagaras_Bucharest = Edge(Fagaras, Bucharest, 211)
    Rimnicu_Vilcea_Pitesti = Edge(Rimnicu_Vilcea, Pitesti, 97)
    Rimnicu_Vilcea_Craiova = Edge(Rimnicu_Vilcea, Craiova, 146)
    Craiova_Pitesti = Edge(Craiova, Pitesti, 138)
    Pitesti_Bucharest = Edge(Pitesti, Bucharest, 101)
    Bucharest_Urziceni = Edge(Bucharest, Urziceni, 85)
    Bucharest_Giurgiu = Edge(Bucharest, Giurgiu, 90)
    Urziceni_Vaslui = Edge(Urziceni, Vaslui, 142)
    Vaslui_Iasi = Edge(Vaslui, Iasi, 92)
    Iasi_Neamt = Edge(Iasi, Neamt, 87)
    Urziceni_Hirsova = Edge(Urziceni, Hirsova, 98)
    Hirsova_Eforie = Edge(Hirsova, Eforie, 86)

    # Adjacency list
    adjacency_list = {
        Arad: [Arad_Sibiu, Arad_Timisoara, Zerind_Arad],
        Bucharest: [Bucharest_Giurgiu, Bucharest_Urziceni, Fagaras_Bucharest, Pitesti_Bucharest],
        Craiova: [Craiova_Pitesti, Drobeta_Craiova, Rimnicu_Vilcea_Craiova],
        Drobeta: [Drobeta_Craiova, Mehadia_Drobeta],
        Eforie: [Hirsova_Eforie],
        Fagaras: [Fagaras_Bucharest, Sibiu_Fagaras],
        Giurgiu: [Bucharest_Giurgiu],
        Hirsova: [Hirsova_Eforie, Urziceni_Hirsova],
        Iasi: [Iasi_Neamt, Vaslui_Iasi],
        Lugoj: [Lugoj_Mehadia, Timisoara_Lugoj],
        Mehadia: [Lugoj_Mehadia, Mehadia_Drobeta],
        Neamt: [Iasi_Neamt],
        Oradea: [Oradea_Sibiu, Oradea_Zerind],
        Pitesti: [Craiova_Pitesti, Pitesti_Bucharest, Rimnicu_Vilcea_Pitesti],
        Rimnicu_Vilcea: [Rimnicu_Vilcea_Craiova, Rimnicu_Vilcea_Pitesti, Sibiu_Rimnicu_Vilcea],
        Sibiu: [Arad_Sibiu, Oradea_Sibiu, Sibiu_Fagaras, Sibiu_Rimnicu_Vilcea],
        Timisoara: [Arad_Timisoara, Timisoara_Lugoj],
        Urziceni: [Bucharest_Urziceni, Urziceni_Hirsova, Urziceni_Vaslui],
        Vaslui: [Urziceni_Vaslui, Vaslui_Iasi],
        Zerind: [Oradea_Zerind, Zerind_Arad]
    }

    # Initialize the graph
    graph = GraphV2(adjacency_list)
    
    print(" ----- Dijkstra's ----- ")

    # Perform Dijkstra's Algorithm
    results, paths = graph.dijkstras_algorithm(Arad, True)

    distance = results[Bucharest]

    # Extract the path betweem the vertices
    path = paths[Bucharest]
    path_string = ""
    for index in range(len(path)):
        if index == 0:
            path_string += path[index].id
        else:
            path_string += ", " + path[index].id
    
    # Task 1. Output the result for shortest path from Arad to Bucharest
    print(f"Shortest distance from {path[0].id} -> {path[-1].id}: {distance}")

    # Task 2. Output the actual path from Arad to Bucharest
    print("(start)", path_string, "(end)")
    
    
    print("\n\n ----- A* ----- ")

    # Task 3. Use a nontrivial function that is always admissible
    def heuristic(current_vertex, end_vertex):
        # Ensure that the arguments are valid (for this assignment only)
        if end_vertex is not Bucharest:
            raise Exception(f"'end_vertex' must be {Bucharest.id}")
        
        # Given straight line distances for this assignment
        heuristic_straight_line_distances = {
            Arad: 366,
            Bucharest: 0,
            Craiova: 160,
            Drobeta: 242,
            Eforie: 161,
            Fagaras: 176,
            Giurgiu: 77,
            Hirsova: 151,
            Iasi: 226,
            Lugoj: 224, 
            Mehadia: 241,
            Neamt: 234,
            Oradea: 380,
            Pitesti: 100,
            Rimnicu_Vilcea: 193,
            Sibiu: 253,
            Timisoara: 329,
            Urziceni: 80,
            Vaslui: 199,
            Zerind: 374
        }
        
        return heuristic_straight_line_distances[end_vertex]

    distance, path = graph.a_star(Arad, Bucharest, heuristic, True)

    # Extract the path betweem the vertices
    path_string = ""
    for index in range(len(path)):
        if index == 0:
            path_string += path[index].id
        else:
            path_string += ", " + path[index].id

    # Task 4. Output the result for shortest path from Arad to Bucharest
    print(f"Shortest distance from {path[0].id} -> {path[-1].id}: {distance}")

    # Task 5. Output the actual path from Arad to Bucharest
    print("(start)", path_string, "(end)")
    
    print("\n\n ----- Floyd-Warshall ----- ")
    
    # Task 7. Output the solution for Problem 0
    result_matrix = graph.floyd_warshall_algorithm(Arad)
    
    # Print resulting matrix
    print(f"D({graph.order}):\n[")
    for row in result_matrix:
        print(f"{row},")
    print("]")
        
    # Output the distance
    print(f"\nShortest distance from {Arad.id} -> {Bucharest.id}: {result_matrix[0][1]}")
    
    # Task 9. If your goal is to solve problems like Problem 0, which of these three algorithms is most appropriate? Which is least appropriate? Explain.
    
    # If the goal is to solve a problem of going from one vertex to another than the best algorithm to use would be A*.
    # This would be due to the principle of the algorithm being designed to strategically search for the end vertex.
    # This not only saves computational time since there are less steps to perform in the algorithm but it also saves
    # user time as the algorithm outputs the direct information. This prevents the user from having to post process the 
    # results in order to get the needed result at the end vertex.
    
    # The worst algorithm would have to be Floyd-Warshall, this being due to the large amount of steps that are needed.
    # This algorithm contains a lot of computational steps along with storing two matrix both n^2 sized.
    # This all around impacts the computer hardware performance. The algorithm also completes the distance
    # for all of the vertices from the starting vertex meaning that more is done than needed. Now one could argue
    # the same for Dijkstra's but with that algorithm the number of computational steps is far less than Floyd-Warshall

if __name__ == "__main__":
    main()