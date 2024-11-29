# Parker Lauders
# Lab02

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.graph_Pl_v2 import GraphV2
from lib.vertex import Vertex
from lib.edge import Edge

def initialize_graph():
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
        Craiova: [Craiova_Pitesti, Drobeta_Craiova, Rimnicu_Vilcea_Pitesti],
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
        Rimnicu_Vilcea: [Craiova_Pitesti, Rimnicu_Vilcea_Pitesti, Sibiu_Rimnicu_Vilcea],
        Sibiu: [Arad_Sibiu, Oradea_Sibiu, Sibiu_Fagaras, Sibiu_Rimnicu_Vilcea],
        Timisoara: [Arad_Timisoara, Timisoara_Lugoj],
        Urziceni: [Bucharest_Urziceni, Urziceni_Hirsova, Urziceni_Vaslui],
        Vaslui: [Urziceni_Vaslui, Vaslui_Iasi],
        Zerind: [Oradea_Zerind, Zerind_Arad],
    }

    # Initialize the graph
    return GraphV2(adjacency_list)

def main():
    # Create the map
    graph = initialize_graph()

if __name__ == "__main__":
    main()