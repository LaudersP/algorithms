# Parker Lauders
# Assignment 4

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lib.vertex import Vertex
from lib.edge import Edge
from lib.graph_Pl_v2 import GraphV2

def do_assignment_core(graph):
    # Matrix printer helper function
    def print_matrix(matrix, string):
        # Print label
        print(f"\n{string}:")

        # Iterate through each row of the matrix
        for row in matrix:
            # Print the row
            print(row)

    # Get the Laplacian of the graph
    laplacian_matrix = graph.get_laplacian_matrix()

    # Output the matrix
    print_matrix(laplacian_matrix, "Laplacian Matrix")

    # Get the Eigenvalues of the graph
    evalues, evect = np.linalg.eig(laplacian_matrix)

    # Sort the eigenvalues
    evalues = np.sort(evalues)

    # Output the values
    print(f"\nEigen Values: {evalues}")

    # Output the matrix
    print_matrix(evect, "Eigen Vectors")

    # Get the spanning trees using Kirchoffs matrix tree theorem
    # Get the reduced laplacian matrix
    laplacian_reduced_matrix = np.delete(laplacian_matrix, 0, 0)
    laplacian_reduced_matrix = np.delete(laplacian_reduced_matrix, 0, 1)

    # Calculate the number of spanning trees
    spanning_trees = int(round(np.linalg.det(laplacian_reduced_matrix)))
    print(f"\nSpanning Trees: {spanning_trees}")

    # Determine if the graph is connected
    ezeros = np.sum(np.isclose(evalues, 0))
    if ezeros and spanning_trees > 0:
        print("The graph is connected!")
    else:
        print("The graph is not connected!")

def main():
    graph_input = int(input("Which graph would you like to run? "))
    adj_list = None

    # Act on the appropriate graph number
    if graph_input == 1:     # Kn graph
        n = int(input("Kn: "))

        # Create a vertices list
        vertices = []
        for i in range(1, n + 1):
            vertices.append(Vertex(i))

        # Setup the adjacency list
        adj_list = {}
        for vertex in vertices:
            adj_list[vertex] = []

        # Iterate through the vertex list
        for i in range(n):
            # Iterate through the other vertices excluding the vertex i represents
            for j in range(i + 1, n):
                # Create the edge
                edge = Edge(vertices[i], vertices[j])

                # Add the edge to the adj_list
                adj_list[vertices[i]].append(edge)
                adj_list[vertices[j]].append(edge)

    elif graph_input == 2:
        pass
    elif graph_input == 3:
        pass
    elif graph_input == 4:
        # Vetices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        F = Vertex('F')
        G = Vertex('G')
        H = Vertex('H')
        I = Vertex('I')
        J = Vertex('J')

        # Edges
        AB = Edge(A, B)
        AE = Edge(A, E)
        AF = Edge(A, F)
        BC = Edge(B, C)
        BI = Edge(B, I)
        CD = Edge(C, D)
        CG = Edge(C, G)
        DE = Edge(D, E)
        DJ = Edge(D, J)
        EH = Edge(E, H)
        FG = Edge(F, G)
        FJ = Edge(F, J)
        GH = Edge(G, H)
        HI = Edge(H, I)
        IJ = Edge(I, J)

        # Adjacency list
        adj_list = {
            A: [AB, AE, AF],
            B: [AB, BC, BI],
            C: [BC, CD, CG],
            D: [CD, DE, DJ],
            E: [AE, DE, EH],
            F: [AF, FG, FJ],
            G: [CG, FG, GH],
            H: [EH, GH, HI],
            I: [BI, HI, IJ],
            J: [FJ, DJ, IJ]
        }

    elif graph_input == 5:
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
        J = Vertex('J')
        K = Vertex('K')
        L = Vertex('L')
        M = Vertex('M')
        N = Vertex('N')
        O = Vertex('O')
        P = Vertex('P')
        Q = Vertex('Q')
        R = Vertex('R')
        S = Vertex('S')
        T = Vertex('T')
        
        # Edges
        AB = Edge(A, B)
        AF = Edge(A, F)
        AE = Edge(A, E)
        BC = Edge(B, C)
        BG = Edge(B, G)
        CD = Edge(C, D)
        CH = Edge(C, H)
        DE = Edge(D, E)
        DI = Edge(D, I)
        EJ = Edge(E, J)
        FK = Edge(F, K)
        FO = Edge(F, O)
        GK = Edge(G, K)
        GL = Edge(G, L)
        HL = Edge(H, L)
        HM = Edge(H, M)
        IM = Edge(I, M)
        IN = Edge(I, N)
        JN = Edge(J, N)
        JO = Edge(J, O)
        KP = Edge(K, P)
        LQ = Edge(L, Q)
        MR = Edge(M, R)
        NS = Edge(N, S)
        OT = Edge(O, T)
        PQ = Edge(P, Q)
        PT = Edge(P, T)
        QR = Edge(Q, R)
        RS = Edge(R, S)
        ST = Edge(S, T)
        
        # Adjacency list
        adj_list = {
            A: [AB, AF, AE],
            B: [AB, BC, BG],
            C: [BC, CD, CH],
            D: [CD, DE, DI],
            E: [AE, DE, EJ],
            F: [AF, FK, FO],
            G: [BG, GK, GL],
            H: [CH, HL, HM],
            I: [DI, IM, IN],
            J: [EJ, JN, JO],
            K: [FK, GK, KP],
            L: [GL, HL, LQ],
            M: [HM, IM, MR],
            N: [IN, JN, NS],
            O: [FO, JO, OT],
            P: [KP, PQ, PT],
            Q: [LQ, PQ, QR],
            R: [MR, QR, RS],
            S: [NS, RS, ST],
            T: [OT, PT, ST]
        }
        
    elif graph_input == 6:
        # Verices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')

        # Edges
        AB = Edge(A, B)
        AD = Edge(A, D)
        BC = Edge(B, C)
        BD = Edge(B, D)
        CD = Edge(C, D)

        # Adjacency list
        adj_list = {
            A: [AB, AD],
            B: [AB, BC, BD],
            C: [BC, CD],
            D: [AD, BD, CD]
        }

    elif graph_input == 7:
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        
        E = Vertex('E')
        F = Vertex('F')
        G = Vertex('G')
        H = Vertex('H')
        
        # Edges
        AB = Edge(A, B)
        AD = Edge(A, D)
        AE = Edge(A, E)
        BC = Edge(B, C)
        BD = Edge(B, D)
        CD = Edge(C, D)
        CG = Edge(C, G)
        EF = Edge(E, F)
        EH = Edge(E, H)
        FG = Edge(F, G)
        FH = Edge(F, H)
        GH = Edge(G, H)

        # Adjacency list
        adj_list = {
            A: [AB, AD, AE],
            B: [AB, BC, BD],
            C: [BC, CD, CG],
            D: [AD, BD, CD],
            E: [AE, EF, EH],
            F: [EF, FG, FH],
            G: [CG, FG, GH],
            H: [EH, FH, GH]
        }
        
    elif graph_input == 8:
        # Vertices
        A = Vertex('A')
        B = Vertex('B')
        C = Vertex('C')
        D = Vertex('D')
        E = Vertex('E')
        
        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        BD = Edge(B, D)
        CE = Edge(C, E)
        
        # Adjacency list
        adj_list = {
            A: [AB],
            B: [AB, BC, BD],
            C: [BC, CE],
            D: [BD],
            E: [CE]
        }
        
    elif graph_input == 9:
        # Vertices
        F = Vertex('F')
        G = Vertex('G')
        H = Vertex('H')
        I = Vertex('I')
        J = Vertex('J')
        
        # Edges
        FG = Edge(F, G)
        FH = Edge(F, H)
        FI = Edge(F, I)
        GI = Edge(G, I)
        IH = Edge(I, H)
        GJ = Edge(G, J)
        
        # Adjacency list
        adj_list = {
            F: [FG, FH, FI],
            G: [FG, GJ, GI],
            H: [IH, FH],
            I: [FI, GI, IH],
            J: [GJ]
        }
        
    elif graph_input == 10:
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
        J = Vertex('J')

        # Edges
        AB = Edge(A, B)
        BC = Edge(B, C)
        BD = Edge(B, D)
        CE = Edge(C, E)
        FG = Edge(F, G)
        FH = Edge(F, H)
        FI = Edge(F, I)
        GI = Edge(G, I)
        IH = Edge(I, H)
        GJ = Edge(G, J)

        # Adjacency list
        adj_list = {
            A: [AB],
            B: [AB, BC, BD],
            C: [BC, CE],
            D: [BD],
            E: [CE],
            F: [FG, FH, FI],
            G: [FG, GJ, GI],
            H: [IH, FH],
            I: [FI, GI, IH],
            J: [GJ]
        }

    else:
        print("Invalid graph number, must be 1-10!")
        return
    
    # Create the graph
    graph = GraphV2(adj_list)

    # Perform the assignment
    do_assignment_core(graph)

if __name__ == "__main__":
    main()