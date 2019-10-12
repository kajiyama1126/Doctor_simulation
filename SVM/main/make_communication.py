# -*- coding: utf-8 -*-
import copy

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Communication:  # (頂点数，辺数，辺確率)
    def __init__(self, n, k, p):
        self.n = n
        self.k = k
        self.p = p
        self.count = 0

    def make_connected_WS_graph(self):
#         self.G = nx.connected_watts_strogatz_graph(self.n, self.k, self.p)
# #        nx.draw(self.G)
#         #         lam = nx.laplacian_spectrum(G)
#         #         print(nx.adjacency_matrix(G))
#         #         print (number_of_nodes(G))
#         #         (nx.degree(G))
# #        print(self.G)
#         Adj = nx.to_numpy_matrix(self.G)
#         np.savetxt('Adj.txt', Adj, fmt='%d', delimiter=",")

        Adj = np.loadtxt('Adj.txt', delimiter=",")
        Adj = Adj.T
        #self.G = nx.from_numpy_matrix(Adj, create_using=nx.MultiDiGraph())
        self.G = nx.from_numpy_matrix(Adj)
        #self.G = nx.DiGraph(self.G)
        #pos = nx.spring_layout(self.G)
        pos = nx.circular_layout(self.G)

        self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
        self.weight_martix()

        plt.figure()
        labels = {}
        for i in range(self.n):
            labels[i] = r"{0}".format(i+1)
        #print(labels)
        #nx.draw_networkx_nodes(self.G, pos, labels, node_size=30, alpha=1.0, node_color="blue")
        nx.draw_networkx_nodes(self.G, pos, node_size=220, alpha=1.0, node_color="lightblue")
        nx.draw_networkx_edges(self.G, pos, width=0.6, arrowsize=14)
        # nx.draw_networkx_edges(self.G, pos, width=0.6, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(self.G, pos, labels, font_size=11)
        plt.axis('off')
        plt.savefig("network.png")
        plt.savefig("network.eps")
        # plt.show()

    #         print(self.A)

    # def make_graph(self,number):
    #     graph = [nx.dense_gnm_random_graph(self.n,self.m) for i in range(number)]

    def weight_martix(self):
        a = np.zeros(self.n)
        #print(self.G.in_degree)
        max_degree = 0
        for i in range(self.n):
            #if(self.G.in_degree[i] > max_degree):
            if (nx.degree(self.G)[i] > max_degree):
                max_degree = nx.degree(self.G)[i]
        #print(max_degree)
        #exit(1)

        for i in range(self.n):
            # a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 5.0))
            #a[i] = copy.copy(0.2 / nx.degree(self.G)[i])
            #a[i] = copy.copy(0.8 / nx.degree(self.G)[i])
            #a[i] = copy.copy(0.1 / nx.degree(self.G)[i]) #<--
            a[i] = copy.copy(0.6 / max_degree)
            #a[i] = copy.copy(0.6 / max_degree)
            
        self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
        for i in range(self.n):
            for j in range(i, self.n):
                if i != j and self.A[i][j] == 1:
                    a_ij = min(a[i], a[j])
                    self.P[i][j] = copy.copy(a_ij)
                    self.P[j][i] = copy.copy(a_ij)


                #         print(self.P)
        for i in range(self.n):
            sum = 0.0
            for j in range(self.n):
                sum += self.P[i][j]
            self.P[i][i] = 1.0 - sum

    def send_P(self):
        return self.P


class Circle_communication(object):
    def __init__(self, n, w):
        self.n = n
        self.w = w
    def make_circle_graph(self):
        self.G = nx.cycle_graph(self.n)
        self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
        self.weight_martix()

    def weight_martix(self):
        a = np.zeros(self.n)
        for i in range(self.n):
            # a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))
            a[i]  = self.w
        self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
        for i in range(self.n):
            for j in range(i, self.n):
                if i != j and self.A[i][j] == 1:
                    a_ij = min(a[i], a[j])
                    self.P[i][j] = copy.copy(a_ij)
                    self.P[j][i] = copy.copy(a_ij)


                #         print(self.P)
        for i in range(self.n):
            sum = 0.0
            for j in range(self.n):
                sum += self.P[i][j]
            self.P[i][i] = 1.0 - sum

    def send_P(self):
        return self.P

if __name__ == '__main__':
    graph = Communication(10,2,0.3)
    graph.make_connected_WS_graph()
    print(graph.P)