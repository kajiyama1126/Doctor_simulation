# -*- coding: utf-8 -*-
import copy

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Communication:
    def __init__(self, n, k, p, Bp, ite): # (頂点数，辺数，辺確率, グラフ連結間隔)
        self.n = n
        self.k = k
        self.p = p
        self.Bp = Bp
        self.ite = ite
        self.count = 0

    def make_connected_WS_graph(self):
        self.G = nx.connected_watts_strogatz_graph(self.n, self.k, self.p)
        #         lam = nx.laplacian_spectrum(G)
        #         print(nx.adjacency_matrix(G))
        #         print (number_of_nodes(G))
        #         (nx.degree(G))
        # print(self.G)
        self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
        self.weight_martix()

        # #グラフ描画
        # pos = nx.spring_layout(self.G)
        # # pos = None
        # labels = {}
        # for i in range(self.n):
        #     labels[i] = r"{0}".format(i + 1)
        #
        # # edge_labels = {}
        # nx.draw_networkx_nodes(self.G, pos, node_size=220, alpha=1.0, node_color="lightblue")
        # # nx.draw_networkx_nodes(self.G, pos, node_size=300, alpha=1.0, node_color="lightgrey")
        # nx.draw_networkx_edges(self.G, pos, width=1.5)
        # # nx.draw_networkx_edge_labels(self.G, pos, labels)
        # # nx.draw_networkx_edge_labels(self.G, pos, labels, edge_labels=edge_labels)
        # nx.draw_networkx_labels(self.G, pos, labels, font_size=11)
        # plt.axis('off')
        # # nx.draw_networkx_edges(G, pos, edge_color='orange', width=1.5)
        # # nx.draw_networkx_nodes(self.G, node_color='green', node_size=600)
        # # nx.draw_networkx_nodes(self.G, pos=nx.spring_layout(self.G), node_color='b')
        # plt.savefig("network.png")
        # plt.savefig("network.eps")
        # plt.show()
    #         print(self.A)

    # def make_graph(self,number):
    #     graph = [nx.dense_gnm_random_graph(self.n,self.m) for i in range(number)]

    def make_B_connected_graph(self):
        #self.G = nx.connected_watts_strogatz_graph(self.n, self.k, self.p)
        #         lam = nx.laplacian_spectrum(G)
        #         print(nx.adjacency_matrix(G))
        #         print (number_of_nodes(G))
        #         (nx.degree(G))
        # print(self.G)
        #self.B = B

        #print(self.Bp)
        if self.Bp == 1:
            #self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
            self.A = [[0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 1, 1, 1],
                      [1, 0, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0]]  # 隣接行列
            self.weight_martix()

        elif self.Bp == 2:
            if self.ite % self.Bp == 0:
                self.A = [[0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0]]
            else:
                self.A = [[0, 0, 1, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 1, 1, 1],
                          [0, 0, 1, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0]]
            self.weight_martix()

        elif self.Bp == 3:
            if self.ite % self.Bp == 0:
                self.A = [[0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            elif self.ite % self.Bp == 1:
                self.A = [[0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]]
            else:
                self.A = [[0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 1, 1, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0]]
            self.weight_martix()

        elif self.Bp == 4:
            if self.ite % 4 == 0:
                self.A = [[0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]]
            elif self.ite % 4 == 1:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            elif self.ite % 4 == 2:
                self.A = [[0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0]]
            else:
                self.A = [[0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            self.weight_martix()

        elif self.Bp == 5:
            if self.ite % self.Bp == 0:
                self.A = [[0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % self.Bp == 1:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            elif self.ite % self.Bp == 2:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % self.Bp == 3:
                self.A = [[0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0]]
            else:
                self.A = [[0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]]
            self.weight_martix()

        elif self.Bp == 6:
            if self.ite % self.Bp == 0:
                self.A = [[0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % self.Bp == 1:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            elif self.ite % self.Bp == 2:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % self.Bp == 3:
                self.A = [[0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % self.Bp == 4:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            else:
                self.A = [[0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]]
            self.weight_martix()

        elif self.Bp == 8:
            if self.ite % 8 == 0:
                self.A = [[0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % 8 == 1:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            elif self.ite % 8 == 2:
                self.A = [[0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0]]
            elif self.ite % 8 == 3:
                self.A = [[0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % 8 == 4:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            elif self.ite % 8 == 5:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]
            elif self.ite % 8 == 6:
                self.A = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
            elif self.ite % 8 == 7:
                self.A = [[0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0]]
            else:
                self.A = [[0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]]
            self.weight_martix()

    def weight_martix(self):
        a = np.zeros(self.n)
        d = np.dot(self.A, np.ones(6))
        for i in range(self.n):
            #a[i] = copy.copy(1.0 / (max(d) + 0.5))
            #a[i] = copy.copy(1.0 / (max(d) + 1.0))
            a[i] = copy.copy(1.0 / (max(d) + 10))
            #a[i] = copy.copy(1.0 / (max(d) + 20))
            #a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))

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


if __name__ == '__main__':
    graph = Communication(10,2,0.3, 1)
    graph.make_connected_WS_graph()
    print(graph.P)