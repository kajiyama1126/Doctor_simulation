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

    def weight_martix(self):
        a = np.zeros(self.n)
        for i in range(self.n):
            a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))

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
    graph = Communication(10,2,0.3)
    graph.make_connected_WS_graph()
    print(graph.P)