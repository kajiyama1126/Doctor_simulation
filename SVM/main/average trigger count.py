# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:40 2019

@author: ago fumiya
"""

from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
from LogisticRegression.main.logistic_agent import Agent_harnessing_logistic
from LogisticRegression.main.make_communication import Communication, Circle_communication

iteration = 10
all_ave_tri_count = 0
for test in range(iteration):
    print(str(test+1) + '回目')
    class Program(object):
        n = 10
        m = 3
        size = int(100/n)
        s = 30
        R = 50
        th_pa = 1
        
        iteration = 50000
        sum_trigger_count = 0
        Agents = []
        
        iris = datasets.load_iris()
        X1 = iris.data[:100, :(m - 1)]  # we only take the first two features.
        One = np.ones([len(X1), 1])
        X1 = np.hstack((X1, One))
        y = iris.target[:100]
#        X2, y1 = shuffle(X1, y)#random select
        agent_x = [X1[size * i:size * (i + 1)] for i in range(n)]
        agent_y = [y[size * i:size * (i + 1)] for i in range(n)]
        
        Graph = Communication(n, 4, 0.3)
        g = Graph.make_connected_WS_graph()
        Weight_matrix = Graph.send_P()
        
        for i in range(n):
            Agents.append(Agent_harnessing_logistic(n, m, agent_x[i], agent_y[i], s, R, Weight_matrix[i], i, th_pa))
        for k in range(iteration):
#            print("iteration = %d" % k)
            for i in range(n):
                for j in range(n):
                    if j != i:
                        x_j,name = Agents[i].send(j)
                        Agents[j].receive(x_j, name)
        
            for i in range(n):
                Agents[i].update(k)
        
        for i in range(n):
            sum_trigger_count += Agents[i].trigger_count / Agents[i].neighbor_agents
        ave_tri_count = sum_trigger_count / n
        print(ave_tri_count)
       
    program = Program()
    all_ave_tri_count += program.ave_tri_count
#    print(program.ave_tri_count)
print(all_ave_tri_count / iteration)

