#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
The Iris Dataset
=========================================================
This data sets consists of 3 different types of irises'
(Setosa, Versicolour, and Virginica) petal and sepal
length, stored in a 150x4 numpy.ndarray

The rows being the samples and the columns being:
Sepal Length, Sepal Width, Petal Length	and Petal Width.

The below plot uses the first two features.
See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.
"""
print(__doc__)

#サポートベクター回帰
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from Logistic.Solver import Solver,Solver_logistic,Solver_hinge
from sklearn import datasets
import numpy as np
from Logistic.logistic_agent import Agent_harnessing_logistic,Agent_hinge_event,Agent_hinge_event_fix,Agent_hinge_event_proj
from Logistic.make_communication import Communication,Circle_communication

# n = 10
n = 20
m = 3
size = int(100/n)
# eta = 0.1
eta = 1.0
lam = 0.1
epsiron = 0.01
proj = 10
iteration = 100000
save_iteration = 1000

np.random.seed(0)

iris = datasets.load_iris()
X1 = iris.data[:100, :(m - 1)]  # we only take the first two features.
One = np.ones([len(X1), 1])
X1 = np.hstack((X1, One))
y = iris.target[:100]
y = [-1 if i == 0 else 1 for i in y]

agent_x = [X1[size * i:size * (i + 1)] for i in range(n)]
agent_y = [y[size * i:size * (i + 1)] for i in range(n)]

#集中型最適解導出
Problem = Solver_hinge(n * size , m, X1,y,lam,epsiron)
Problem.solve()
f_opt,x_opt = Problem.send_opt()

# Graph = Communication(n, 4, 0.3)
# Graph = Circle_communication(n,0.25)
Graph = Communication(n,4,0.25)
Graph.make_connected_WS_graph()
# Graph.make_connected_WS_graph()
# Graph.make_circle_graph()
Weight_matrix = Graph.send_P()
# Weight_matrix=[1]
fig = plt.figure()
# ims= [[] for i in range(n)]
f_hist = []
ims = []
Agents = []
for i in range(n):
    # Agents.append(Agent_harnessing_logistic(n, m, agent_x[i], agent_y[i], eta, Weight_matrix[i],i,lam/n))
    # Agents.append(Agent_hinge_event_fix(n, m, agent_x[i], agent_y[i], eta, Weight_matrix[i],i,lam/n,epsiron))
    Agents.append(Agent_hinge_event_proj(n, m, agent_x[i], agent_y[i], eta, Weight_matrix[i],i,lam/n,epsiron,proj))
for k in range(iteration):
    for i in range(n):
        for j in range(n):
            x_j,name = Agents[i].send(k,j)
            Agents[j].receive(x_j, name)

    for i in range(n):
        Agents[i].update(k)

    x0 = [np.linspace(0, 10) for i in range(n)]
    # x_ax = [[i for j in range(n)] for i in range(10)]
    x1 = [[]for i in range(n)]
    im = plt.plot()
    if k % int(save_iteration) == 0:
        for i in range(n):
            # x1[i] = (Agents[i].x_i[0] * x0[i] + Agents[i].x_i[2]) / (-Agents[i].x_i[1])
            x1[i] = (Agents[i].x_i[0] * x0[i] + Agents[i].x_i[2]) / (-Agents[i].x_i[1])
            im += plt.plot(x0[i], x1[i])
        ims.append(im)
    print(k)

    x_i = Agents[0].x_i
    f = 0
    for i in range(n*size):
        f += max(abs(y[i]-np.dot(X1[i],x_i)) -epsiron,0)
        # f +=  math.log(1+ math.exp(np.dot(X1[i],x_i))) - y[i]*np.dot(X1[i],x_i)
    f+= 1 / 2 * lam* np.linalg.norm(x_i) ** 2
    print(f-f_opt)
    f_hist.append(f-f_opt)

for i in range(n):
    print(Agents[i].x_i)




ax = plt.subplot(1,1,1)
p1 = plt.scatter(X1[:50,0],X1[:50,1],color='r',label='Setosa')
p2 = plt.scatter(X1[50:100,0],X1[50:100,1] ,color='b',label='Versicolour')
# plt.plot(x0,x1)
# plt.legend()
legend = ax.legend(handles=[p1,p2],labels=['Setosa','Versicolour'],loc='upper left')
# ani = [None for i in range(n)]
# ani = None
# ani = animation.ArtistAnimation(fig,None)
# for i in range(n):
ani = animation.ArtistAnimation(fig,ims,interval=30)
plt.xlim([0,8])
plt.ylim([0,6])

# plt.xlim([-100,100])
# plt.ylim([-100,100])

plt.title('Logistic regression in a part of Iris data set')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
# plt.legend()

# fig.legend()
# ani.save(filename='hoge3.gif', writer="imagemagick")
# ani.save(filename='hoge3.html', writer="imagemagick")
# ani.save(filename='hoge3.gif', writer="pillow")
# print('save完了')
plt.show()

plt.plot(f_hist)
plt.yscale('log')
plt.show()