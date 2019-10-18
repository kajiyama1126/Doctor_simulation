#!/usr/bin/python
# -*- coding: utf-8 -*-

print(__doc__)

import os
#import configparser
#import datetime
import sys
#import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import cvxpy as cvx
#from LogisticRegression.main import svmcmpl
from progressbar import ProgressBar
#from matplotlib.colors import ListedColormap
from SVM.main.agent import Agent_subgrad, Agent_harnessing, Agent_harnessing_EventTriggered_trigger
#from LogisticRegression.main.logistic_agent import Agent_harnessing_logistic
from SVM.main.make_communication import Communication, Circle_communication
from SVM.main import mlbench as ml
from SVM.main.plot_funcs import plot_decision_regions, plot_decision_regions2
from SVM.main.Gaussian_kernel import GuassianKernelSVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC

# ---------------------------------------------------------------------------#
# Parameters

# Number of agents
n = 10

# Number of data
nd = 200

# Data size of each agent
M = int(nd / n)

# Number of dimensions of the decision variable
m = nd + 1

# Step-size for the algorithm
#s = 50
#s = 0.2
#s = 0.01
s = 1

# Regularization parameter for SVM
Ccvx = 10
#Ccvx = 8
C = Ccvx * np.ones(n)

# parameter of the Gaussian kernel
gam = 200

# Randomization seed
np.random.seed(1)

# Number of iterations
iteration = 50000

# Interval for figure plot
fig_interval = 5000

#sum_trigger_count = 0

# Data folder
os.chdir('data')

# # Plot region
x1_min = -1.5
x1_max = 1.5
x2_min = -1.5
x2_max = 1.5

#
resolution = 0.05

# ---------------------------------------------------------------------------#

#X0, y = ml.spirals(nd, cycles=1, sd=0.15)
X0, y = ml.spirals(nd, cycles=1, sd=0.1)
Xy = np.hstack((X0, y.reshape(-1, 1)))
#Xy = sorted(Xy, key=lambda x: x[2]) # sort data according to a label (0-->1)
Xy = np.array(Xy)
#print(Xy)
X0 = Xy[:, 0:2]
y0 = Xy[:, 2]
y0 = [-1 if i == 0 else i for i in y0]

# # Iris dataset
# iris = datasets.load_iris()
# X0 = iris.data[:100, :(m - 1)]  # we only take the first two features.
# y0 = iris.target[:100]
# y0 = [-1 if i == 0 else i for i in y0]  # Change the label of Setosa from 0 to -1

# xt1 = np.random.normal(0, 1, (int(nd/2), 1))
# yt1 = np.random.normal(0, 1, (int(nd/2), 1))
#
# xt2 = np.random.normal(4, 1, (int(nd/2), 1))
# yt2 = np.random.normal(4, 1, (int(nd/2), 1))
#
# xt = np.vstack((xt1, xt2))
# yt = np.vstack((yt1, yt2))
#
# X0 = np.hstack((xt, yt))
# y0 = np.vstack((np.ones([len(xt1), 1]), -1 * np.ones([len(xt2), 1])))

# ---------------------------------------------------------------------------#
# Data for agents

#(X1, y1) = (X0, y0)
X1, y1 = shuffle(X0, y0)  # random select

# X2 = np.hstack((X1, np.ones([len(X1), 1])))
X2=X1
agent_x = [X2[M * i:M * (i + 1)] for i in range(n)]
agent_y = [y1[M * i:M * (i + 1)] for i in range(n)]

# ---------------------------------------------------------------------------#
# Centralized SVM (SVC from sklearn)

# svm = SVC(kernel="rbf", gamma=gam, C=Ccvx)
# svm.fit(X1, y1)
# plot_decision_regions(X1, y1, svm, 'opt', None, resolution, x1_min, x1_max, x2_min, x2_max)

# Centralized SVM (CVXPY)
xc = cvx.Variable(M)
fopt = 0
ftmp_i = [0 for i in range(n)]
ftmp = 0
x_opt = np.zeros(M)

# Kc = np.zeros((m, m))
Kc = np.zeros((n,M,M))
# for i in range(n):
#     for p in range(m-1):
#         for q in range(m-1):
#             Kc[n][p][q] = np.exp(-gam * np.linalg.norm(X2[p] - X2[q]) ** 2)
for i in range(n):
    for p in range(M):
        for q in range(M):
            Kc[i][p][q] = np.exp(-gam * np.linalg.norm(X2[p] - X2[q]) ** 2)
for i in range(n):
    for p in range(M):
    # print(np.size(y1[p]))
    # print(np.size(Kc[p]))
        ftmp_i[i] += Ccvx / M* cvx.pos(1 - agent_y[i][p] * Kc[i][p]*xc)
    ftmp_i[i]+= 1/2*cvx.quad_form(xc, Kc[i])

    fopt += ftmp_i[i]

# fopt = Ccvx * ftmp / M + cvx.quad_form(xc, Kc)
#fopt = C * sum(cvx.pos(1 - y1 * xc.T * X2)) / csize + cvx.sum_squares(xc[0:2]) / 2

obj = cvx.Minimize(fopt)
#constrains = [cvx.norm(xc, 2) <= R]
prob = cvx.Problem(obj)
# prob.solve(solver=cvx.ECOS, verbose=True, abstol=1e-20, reltol=1e-20, feastol=1e-20, max_iters=25000)
prob.solve(solver=cvx.SCS,verbose=True,max_iters=2000,eps=1e-12)
x_opt = xc.value
# for i in range(m):
#     x_opt[i] = xc.value
fopt = prob.value
# print(x_opt)
print('xopt_cvx: ', x_opt)
print('fopt_cvx: ', fopt)

for i in range(n):
    plot_decision_regions2(X0, y0, GuassianKernelSVC(), -2, x_opt, agent_x[i], gam, 0, M, resolution, x1_min, x1_max, x2_min, x2_max)
plt.show()
# ---------------------------------------------------------------------------#
# Communication Graph

Graph = Communication(n, 4, 0.3)
g = Graph.make_connected_WS_graph()
graph_type = "WS"
Weight_matrix = Graph.send_P()

# ---------------------------------------------------------------------------#
K = np.zeros((n, m-1, m-1))
for i in range(n):
    u = X1[i * M: (i+1) * M]
    y = y1[i * M: (i+1) * M]
    for p in range(M):
        for q in range(M):
            K[i][i * M + p][i * M + q] = np.exp(-gam * np.linalg.norm(u[p] - u[q]) ** 2)

# ---------------------------------------------------------------------------#
bK = np.zeros((n, m-1, m))
tK = np.zeros((n, m, m))

for i in range(n):
    for p in range(m-1):
        bK[i][p][m - 1] = 1
        for q in range(m-1):
            bK[i][p][q] = K[i][p][q]
            tK[i][p][q] = K[i][p][q]

# np.savetxt('bKA.txt', bKA[0])
# np.savetxt('tKA.txt', tKA[0])

# ---------------------------------------------------------------------------#
# Initialization

Agents = []
x_update = np.zeros(m)
prog = ProgressBar(max_value=iteration)

for i in range(n):
    #Agents.append(Agent_harnessing(n, m, agent_x[i], agent_y[i], s, Weight_matrix[i], i, C[i]))
    #Agents.append(Agent_subgrad(n, m, agent_x[i], agent_y[i], s, Weight_matrix[i], i, C[i]))
    Agents.append(Agent_subgrad(n, m, agent_x[i], agent_y[i], s, Weight_matrix[i], i, bK[i], tK[i], C[i]))


plt.figure()
plot_decision_regions2(X0, y0, GuassianKernelSVC(), -1, x_update, X1, gam, 0, m - 1, resolution, x1_min, x1_max,
                               x2_min, x2_max)
plt.show()

# ---------------------------------------------------------------------------#
# Algorithm

f_er_base = np.zeros(n)
f_er = 0
normalized_f_er = 0
f_list = []

for k in range(iteration):
    prog.update(k)

    for i in range(n):
        for j in range(n):
            if j != i:
                state, name = Agents[i].send(j)
                Agents[j].receive(state, name)

    for i in range(n):
        Agents[i].update(k)

    # Evaluate the costs of agents
    f = 0
    for i in range(n):
        x_update = Agents[i].x_i
        ftmp = 0
        for j in range(n):
            for p in range(M):
                ftmp = ftmp + max(1 - agent_y[j][p] * np.dot(bK[j][p], x_update), 0)
            f = f + C[j] * ftmp / M + np.dot(np.dot(tK[j], x_update), x_update) / 2

    if k == 0:
        f_er_base = f - n * fopt
        print('f_er_base: ', f_er_base)

    f_er = f - n * fopt
    normalized_f_er = f_er / f_er_base
    f_list.append(normalized_f_er)
    # f_list.append(f)

    if (k+1) % fig_interval == 0:
        plt.figure()
        #plt.plot(L_er_list)
        plt.plot(f_list)
        plt.xlabel("Iteration $k$")
        plt.ylabel('Normalized Error')
        #plt.ylabel('Cost Function')
        plt.yscale("log")
        plt.xlim([0, k+2])
        plt.xticks(np.arange(0, k + 2, fig_interval))
        plt.grid(which='major', color='gray', linestyle=':')
        plt.grid(which='minor', color='gray', linestyle=':')
        plt.tight_layout()
        # plt.savefig("cost_gaussian_svm_k={}".format(k + 1) + ".eps")
        # plt.savefig("cost_gaussian_svm_k={}".format(k+1) + ".png")
        plt.savefig("normalized_error_gaussian_svm_k={}".format(k+1) + ".eps")
        plt.savefig("normalized_error_gaussian_svm_k={}".format(k+1) + ".png")

        plt.figure()
        #i = n - 1
        #plot_decision_regions2(X0, y0, GuassianKernelSVC(), k, x_update, X1[i * M: (i+1) * M], gam, i, M, resolution, x1_min, x1_max, x2_min, x2_max)
        plot_decision_regions2(X0, y0, GuassianKernelSVC(), k, x_update, X1, gam, 0, m - 1, resolution, x1_min, x1_max,
                               x2_min, x2_max)

        # plt.show()
        # plt.figure()
        # for i in range(n):
        #     y1[i] = (Agents[i].x_i[0] * x0[i] + Agents[i].x_i[2]) / (-Agents[i].x_i[1])
        #     plt.plot(x0[i], y1[i])
        # x01 = (x_opt[0] * x00 + x_opt[2]) / (-x_opt[1])
        # plt.plot(x00, x01, "k--", lw=2, label="optimal")
        # plt.scatter(X0[:int(nd / 2), 0], X0[:int(nd / 2), 1], color='b', marker='o', label='Data 1')
        # plt.scatter(X0[int(nd / 2):nd, 0], X0[int(nd / 2):nd, 1], color='r', marker='x', label='Data -1')
        # plt.xlim([xmin, xmax])
        # plt.ylim([ymin, ymax])
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()
        # plt.savefig("result_gaussian_svm_k={}".format(k + 1) + ".eps")
        # plt.savefig("result_gaussian_svm_k={}".format(k + 1) + ".png")

#print('f_fin: ', f)
