# -*- coding: utf-8 -*-
import time

import matplotlib.pylab as plt
import numpy as np
# from numba import jit, f8, i8
# import seaborn
from moment_subgrad.jikken1.agent import Agent, Agent_moment_CDC2017, Agent_L2, Agent_moment_CDC2017_L2, Agent_Dist, Agent_moment_CDC2017_Dist,Agent_moment_CDC2017_paper
from moment_subgrad.jikken1.make_communication import Communication
from moment_subgrad.jikken1.problem import Lasso_problem, Ridge_problem, Dist_problem,New_Lasso_problem


# @jit(f8(f8[:], f8[:], i8, i8, f8))
def L1_optimal_value(x_i, p, n, m, lamb):
    """\
    :param x_i: float
    :param p:float
    :param n:int
    :param m:int
    :param lamb:float
    :return:float
    """
    p_all = np.reshape(p, (-1,))
    c = np.ones(n)
    d = np.reshape(c, (n, -1))
    A = np.kron(d, np.identity(m))
    tmp = np.dot(A, x_i) - p_all
    L1 = lamb * n * np.linalg.norm(x_i, 1)
    f_opt = 1 / 2 * (np.linalg.norm(tmp)) ** 2 + L1
    return f_opt


# @jit(f8(f8[:], f8[:], i8, i8, f8))
def L2_optimal_value(x_i, p, n, m, lamb):
    """
    :param x_i: float
    :param p:float
    :param n:int
    :param m:int
    :param lamb:float
    :return:float
    """
    p_all = np.reshape(p, (-1,))
    c = np.ones(n)
    d = np.reshape(c, (n, -1))
    A = np.kron(d, np.identity(m))
    tmp = np.dot(A, x_i) - p_all
    L2 = lamb * n * np.linalg.norm(x_i, 2) ** 2
    f_opt = 1 / 2 * (np.linalg.norm(tmp)) ** 2 + L2
    return f_opt


# @jit(f8(f8[:], f8[:], i8, i8, f8))
def Dist_optimal_value(x_i, p, n, m, lamb):
    """
    :param x_i: float
    :param p:float
    :param n:int
    :param m:int
    :param lamb:float
    :return:float
    """
    f_opt = 0
    for i in range(n):
        f_opt += np.linalg.norm(x_i - p[i])
    # p_all = np.reshape(p, (-1,))
    # c = np.ones(n)
    # d = np.reshape(c, (n, -1))
    # A = np.kron(d, np.identity(m))
    # tmp = np.dot(A, x_i) - p_all
    # f_opt = np.linalg.norm(tmp)
    return f_opt


def optimal_L1(n, m, lamb,R):
    """
    :param n: int
    :param m: int
    :param lamb: float
    :return: float,float,float
    """
    p = [np.random.randn(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    p_num = np.array(p)
    # np.reshape(p)
    prob = Lasso_problem(n, m, p_num, lamb,R)
    prob.solve()
    x_opt = np.array(prob.x.value)  # 最適解
    x_opt = np.reshape(x_opt, (-1,))  # reshape
    f_opt = prob.send_f_opt()
    return p, x_opt, f_opt

def new_optimal_L1(n,m,lamb,R):
    p = [np.random.randn(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    A = [np.random.randn(m,m) + np.identity(m) for i in range(n)]
    p_num = np.array(p)
    A_num = np.array(A)
    # np.reshape(p)
    prob = New_Lasso_problem(n, m,p_num, lamb,R,A_num)
    prob.solve()
    x_opt = np.array(prob.x.value)  # 最適解
    x_opt = np.reshape(x_opt, (-1,))  # reshape
    f_opt = prob.send_f_opt()
    return A,p, x_opt, f_opt

def optimal_L2(n, m, lamb,R):
    """
    :param n: int
    :param m: int
    :param lamb: float
    :return: float,float,float
    """
    p = [np.random.randn(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    p_num = np.array(p)
    # np.reshape(p)
    prob = Ridge_problem(n, m, p_num, lamb,R)
    prob.solve()
    x_opt = np.array(prob.x.value)  # 最適解
    x_opt = np.reshape(x_opt, (-1,))  # reshape
    f_opt = prob.send_f_opt()
    return p, x_opt, f_opt


def optimal_Dist(n, m, lamb,R):
    """
    :param n: int
    :param m: int
    :param lamb: float
    :return: float,float,float
    """
    p = [np.random.randn(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    p_num = np.array(p)
    # np.reshape(p)
    prob = Dist_problem(n, m, p_num, lamb,R)
    prob.solve()
    x_opt = np.array(prob.x.value)  # 最適解
    x_opt = np.reshape(x_opt, (-1,))  # reshape
    f_opt = prob.send_f_opt()
    return p, x_opt, f_opt


# @jit(f8[:](i8, i8, f8, f8, f8, i8, f8[:, :, :], f8, i8))
def iteration_L1(n, m, p, step, lamb,R, test, P_history, f_opt, pattern):
    Agents = []
    s = step[pattern]
    for i in range(n):
        if pattern % 2 == 0:
            Agents.append(Agent(n, m, p[i], s, lamb, name=i, weight=None,R=R))
        elif pattern % 2 == 1:
            Agents.append(Agent_moment_CDC2017(n, m, p[i], s, lamb, name=i, weight=None,R=R))

    f_error_history = []
    for k in range(test):
        # グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i, name = Agents[i].send()
                Agents[j].receive(x_i, name)

        for i in range(n):
            Agents[i].update(k)

        # x_ave = 0
        # for i in range(n):
        #     x_ave += 1.0/n * Agents[i].x_i
        f_value = []
        for i in range(n):
            x_i = Agents[i].x_i
            estimate_value = L1_optimal_value(x_i, p, n, m, lamb)
            f_value.append(estimate_value)

        # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
        f_error_history.append(np.max(f_value) - f_opt)

    return f_error_history

def iteration_L1_paper(n, m, p, step, lamb,R, test, P_history, f_opt, pattern):
    Agents = []
    s = step[pattern]
    for i in range(n):
        if pattern % 2 == 0:
            Agents.append(Agent(n, m, p[i], s, lamb, name=i, weight=None,R=R))
        elif pattern % 2 == 1:
            Agents.append(Agent_moment_CDC2017_paper(n, m, p[i], s, lamb, name=i, weight=None,R=R))

    f_error_history = []
    for k in range(test):
        # グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i, name = Agents[i].send()
                Agents[j].receive(x_i, name)

        for i in range(n):
            Agents[i].update(k)

        # x_ave = 0
        # for i in range(n):
        #     x_ave += 1.0/n * Agents[i].x_i
        f_value = []
        for i in range(n):
            x_i = Agents[i].x_i
            estimate_value = L1_optimal_value(x_i, p, n, m, lamb)
            f_value.append(estimate_value)

        # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
        f_error_history.append(np.max(f_value) - f_opt)

    return f_error_history

# @jit(f8[:](i8, i8, f8, f8, f8, i8, f8[:, :, :], f8, i8))
def iteration_L2(n, m, p, step, lamb,R, test, P_history, f_opt, pattern):
    Agents = []
    s = step[pattern]
    for i in range(n):
        if pattern % 2 == 0:
            Agents.append(Agent_L2(n, m, p[i], s, lamb, name=i, weight=None,R=R))
        elif pattern % 2 == 1:
            Agents.append(Agent_moment_CDC2017_L2(n, m, p[i], s, lamb, name=i, weight=None,R=R))

    f_error_history = []
    for k in range(test):
        # グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i, name = Agents[i].send()
                Agents[j].receive(x_i, name)

        for i in range(n):
            Agents[i].update(k)

        # x_ave = 0
        # for i in range(n):
        #     x_ave += 1.0/n * Agents[i].x_i
        f_value = []
        for i in range(n):
            x_i = Agents[i].x_i
            estimate_value = L2_optimal_value(x_i, p, n, m, lamb)
            f_value.append(estimate_value)

        # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
        f_error_history.append(np.max(f_value) - f_opt)

    return f_error_history


# @jit(f8[:](i8, i8, f8, f8, f8, i8, f8[:, :, :], f8, i8))
def iteration_Dist(n, m, p, step, lamb,R, test, P_history, f_opt, pattern):
    Agents = []
    s = step[pattern]
    for i in range(n):
        if pattern == 0:
            Agents.append(Agent_Dist(n, m, p[i], s, lamb, name=i, weight=None,R=R))
        elif pattern == 1:
            Agents.append(Agent_moment_CDC2017_Dist(n, m, p[i], s, lamb, name=i, weight=None,R=R))

    f_error_history = []
    for k in range(test):
        # グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i, name = Agents[i].send()
                Agents[j].receive(x_i, name)

        for i in range(n):
            Agents[i].update(k)

        f_value = []
        for i in range(n):
            x_i = Agents[i].x_i
            estimate_value = Dist_optimal_value(x_i, p, n, m, lamb)
            f_value.append(estimate_value)

        # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
        f_error_history.append(np.max(f_value) - f_opt)

    return f_error_history


def make_communication_graph(test):  # 通信グラフを作成＆保存
    weight_graph = Communication(n, 4, 0.3)
    weight_graph.make_connected_WS_graph()
    P = weight_graph.P
    P_history = []
    for k in range(test):  # 通信グラフを作成＆保存
        weight_graph.make_connected_WS_graph()
        P_history.append(weight_graph.P)
    return P, P_history


def main_L1(n, m, step, lamb,R, pattern, test):
    p, x_opt, f_opt = optimal_L1(n, m, lamb,R)
    P, P_history = make_communication_graph(test)
    f_error_history = [[] for i in range(pattern)]
    for agent in range(pattern):
        f_error_history[agent] = iteration_L1(n, m, p, step, lamb,R, test, P_history, f_opt, agent)
    print('finish')

    make_graph(pattern, f_error_history,step)

def main_L1_paper(n, m, step, lamb,R, pattern, test):
    p, x_opt, f_opt = optimal_L1(n, m, lamb,R)
    P, P_history = make_communication_graph(test)
    f_error_history = [[] for i in range(pattern)]
    for agent in range(pattern):
        f_error_history[agent] = iteration_L1_paper(n, m, p, step, lamb,R, test, P_history, f_opt, agent)
    print('finish')

    make_graph(pattern, f_error_history,step)



def main_L2(n, m, step, lamb,R, pattern, test):
    p, x_opt, f_opt = optimal_L2(n, m, lamb,R)
    P, P_history = make_communication_graph(test)
    f_error_history = [[] for i in range(pattern)]
    for agent in range(pattern):
        f_error_history[agent] = iteration_L2(n, m, p, step, lamb,R, test, P_history, f_opt, agent)
    print('finish')
    make_graph(pattern, f_error_history,step)


def main_Dist(n, m, step, lamb,R, pattern, test):
    p, x_opt, f_opt = optimal_Dist(n, m, lamb,R)
    P, P_history = make_communication_graph(test)
    f_error_history = [[] for i in range(pattern)]
    for agent in range(pattern):
        f_error_history[agent] = iteration_Dist(n, m, p, step, lamb,R, test, P_history, f_opt, agent)
    print('finish')
    make_graph(pattern, f_error_history,step)


def make_graph(pattern, f_error,step):
    label = ['DSM','Proposed']
    line = ['-','-.']
    for i in range(pattern):
        stepsize = '_s(k)=' + str(step[i]) + '/k+1'
        plt.plot(f_error[i], label=label[i%2]+stepsize,linestyle=line[i%2])
    plt.legend()
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    t = time.time()
    n = 20
    m = 10
    lamb = 0.1
    R = 10
    np.random.seed(0)  # ランダム値固定
    pattern = 8
    test = 1000
    step = [0.2,0.2,0.5,0.5,1.,1,2.,2.]
    main_L1_paper(n, m,  step, lamb,R, pattern, test)
    print(time.time() - t)
    # main_L2(n, m, step, lamb, pattern, test)
    # main_Dist(n, m, step, lamb, pattern, test)
