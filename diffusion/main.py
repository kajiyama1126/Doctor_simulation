import numpy as np
import matplotlib.pylab as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import copy

from diffusion.Diffsuion_matrix import Make_Matrix, Diffusion
import random

from agent.agent import Agent_harnessing

random.seed(0)
if __name__=='__main__':
    pos = []
    for i in range(6):
        for j in range(6):
            pos_i = [4*i+2,4*j+2]
            pos.append(pos_i)

    # for i in range(100):
    #     pos_i = [random.randint(0,17),random.randint(0,17)]
    #     pos.append(pos_i)

    # pos = [[1,1],[7,1],[4,2],[16,2],[8,3],[14,4],[17,4],[6,6],[7,6],[12,8],
    #        [15,9],[5,10],[13,11],[10,10],[14,12],[3,13],[5,14],[11,14],[7,16],[2,17],[17,17],[10,17],[15,17]]
    # pos = [[2,5],[4,7],[5,5],[4,6],[6,4],[2,7],[7,2],[2,2],[7,7],[6,6],[4,4],[3,8],[4,5],[6,2]]
    n= len(pos)
    # n= 9
    for x,y, in pos:
        plt.scatter(x,y)
    plt.show()

    x = 5
    y = 5

    x_div = 25
    y_div = 25
    m = (x_div-1)*(y_div-1)
    D = 2.8*(1e-3 )*5*2*2
    delta_t = 0.04

    delta_start = 5
    delta_end = 5
    # iteration = 1000
    iteration = delta_end-delta_start
    update_iteration = 10000 # エージェントの計算回数
    eta = 0.02
    lam = 0.05
    # pos = [[5,5],[4,6],[6,4]]
    # pos = [[5,5],[4,6],[6,4],[2,8],[8,2],[2,2],[8,8],[6,6],[4,4],[10,10],[13,13],[15,15],[18,12],[12,18]]
    # pos = [[5,5],[4,6],[6,4],[2,7],[7,2],[2,2],[7,7],[6,6],[4,4]]

    phi = []
    Omega = []
    for i in range(n):
        omega = Make_Matrix(x,y,x_div,y_div,D,delta_t,delta_start,delta_end)
        Omega.append(omega)

    for i in range(n):
        Omega[i].omega()
        Omega[i].make_theta(pos[i])
        Omega[i].make_R()
        Omega[i].make_Psi()
    print('finish Omega')
    omega_param = Omega[0].omega_maxtrix

    Map = Diffusion(x,y,x_div,y_div,omega_param)
    Map.make_map()
    # Map.make_initial_distribution()
    Map.make_initial_distribution2()
    # Map.draw3D()
    Map.draw2()

    # m = (x_div-1)*(y_div-1)
    # Omega = np.zeros(m,m)
    # Omega_ij =
    # for h in range(x_div-1):
    #     for h2 in rangex_div-1):
    #
    #         for i in range(y_div-1):
    #             for j in range(y_div-1):

    Map.update_2(iteration=delta_start)
    Map.draw2()
    for i in range(n):
        x_pos = pos[i][0]
        y_pos = pos[i][1]
        phi_i = np.array([[Map.value_return2(x_pos+1,y_pos)-Map.value_return2(x_pos,y_pos)],
                       [Map.value_return2(x_pos,y_pos+1)-Map.value_return2(x_pos,y_pos)]])
        phi.append(phi_i)
    for k in range(iteration):
        # Map.update_1(1,D,delta_t)
        Map.update_2(iteration = 1)
        for i in range(n):
            x_pos = pos[i][0]
            y_pos = pos[i][1]
            phi_i2 = np.array([[Map.value_return2(x_pos + 1, y_pos) - Map.value_return2(x_pos, y_pos)],
                            [Map.value_return2(x_pos, y_pos + 1) - Map.value_return2(x_pos, y_pos)]])
            phi[i] = np.concatenate([phi[i],phi_i2])
    Map.draw2()
    # Map.draw3D()




    A =[]
    b = []
    A_sup =np.array([])
    b_sup=np.array([])
    for i in range(n):
        A_i = np.dot(Omega[i].Psi,Omega[i].R)
        print('finish 1')
        b_i = np.dot(Omega[i].Psi,phi[i])
        print('finish 2')
        A.append(A_i)
        b.append(b_i)

    #     if len(A_sup) ==0:
    #         A_sup = A_i
    #         b_sup = b_i
    #     else:
    #         A_sup = np.vstack(A_sup,A_i)
    #         b_sup = np.vstack(b_sup,b_i)
    # # lam = 0.0

    from diffusion.problem_optimizaiton import Problem_L2
    Optimization = Problem_L2(n,m,A,b,lam)
    Optimization.solve()
    optimal_value = Optimization.send_f_opt()

    estimate_diffusion = Optimization.send_x()
    estimate_diffusion = np.array(estimate_diffusion).ravel()


    del_x = x/ (x_div-1)
    del_y = y/ (y_div-1)
    x_plot = np.arange(0, x, del_x)
    y_plot = np.arange(0, y, del_y)
    X, Y = plt.meshgrid(x_plot, y_plot)
    Z = estimate_diffusion.reshape([x_div-1,y_div-1])

    plt.pcolor(X, Y, Z.T)
    plt.colorbar()
    plt.show()


    from agent.agent import Agent_harnessing_diffusion
    Agent = []

    # weight =[1/3,1/3,1/3]
    weight = [1/n for i in range(n)]
    # weight = [1]
    for i in range(n):
        agent = Agent_harnessing_diffusion(n,m,A[i],b[i],eta=eta ,weight = weight ,name=i,lam = lam)
        Agent.append(agent)



    cost_value_estimate = []
    for k in range(update_iteration):
        print(k)
        for i in range(n):
            for j in range(n):
                state , name = Agent[i].send(j)
                Agent[j].receive(state,name)

        for i in range(n):
            Agent[i].update(k)

        cost_value = 0
        for i in range(n):
            cost_value += 1/2* np.linalg.norm(np.dot(A[i],(Agent[0].x_i.reshape([-1,1])))-b[i]) ** 2 + 1/2*lam * np.linalg.norm(Agent[0].x_i)**2
        cost_value +=  -optimal_value

        print(cost_value)
        cost_value_estimate.append(cost_value)


    print(agent.x_i)
    estimate_diffusion = Agent[0].x_i
    estimate_diffusion = estimate_diffusion.reshape([x_div-1,y_div-1])

    del_x = x/ (x_div-1)
    del_y = y/ (y_div-1)
    x_plot = np.arange(0, x, del_x)
    y_plot = np.arange(0, y, del_y)
    X, Y = plt.meshgrid(x_plot, y_plot)
    Z = estimate_diffusion

    plt.pcolor(X, Y, Z.T)
    plt.colorbar()
    plt.show()

    x_axis = np.linspace(0,update_iteration-1,update_iteration)
    plt.plot(x_axis,cost_value_estimate)
    plt.yscale('log')
    plt.show()
    # print(omega)