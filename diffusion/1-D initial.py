import numpy as np
import matplotlib.pylab as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import copy

from diffusion.Diffsuion_matrix import Make_Matrix, Diffusion

from agent.agent import Agent_harnessing

class data_saver(object):
    def __init__(self,pos,iteration):
        self.dim = len(pos)
        self.b = np.zeros(self.dim*iteration)

    def get(self,value,iteration):
        for i in range(self.dim):
            self.b[self.dim * iteration + i]  = value[i]


if __name__=='__main__':
    pos = []
    for i in range(8):
        for j in range(8):
            pos_i = [i,j]
            pos.append(pos_i)
    n= len(pos)
    # n= 9

    x = 5
    y = 5

    x_div = 25
    y_div = 25
    D = 2.8*(1e-3 )*5
    delta_t = 0.04
    # iteration = 1000
    iteration =1
    update_iteration = 2500
    # pos = [[5,5],[4,6],[6,4]]
    # pos = [[5,5],[4,6],[6,4],[2,8],[8,2],[2,2],[8,8],[6,6],[4,4],[10,10],[13,13],[15,15],[18,12],[12,18]]
    # pos = [[5,5],[4,6],[6,4],[2,7],[7,2],[2,2],[7,7],[6,6],[4,4]]

    phi = []
    Omega = []
    for i in range(n):
        omega = Make_Matrix(x,y,x_div,y_div,D,delta_t,iteration)
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
    # Map.draw2()

    # m = (x_div-1)*(y_div-1)
    # Omega = np.zeros(m,m)
    # Omega_ij =
    # for h in range(x_div-1):
    #     for h2 in rangex_div-1):
    #
    #         for i in range(y_div-1):
    #             for j in range(y_div-1):

    for i in range(n):
        x_pos = pos[i][0]
        y_pos = pos[i][1]
        phi_i = np.array([[Map.value_return2(x_pos+1,y_pos)-Map.value_return2(x_pos,y_pos)],
                       [Map.value_return2(x_pos,y_pos+1)-Map.value_return2(x_pos,y_pos)]])
        phi.append(phi_i)
    for k in range(iteration-1):
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
    for i in range(n):
        A_i = np.dot(Omega[i].Psi,Omega[i].R)
        print('finish 1')
        b_i = np.dot(Omega[i].Psi,phi[i])
        print('finish 2')
        A.append(A_i)
        b.append(b_i)
    # lam = 0.0
    lam = 0.1

    from agent.agent import Agent_harnessing_diffusion
    Agent = []

    # weight =[1/3,1/3,1/3]
    weight = [1/n for i in range(n)]
    # weight = [1]
    for i in range(n):
        agent = Agent_harnessing_diffusion(n,(x_div-1)*(y_div-1),A[i],b[i],eta=0.0025,weight = weight ,name=i,lam = lam)
        Agent.append(agent)



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
            cost_value += 1/2* np.linalg.norm(np.dot(A[i],(Agent[0].x_i.reshape([-1,1])))-b[i]) ** 2 + 1/2/n*lam * np.linalg.norm(Agent[0].x_i)**2
        print(cost_value)

    print(agent.x_i)
    estimate_diffusion = Agent[10].x_i
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
    # print(omega)