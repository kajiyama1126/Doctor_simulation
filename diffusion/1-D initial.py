import numpy as np
import matplotlib.pylab as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import copy


class Diffusion(object):
    def __init__(self,x,y,x_div,y_div,omega):


        self.h_x = x
        self.h_y = y

        self.x_div = x_div-1
        self.y_div = y_div-1

        # self.x_size_div = int(self.h_x / self.x_div)
        # self.y_size_div = int(self.h_y / self.y_div)
        self.del_x = self.h_x /self.x_div
        self.del_y = self.h_y / self.y_div
        self.time = 0
        self.omega = omega
        self.map = None

    def make_map(self):
        self.map = np.zeros((self.y_div,self.x_div))
        self.u = np.zeros(self.x_div*self.y_div)


    def make_initial_distribution(self):
        for i in range(self.x_div):
            for j in range(self.y_div):
                x = self.del_x * i
                y = self.del_y * j
                self.map[j][i] = self.distribution_function(x,y)

    def distribution_function(self,x,y):
        # tmp = math.exp(-0.1*((x-15)**2 + (y-10)**2))
        # tmp1 = 0.5* math.exp(-0.2*((x-5)**2+(y-15)**2))
        initial = 0.6*math.exp(-3.0*((x-3.4)**2 + (y-1.6)**2))
        initial2 = 0.8*math.exp(-2.0*((x-1.4)**2+(y-2.8)**2))
        return initial+initial2

    def make_initial_distribution2(self):
        for i in range(self.x_div):
            for j in range(self.y_div):
                x = self.del_x * i
                y = self.del_y * j
                self.u[i * self.y_div + j] = self.distribution_function(x,y)


    def draw(self):
        x = np.arange(0, self.h_x, self.del_x)
        y= np.arange(0, self.h_y, self.del_y)
        X,Y = plt.meshgrid(x,y)
        Z = self.map

        plt.pcolor(X,Y,Z)
        plt.colorbar()
        plt.show()

    def draw3D(self):
        x = np.arange(0, self.h_x, self.del_x)
        y= np.arange(0, self.h_y, self.del_y)
        X,Y = plt.meshgrid(x,y)
        Z = self.map

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.plot_wireframe(X,Y,Z)

        plt.show()

    def draw2(self):
        x = np.arange(0, self.h_x, self.del_x)
        y= np.arange(0, self.h_y, self.del_y)
        X,Y = plt.meshgrid(x,y)
        Z = self.u.reshape([self.x_div,self.y_div])

        plt.pcolor(X,Y,Z.T)
        plt.colorbar()
        plt.show()


    # def update(self):

        # h_x = int(self.x_size/self.x_div)
        # h_y = int(self.y_size/self.y_div)
        # h = (h_x-1)*(h_y-1)
        #
        # self.Omega = np.zeros((self.h,self.h))
        # for i in range(self.h):
        #     for j in range(self.h):
        #         if i+1 == j or i-1 == j:
        #             self.Omega[i][j]


    def update_1(self,iteration,D,delta_t):
        # self.map_update = copy.copy(self.map)
        alpha = D*delta_t/(self.del_x)**2
        beta = D*delta_t/(self.del_y)**2
        gamma = 1-2*alpha - 2*beta

        for k in range(iteration):
            map_bf = copy.copy(self.map)
            for i in range(self.x_div):
                for j in range(self.y_div):
                    if i==0 or i == self.x_div-1 or j==0 or j == self.y_div-1:
                        self.map[j][i] =0
                    else:
                        self.map[j][i] = alpha * (map_bf[j+1][i]+map_bf[j-1][i]) + beta * (map_bf[j][i+1]+map_bf[j][i-1])+gamma * map_bf[j][i]


    def update_2(self,iteration):
        for k in range(iteration):
            self.u = np.dot(self.omega,self.u)

    def value_return(self,x_pos,y_pos):
        return self.map[int(y_pos)][int(x_pos)]

    def value_return2(self,x_pos,y_pos):
        return self.u[x_pos * self.y_div + y_pos]


from agent.agent import Agent_harnessing

class data_saver(object):
    def __init__(self,pos,iteration):
        self.dim = len(pos)
        self.b = np.zeros(self.dim*iteration)

    def get(self,value,iteration):
        for i in range(self.dim):
            self.b[self.dim * iteration + i]  = value[i]

# class omega(object):
#     def __init__(self):

class Make_Matrix(object):
    def __init__(self,x,y,x_div,y_div,D,iteration):
        self.h_x = x
        self.h_y = y
        self.x_div = x_div-1
        self.y_div = y_div-1
        self.dim = 2 #x,y

        self.del_x = self.h_x /self.x_div
        self.del_y = self.h_y / self.y_div

        self.alpha = D*delta_t/(self.del_x)**2
        self.beta = D*delta_t/(self.del_y)**2
        self.gamma = 1-2*self.alpha - 2*self.beta

        self.m = self.x_div *self.y_div
        self.iteration = iteration

    def make_omega_ij(self,i,j):
        if i==j:
            omega_ij =self.gamma* np.identity(self.y_div)
            for i1 in range(self.y_div):
                for j1 in range(self.y_div):
                    if (i1 == j1+1) or (i1 == j1-1) :
                        omega_ij[i1][j1] = self.beta

        elif (i == j+1) or (i == j-1) :
            omega_ij = self.alpha*np.identity(self.y_div)
        else:
            omega_ij = np.zeros([self.y_div,self.y_div])

        return omega_ij

    def omega(self):
        self.omega_maxtrix = np.zeros([self.m,self.m])

        for i in range(self.x_div):
            for j in range(self.x_div):
                # print(i,j)
                omega_ij = self.make_omega_ij(i,j)
                # for i1 in range(self.y_div):
                #     for j1 in range(self.y_div):
                i
                for i1 in range(self.y_div):
                    for j1 in range(self.y_div):
                        self.omega_maxtrix[i*self.y_div+i1][j*self.y_div+j1] = omega_ij[i1][j1]

    def make_theta(self,pos):
        # self.n = n
        x = pos[0]
        y = pos[1]

        self.theta = np.zeros([self.dim,self.m])
        mu = x*self.y_div + y -1
        nu = (x-1)*self.y_div + y -1
        for i in range(self.dim):
            for j in range(self.m):
                if (((i ==0) and (j==mu)) or ((i==1 and j==nu+1))):
                    self.theta[i][j] = 1
                elif (((i ==0) and (j==nu)) or ((i==1 and j==nu))):
                    self.theta[i][j]=-1
                else:
                    self.theta[i][j]=0

    def make_R(self):
        self.R = np.dot(self.theta,self.omega_maxtrix)
        omega_dot = self.omega_maxtrix
        for i in range(self.iteration-1):
            print(i)
            omega_dot = np.dot(omega_dot,self.omega_maxtrix)
            self.R = np.concatenate([self.R,np.dot(self.theta,omega_dot)])

    def make_Psi(self):
        diag_list = []
        for i in range((self.iteration)):
            diag_list.append(1.0/self.del_x)
            diag_list.append(1.0/self.del_y)

        self.Psi = np.diag(diag_list)



# class Agent_harnessing_diffusion(Agent_harnessing):

# m =
# Omega = np.zeros()

#
# def omega_ij(alpha,beta,gamma,i,j,y_div):
#     if (i==j+1) or (i+1==j):
#         return alpha*np.identity(y_div-1)
#     elif (i==j):
#         tmp = gamma*np.identity(y_div-1)
#         for ell1 in range(y_div-1):
#             for ell2 in range(y_div-1):
#                 if (ell1+1 == ell2) or (ell1 == ell2+1):
#                     tmp[ell1][ell2] = beta

# class


if __name__=='__main__':
    n= 9

    x = 5
    y = 5

    x_div = 10
    y_div = 10
    D = 2.8*(1e-3 )*5
    delta_t = 0.04
    iteration = 1000

    # pos = [[5,5],[4,6],[6,4]]
    # pos = [[5,5],[4,6],[6,4],[2,8],[8,2],[2,2],[8,8],[6,6],[4,4],[10,10],[13,13],[15,15],[18,12],[12,18]]
    pos = [[5,5],[4,6],[6,4],[2,7],[7,2],[2,2],[7,7],[6,6],[4,4]]
    phi = []
    Omega = []
    for i in range(n):
        omega = Make_Matrix(x,y,x_div,y_div,D,iteration)
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
    lam = 0.0
    
    from agent.agent import Agent_harnessing_diffusion
    Agent = []

    # weight =[1/3,1/3,1/3]
    weight = [1/n for i in range(n)]
    # weight = [1]
    for i in range(n):
        agent = Agent_harnessing_diffusion(n,(x_div-1)*(y_div-1),A[i],b[i],eta=0.0005,weight = weight ,name=i,lam = lam)
        Agent.append(agent)



    for k in range(100000):
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
    estimate_diffusion = Agent[0].x_i
    estimate_diffusion = estimate_diffusion.reshape([x_div-1,y_div-1])

    del_x = x/ x_div
    del_y = y/ y_div
    x = np.arange(0, x, del_x)
    y = np.arange(0, y, del_y)
    X, Y = plt.meshgrid(x, y)
    Z = estimate_diffusion

    plt.pcolor(X, Y, Z)
    plt.colorbar()
    plt.show()
    # print(omega)