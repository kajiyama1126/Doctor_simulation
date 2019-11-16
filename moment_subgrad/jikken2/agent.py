import numpy as np


class Agent(object):
    def __init__(self, n, m, p, step, lamb, name, weight=None, R = 100000):
        self.n = n
        self.m = m
        self.p = p
        self.step = step
        self.lamb = lamb
        self.name = name
        self.weight = weight
        self.R = R
        #self.x_i = np.random.rand(self.m)
        self.x_i = np.zeros(self.m)
        self.x_i = self.project(self.x_i)
        self.x = np.zeros([self.n, self.m])

    def subgrad(self):
        grad = self.x_i - self.p
        subgrad_l1 = self.lamb*np.sign(self.x_i)
        subgrad = grad + subgrad_l1
        return subgrad

    def send(self):
        return self.x_i, self.name

    def receive(self, x_j, name):
        self.x[name] = x_j

    def s(self, k):
        # return self.step/ (k + 1.0)
        #return self.step/(k+10)
        return self.step / (k + 100)
        #return self.step / (k + 100)
        #return self.step / (k + 5000)
    def project(self,x):
        if np.linalg.norm(x) <= self.R:
            return x
        else:
            y = (self.R/np.linalg.norm(x)) * x
            return y


    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x)
        self.x_i = self.x_i- self.s(k) * self.subgrad()
        self.x_i = self.project(self.x_i)

class Agent_L2(Agent):
    def subgrad(self):
        grad = self.x_i-self.p
        grad_l2 = 2*self.lamb * self.x_i
        return grad+grad_l2

class Agent_Dist(Agent):
    def subgrad(self):
        grad = (self.x_i-self.p)/np.linalg.norm((self.x_i-self.p),2)
        return grad

class new_Agent(Agent):
    def __init__(self, n, m, A, p, step, lamb, name, weight=None, R=100000):
        super(new_Agent,self).__init__(n,m,p,step,lamb,name,weight=weight,R=R)
        self.A = A

    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
        subgrad_l1 = self.lamb*np.sign(self.x_i)
        subgrad = grad + subgrad_l1
        return subgrad

class new_Agent_L2(new_Agent):
    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
        grad_l2 = 2 * self.lamb * self.x_i
        subgrad = grad + grad_l2
        return subgrad




class Agent_moment_CDC2017(Agent):
    def __init__(self, n, m, p,step, lamb, name, weight=None, R = 100000):
        super(Agent_moment_CDC2017, self).__init__(n, m, p,step, lamb, name, weight,R)
        self.gamma = 0.9

        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])

    def send(self):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma * np.dot(self.weight, self.v) + self.s(k)*(0.1) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.x_i = self.project(self.x_i)

class new_Agent_moment_CDC2017(new_Agent):
    def __init__(self, n, m,A, p, step, lamb, name, weight=None,R = 100000):
        super(new_Agent_moment_CDC2017, self).__init__(n, m, A, p, step, lamb, name, weight,R)
        #self.gamma = 0.9
        self.gamma = 0.8
        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])

    def send(self):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name):
        self.x[name] = x_j[0]
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma * np.dot(self.weight, self.v) + self.s(k) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.x_i = self.project(self.x_i)

class new_Agent_moment_CDC2017_paper(new_Agent_moment_CDC2017):
    def hat_step(self,k):
        return (1000/(1000+ k)) ** 0.51

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma *self.hat_step(k) * np.dot(self.weight, self.v) + self.s(k) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.x_i = self.project(self.x_i)

class new_Agent_moment_CDC2017_paper2(new_Agent_moment_CDC2017_paper):
    def __init__(self, n, m,A, p, step, lamb, name, weight=None, R = 100000):
        super(new_Agent_moment_CDC2017_paper2, self).__init__(n, m, A, p, step, lamb, name, weight,R)
        self.gamma = step
        #self.step = 0.5
        # self.gamma = gamma_box[int(self.name / 2.0)]

class new_Agent_moment_CDC2017_L2(new_Agent_moment_CDC2017):
    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
        grad_l2 = 2 * self.lamb * self.x_i
        subgrad = grad + grad_l2
        return subgrad

class Agent_moment_CDC2017_paper(Agent_moment_CDC2017):
    def hat_step(self,k):
        return (1000/(1000 + k)) ** 0.51
        #return (1000 / (1000 + k)) ** 0.501

    def update(self, k):
        self.gamma = 0.9
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma *self.hat_step(k)* np.dot(self.weight, self.v) + self.s(k) * 0.1* self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.x_i = self.project(self.x_i)

class Agent_moment_CDC2017_L2(Agent_moment_CDC2017):
    def subgrad(self):
        grad = self.x_i-self.p
        grad_l2 = 2*self.lamb * self.x_i
        return grad+grad_l2

class Agent_moment_CDC2017_Dist(Agent_moment_CDC2017):
    def subgrad(self):
        grad = (self.x_i-self.p)/np.linalg.norm((self.x_i-self.p),2)
        return grad

# # class Agent_moment_CDC2017_s(Agent_moment_CDC2017):
#     def update(self, k):
#         self.x[self.name] = self.x_i
#         self.v[self.name] = self.v_i
#
#         self.v_i = self.gamma *self.s(k)/self.s(k-1)*np.dot(self.weight, self.v) + self.s(k)*(0.2) * self.subgrad()
#         self.x_i = np.dot(self.weight, self.x) - self.v_i

class Agent_harnessing(new_Agent):
    def __init__(self, n, m,A, p,s, lamb, name, weight=None,R = 100000):
        super(Agent_harnessing, self).__init__(n, m, A, p,s, lamb, name, weight,R=R)

        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])
        self.eta = 0.01

    def send(self):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i
        grad_bf = self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.eta*self.v_i
        self.v_i = np.dot(self.weight, self.v) +  self.subgrad() -grad_bf

class new_Agent_harnessing_L2(Agent_harnessing):
    def __init__(self, n, m,A, p,s, lamb, name, weight=None,R=1000000):
        self.A = A
        super(new_Agent_harnessing_L2, self).__init__(n, m, A, p, s,lamb, name, weight,R=R)


    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i) - self.p))
        grad_l2 = 2 * self.lamb * self.x_i
        subgrad = grad + grad_l2
        return subgrad

class new_Agent_Dist(new_Agent):
    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i)-self.p))/np.linalg.norm((np.dot(self.A,self.x_i)-self.p),2)
        return grad

class new_Agent_moment_CDC2017_Dist(new_Agent_moment_CDC2017):
    def subgrad(self):
        A_to = self.A.T
        grad = np.dot(A_to,(np.dot(self.A,self.x_i)-self.p))/np.linalg.norm((np.dot(self.A,self.x_i)-self.p),2)
        return grad

