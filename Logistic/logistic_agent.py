from agent.agent import Agent_harnessing_quantize_add_send_data,Agent_harnessing,Agent
import math
import numpy as np

class Agent_harnessing_logistic(Agent_harnessing):
    def __init__(self,n, m, A, b,eta, weight, name,lam):
        self.lam = lam
        super(Agent_harnessing_logistic, self).__init__(n, m, A, b,eta, weight, name)


    def grad(self):
        g = 0
        # print(len(self.A))
        for i in range(len(self.A)):
            x = np.dot(self.x_i,self.A[i])
            g += ((math.exp(x) / (1 + math.exp(x))) - self.b[i])*self.A[i]
        g+= self.lam * self.x_i

        return g

class Agent_harnessing_logistic_quantize_add_send_data(Agent_harnessing_logistic,Agent_harnessing_quantize_add_send_data):
    def __init__(self, n, m, A, b, weight, name):
        super(Agent_harnessing_logistic_quantize_add_send_data, self).__init__(n, m, A, b, name, eta)

class Agent_event(Agent):
    def __init__(self, n, m, A, b, weight, name):
        super(Agent_event, self).__init__(n, m, A, b, weight, name)

        self.tilde_x_ij = np.zeros([self.n, self.m])
        self.tilde_x_ji = np.zeros([self.n, self.m])

        self.x_i_hat = self.x_i
        self.thresh = np.ones(n)/100
        self.trigger_count = np.zeros(n)

    def send(self, k,j):
        if (self.threshold(k,j) <= np.linalg.norm(self.tilde_x_ij[j] - self.x_i)):
            self.trigger_count[j] += 1
            # print(self.trigger_count[j])
            # print(self.trigger_count)
            return self.x_i, self.name
        else:
            return None, self.name

    def receive(self, x_j, name):
        if x_j is None:
            pass
        else:
            self.x[name] = x_j

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x) - self.step(k)*self.grad()

    def step(self,k):
        pass

    def grad(self):
        pass

    def threshold(self,k,j):
        return self.thresh[j]*1/((k+1)**2)

class Agent_event_step_fix(Agent_event):
    def __init__(self,n, m, A, b, weight, name,step):
        super(Agent_event_step_fix, self).__init__(n, m, A, b, weight, name)
        self.s = step

    def step(self,k):
        return self.s

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x) - self.step(k)*self.grad()
        self.x_i_hat = 1/(k+1)*self.x_i + (k/(k+1))*self.x_i_hat


class Agent_event_step_proj(Agent_event):
    def __init__(self, n, m, A, b, weight, name, step,proj):
        super(Agent_event_step_proj, self).__init__(n, m, A, b, weight, name)

        self.s = step
        self.proj = proj

    def step(self, k):
        return self.s * 1./(k+1)

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x) - self.step(k)*self.grad()

        if (np.linalg.norm(self.x_i) > self.proj):
            self.x_i = self.proj / np.linalg.norm(self.x_i) * self.x_i


class Agent_hinge_event(Agent_event):
    def __init__(self,n, m, A, b,s, weight, name,lam,epsiron):
        super(Agent_hinge_event, self).__init__( n, m, A, b, weight, name)
        self.s = s
        self.lam = lam
        self.epsiron= epsiron


    def grad(self):
        g = 0
        # print(len(self.A))
        for i in range(len(self.A)):
            h = abs(self.b[i] -np.dot(self.A[i],self.x_i))-self.epsiron
            if h>0:
                if (self.b[i] -np.dot(self.A[i],self.x_i)) > 0:
                    g+= - self.A[i]
                else:
                    g+= self.A[i]
            else:
                g+= 0
            # x = np.dot(self.x_i,self.A[i])
            # g += ((math.exp(x) / (1 + math.exp(x))) - self.b[i])*self.A[i]
        g+= self.lam * self.x_i

        return g

class Agent_hinge_event_fix(Agent_hinge_event):
    def step(self, k):
        return self.s

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x) - self.step(k)*self.grad()

        # self.x_i = np.dot(self.weight, self.x) - self.step(k)*self.grad()
        self.x_i_hat = 1/(k+1)*self.x_i + (k/(k+1))*self.x_i_hat


        # if (np.linalg.norm(self.x_i) > self.proj):
        #     self.x_i = self.proj / np.linalg.norm(self.x_i) * self.x_i

class Agent_hinge_event_proj(Agent_hinge_event):
    def __init__(self,n, m, A, b,s, weight, name,lam,epsiron ,proj):
        super(Agent_hinge_event_proj, self).__init__( n, m, A, b,s, weight, name,lam,epsiron)
        self.proj = proj

    def step(self, k):
        return self.s * 1./(k+1)

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x) - self.step(k)*self.grad()

        if (np.linalg.norm(self.x_i) > self.proj):
            self.x_i = self.proj / np.linalg.norm(self.x_i) * self.x_i