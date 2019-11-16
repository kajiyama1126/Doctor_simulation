from agent.agent import Agent
import numpy as np


class Agent_event(Agent):
    def __init__(self, n, m, A, b, weight, name):
        super(Agent_event, self).__init__(n, m, A, b, weight, name)

        self.tilde_x_ij = np.zeros([self.n, self.m])
        self.tilde_x_ji = np.zeros([self.n, self.m])

        self.thresh = np.ones(n)
    def send(self, j):
        if (self.thresh[j] <= np.linalg.norm(self.tilde_x_ij - self.x_i)):
            return self.x_i, self.name
        else:
            return None.j

    def receive(self, x_j, name):
        if x_j is None:
            pass
        else:
            self.x[name] = x_j