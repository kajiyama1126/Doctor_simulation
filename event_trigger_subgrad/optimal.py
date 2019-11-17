import numpy as np
import math
import cvxpy as cvx

def log_loss(x,y):
    return math.log(1+math.exp(x))-y*x
def sigmoid(x,y):
    return math.exp(x)/(1+math.exp(x))-y

class Optimal(object):
    def __init__(self, n, m,A,b,  lamb,R):
        """
        :param n: int
        :param m: int
        :param lamb: float
        :return: float,float,float
        """
        self.n = n
        self.m = m
        # self.step = step
        self.lamb = lamb
        self.R = R

        self.A = A
        self.b = b


    def optimal(self):  # L1
        """
        :return:  float, float
        """
        x = cvx.Variable(self.m)
        # self.p = [np.random.randn(self.m) for i in range(self.n)]
        # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
        # self.p_num = np.array(self.p)
        # np.reshape(p)
        prob = cvx.Minimize(0)
        for i in range(self.n):
            for j in range(len(self.A[i])):
                prob += cvx.Minimize(cvx.logistic(self.A[i][j] * x ) -self.b[i][j] *(self.A[i][j] * x ) )
            prob += cvx.Minimize(self.lamb* cvx.norm(x,1))

        prob = cvx.Problem(prob)
        prob.solve()
        x_opt = x.value # 最適解
        # x_opt = np.reshape(x_opt, (-1,))  # reshape
        # f_opt = prob.send_f_opt()
        return x_opt,prob.value
            # , f_opt