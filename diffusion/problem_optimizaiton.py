import cvxpy as cvx
import numpy as np


class Problem(object):
    def __init__(self, n, m, A, b):
        self.n = n
        self.m = m
        self.A = A
        self.b = b


    def solve(self):
        # print(cvx.installed_solvers())
        n, m = self.n, self.m
        self.x = cvx.Variable(m)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj+=cvx.Minimize(1 / 2 * cvx.power(cvx.norm((self.A[i]*self.x - self.b[i]), 2), 2))

        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=True,abstol=1.0e-15,feastol=1.0e-15)
        print(self.prob.status, self.x.value)

    def send_f_opt(self):
        return self.prob.value

    def send_x(self):
        return self.x.value

class Problem_L2(Problem):
    def __init__(self,n,m,A,b,lam,b_m):
        super(Problem_L2,self).__init__(n,m,A,b)
        self.lam = lam
        self.b_m = b_m
    def solve(self):
        A_matrix = np.array(self.A[0])
        b_matrix = np.array(self.b[0])
        for i in range(self.n-1):
            A_matrix = np.vstack((A_matrix,self.A[i+1]))
            b_matrix = np.vstack((b_matrix, self.b[i + 1]))


        # print(cvx.installed_solvers())
        n, m = self.n, self.m
        b_matrix = b_matrix.reshape(self.b_m)
        self.x = cvx.Variable(m)
        # obj = cvx.Minimize(0)
        # obj += cvx.Minimize(0)
        # for i in range(n):
            # tmp = self.A[i]*self.x
        obj = 1/2*cvx.Minimize(cvx.sum_squares(A_matrix * self.x - b_matrix))+1/2*n*self.lam*cvx.Minimize(cvx.sum_squares(self.x))
            # obj = cvx.Minimize(cvx.sum_squares(A_matrix * self.x ))
            # obj += cvx.Minimize(cvx.sum_squares(tmp - self.b[i]))
            # obj+=cvx.Minimize(1 / 2 * cvx.power(cvx.norm((self.A[i]*self.x - self.b[i]), 2), 2))
        # obj += cvx.Minimize(1 / 2 * n * self.lam * cvx.power(cvx.norm((self.x), 2), 2))
        #                  + 1 / 2 * n * self.lam * cvx.power(cvx.norm((self.x), 2), 2))
        # cvx.Minimize()
        self.prob = cvx.Problem(obj)
        self.prob.solve(solver=cvx.SCS,verbose=True,max_iters=50000,eps=1e-15)
        # ,verbose=True,abstol=1e-15,feastol=1e-15)
        print(self.prob.status, self.x.value)


