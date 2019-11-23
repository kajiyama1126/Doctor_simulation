import cvxpy as cvx

class Solver(object):
    def __init__(self, n, m, A, b):
        self.n = n
        self.m = m
        self.A = A
        self.b = b

        self.solve()

    def solve(self):
        # print(cvx.installed_solvers())
        n, m = self.n, self.m
        self.x = cvx.Variable(m)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(1 / 2 * cvx.power(cvx.norm((self.A[i] * self.x - self.b[i]), 2), 2))
        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=True, abstol=1.0e-10, feastol=1.0e-10)
        print(self.prob.status, self.x.value)

    def send_opt(self):
        return self.prob.value,self.x.value

class Solver_logistic(Solver):
    def __init__(self, n, m, A, b,lam):
        self.lam = lam
        super(Solver_logistic, self).__init__(n,m,A,b)


    def solve(self):
        n, m = self.n, self.m
        self.x = cvx.Variable(m)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(cvx.logistic(self.A[i] * self.x) - self.b[i]*self.A[i]*self.x)
        obj += cvx.Minimize(1/2*self.lam * cvx.norm(self.x)**2)
        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=True, abstol=1.0e-10, feastol=1.0e-10)
        print(self.prob.status, self.x.value)

class Solver_hinge(Solver):
    def __init__(self, n, m, A, b,lam,epsiron):
        self.lam = lam
        self.epsiron = epsiron
        super(Solver_hinge, self).__init__(n,m,A,b)


    def solve(self):
        n, m = self.n, self.m
        self.x = cvx.Variable(m)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(cvx.pos(cvx.abs(self.b[i]-self.A[i] * self.x) -self.epsiron))
        obj += cvx.Minimize(1/2*self.lam * cvx.norm(self.x)**2)
        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=True, abstol=1.0e-10, feastol=1.0e-10)
        print(self.prob.status, self.x.value)
