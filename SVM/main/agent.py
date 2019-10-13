import numpy as np
#from autograd import grad

class Agent_subgrad(object):
    #def __init__(self, n, m, A, b, eta, weight, name, C_i):
    def __init__(self, n, m, A, b, eta, weight, name, bKi, tKi, C_i):
        self.n = n
        self.m = m
        self.A_i = A
        self.b_i = b
        self.name = name
        self.weight = weight
        self.eta = eta
        self.bKi = bKi
        self.tKi = tKi
        self.C_i = C_i

        self.initial_state()

        self.trigger_x_ij = [[] for i in range(self.n)]
        self.trigger_v_ij = [[] for i in range(self.n)]

    def initial_state(self):
        #self.x_i = 2 * np.random.rand(self.m) - 1
        self.x_i = np.random.rand(self.m)
        #self.x_i = np.random.rand(self.m)
        # self.x_i = 1 * np.ones(self.m)
        self.x = np.zeros([self.n, self.m])
        #self.x = np.ones([self.n, self.m])
        #self.x = np.rand([self.n, self.m])
        self.v_i = self.grad()
        self.v = np.zeros([self.n, self.m])
        self.trigger_count = 0
        self.neighbor_agents = np.sum(np.sign(self.weight)) - 1

    def bsgn(self, x):
        sy = 0
        if x >= 0:
            sy = 1
        return sy

    def grad(self):
        M = len(self.b_i)
        gtmp = np.zeros(self.m)
        #gtmp = 0
        #print((self.bKi[self.name * M: (self.name+1) * M]).shape)
        for p in range(M):
            gtmp = gtmp + self.bsgn(1-self.b_i[p]*np.dot(self.bKi[self.name * M: (self.name+1) * M][p], self.x_i)) * (-self.b_i[p] * self.bKi[self.name * M: (self.name+1) * M][p].T)

        g = self.C_i * gtmp / M + np.dot(self.tKi, self.x_i)

        return g


    def send(self, j):
        self.trigger_count += 1

        if self.weight[j] != 0:
            self.trigger_x_ij[j].append(j + 1)
            self.trigger_v_ij[j].append(None)
            self.trigger_count += 1

        else:
            self.trigger_x_ij[j].append(None)
            self.trigger_v_ij[j].append(None)

        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name):
        self.x[name] = x_j[0]
        self.v[name] = None

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = None
        #grad_bf = self.grad()
        self.x_i = np.dot(self.weight, self.x) - self.step_size(k, self.eta) * self.grad()
        self.v_i = None

    def step_size(self, k, eta):
        #return eta / np.sqrt(k+900000000)
        #return eta / np.sqrt(k + 1)
        return eta / np.sqrt(k + 1000000)
        #return eta

    def trigger_x_ij_v_ij(self):
        return self.trigger_x_ij, None


##=======================================================================================================##

class Agent_harnessing(object):
    def __init__(self, n, m, A, b, eta, weight, name, C_i):
        self.n = n
        self.m = m
        self.A_i = A
        self.b_i = b
        self.name = name
        self.weight = weight
        self.eta = eta
        self.C_i = C_i

        self.initial_state()

        self.trigger_x_ij = [[] for i in range(self.n)]
        self.trigger_v_ij = [[] for i in range(self.n)]

    def initial_state(self):
        self.x_i = np.random.rand(self.m)
        #self.x_i = np.random.rand(self.m)
        # self.x_i = 0.5 * np.ones(self.m)
        # self.x_i = 1 * np.ones(self.m)
        self.x = np.zeros([self.n, self.m])
        #self.x = np.ones([self.n, self.m])
        #self.x = np.rand([self.n, self.m])
        self.v_i = self.grad()
        self.v = np.zeros([self.n, self.m])
        self.trigger_count = 0
        self.neighbor_agents = np.sum(np.sign(self.weight)) - 1

    def bsgn(self, x):
        sy = 0
        if x > 0:
            sy = 1
        return sy

    def grad(self):
        size = len(self.b_i)
        gtmp = np.zeros(self.m)

        for p in range(size):
            gtmp = gtmp + self.bsgn(1-self.b_i[p]*np.dot(self.x_i, self.A_i[p])) * (-self.b_i[p] * self.A_i[p])

        tilde_x = self.x_i
        # tilde_x[2] = 0

        g = self.C_i * gtmp / size + tilde_x / self.n

        return g

    # def grad(self):
    #     g = 0
    #     for l in range(len(self.A_i)):
    #         x = np.dot(self.x_i, self.A_i[l])
    #         #y = ((np.exp(np.clip(x, -744, 709)) / (1 + np.exp(np.clip(x, -744, 709)))) - self.b_i[l])
    #         y = np.exp(x) / (1 + np.exp(x)) - self.b_i[l]
    #         # if y == 0:
    #         #     y = 10 ** -323
    #         g += y * self.A_i[l]
    #     g = g + self.lam * self.x_i
    #     return g

    def send(self, j):
        self.trigger_count += 1

        if self.weight[j] != 0:
            self.trigger_x_ij[j].append(j + 1)
            self.trigger_v_ij[j].append(j + 1)
            self.trigger_count += 1

        else:
            self.trigger_x_ij[j].append(None)
            self.trigger_v_ij[j].append(None)

        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i
        grad_bf = self.grad()
        self.x_i = np.dot(self.weight, self.x) - self.eta * self.v_i
        self.v_i = np.dot(self.weight, self.v) + (self.grad() - grad_bf)

    def trigger_x_ij_v_ij(self):
        return self.trigger_x_ij, self.trigger_v_ij


##=======================================================================================================##
class Agent_harnessing_logistic(Agent_harnessing):

    def __init__(self, n, m, A, b, eta, weight, name, bKi, tKi, C_i):
        self.n = n
        self.m = m
        self.A_i = A
        self.b_i = b
        self.name = name
        self.weight = weight
        self.eta = eta
        self.bKi = bKi
        self.tKi = tKi
        self.C_i = C_i

        self.initial_state()
        self.trigger_x_ij = [[] for i in range(self.n)]
        self.trigger_v_ij = [[] for i in range(self.n)]

    def grad(self):
        size = len(self.b_i)
        gtmp = np.zeros(self.m)
        M = size

        for p in range(size):
            # gtmp = gtmp + self.bsgn(1-self.b_i[p]*np.dot(self.x_i, self.A_i[p])) * (-self.b_i[p] * self.A_i[p])
            gtmp1 = (self.b_i[p] * self.bKi[self.name * M: (self.name+1) * M][p].T)
            # gtmp2 = np.exp(self.b_i[p] * np.dot(self.x_i,self.bKi[self.name * M: (self.name+1) * M][p]))/(1+np.exp(self.b_i[p] * np.dot(self.x_i,self.bKi[self.name * M: (self.name+1) * M][p])))
            gtmp2 = 1 / (1 + np.exp(-self.b_i[p] * np.dot(self.x_i, self.bKi[self.name * M: (self.name + 1) * M][p])))
            gtmp += gtmp1 * gtmp2
        tilde_x = self.x_i
        # tilde_x[2] = 0

        g = self.C_i * gtmp / size + tilde_x / self.n

        return g

##=======================================================================================================##
class Agent_harnessing_EventTriggered_trigger(object):
    def __init__(self, n, m, A, b, eta, lam, weight, name, th_pa):

        self.n = n
        self.m = m
        self.A_i = A
        self.b_i = b
        self.name = name
        self.weight = weight
        self.eta = eta
        self.lam = lam

        self.TriggerSend = TriggerSend(self.n, self.m)
        self.TriggerRecieve = TriggerRecieve(self.n, self.m)

        self.trigger_x_ij = [[] for i in range(self.n)]
        self.trigger_v_ij = [[] for i in range(self.n)]
        self.initial_state()

        # Initialization of the state and the auxiliary variable
    def initial_state(self):
        # Create an inital state
        self.x_i = 1 * np.random.randn(self.m) # 状態
        np.save('x_i_init.npy', self.x_i)

        # Load the inital state
        self.x_i = np.load('x_i_init.npy')
        #self.x_i = 0.5 * np.ones(self.m)
        # self.x_i = np.zeros(self.m)
        self.x = np.zeros([self.n, self.m])
        self.v_i = self.grad()  # 補助変数
        self.v = np.zeros([self.n, self.m])
        #self.threshold_x = 1.2 * np.ones([self.n, 1])
        #self.threshold_v = 1.2 * np.ones([self.n, 1])
        #self.threshold_x = 20 * np.ones([self.n, 1])
        #self.threshold_v = 20 * np.ones([self.n, 1])
        # self.threshold_x = 15 * np.ones([self.n, 1])
        # self.threshold_v = 15 * np.ones([self.n, 1])
        #self.threshold_x = np.zeros([self.n, 1])
        #self.threshold_v = np.zeros([self.n, 1])
        self.tilde_x_ij = np.kron((self.x_i + self.threshold_x), np.ones([self.n, 1]))  # 初期時刻では全エージェントのトリガがかかるように設定
        self.tilde_v_ij = np.kron((self.v_i + self.threshold_v), np.ones([self.n, 1]))

        self.trigger_count = 0
        self.neighbor_agents = np.sum(np.sign(self.weight)) - 1

    # Computation of the thresholds
    def comp_threshold(self, k, n):
        self.ax = 0.9999
        self.av = 0.9999
        #self.b = 0.975
        #self.c = (self.a-self.b)/n
        for j in range(n):
            #self.threshold_x[j] = 1 / (k+100)
            #self.threshold_v[j] = 1 / (k+100)
            self.threshold_x[j] = self.threshold_x[j] * self.ax
            self.threshold_v[j] = self.threshold_v[j] * self.av
            #self.threshold_x[j] = self.threshold_x[j] * (self.b + self.c * j)
            #self.threshold_v[j] = self.threshold_v[j] * (self.b + self.c * (n-j))

    def grad(self):
        g = 0
        for l in range(len(self.A_i)):
            x = np.dot(self.x_i, self.A_i[l])
            #            if self.b_i[l] == 1:
            #               print((np.exp(np.clip(x,-744, 709)) / (1 + np.exp(np.clip(x,-744, 709)))))
            #y = ((np.exp(np.clip(x, -744, 709)) / (1 + np.exp(np.clip(x, -744, 709)))) - self.b_i[l])
            y = np.exp(x) / (1 + np.exp(x)) - self.b_i[l]
            # if y == 0:
            #     y = 10 ** -323
            g += y * self.A_i[l]
        g = g + self.lam * self.x_i
        return g


    # # Computation of the gradient (least mean squares method)
    # def grad(self):
    #     A_to = self.A.T
    #     grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.b))
    #     return grad

    def send(self, j):
        if self.weight[j] == 0:
            return None, j
        else:
            self.TriggerSend.x_v_encode(self.x_i, self.tilde_x_ij, self.threshold_x, self.v_i, self.tilde_v_ij, self.threshold_v, j)
            state, name = self.TriggerSend.send_x_ij_v_ij(j, self.name)

            if self.weight[j] != 0:
                temp = self.TriggerSend.send_trigger_x_ij_v_ij(j)
                if temp[0] != 0:
                    self.trigger_x_ij[j].append(j+1)
                    self.trigger_count += 1
                elif temp[0] == 0:
                    self.trigger_x_ij[j].append(-1)

                if temp[1] != 0:
                    self.trigger_v_ij[j].append(j+1)
                elif temp[1] == 0:
                    self.trigger_v_ij[j].append(-1)
            else:
                self.trigger_x_ij[j].append(None)
                self.trigger_v_ij[j].append(None)
            return state, name

    def receive(self, x_j, name):
        if x_j is None:
            pass
        else:
            self.TriggerRecieve.get_x_ji_v_ji(x_j, name)

    def update(self, k):
        self.x_ij, self.v_ij = self.TriggerSend.send_x_ij_v_ij_all()
        self.x_ji, self.v_ji = self.TriggerRecieve.send_x_ji_v_ji()
        x = self.x_ji - self.x_ij
        x[self.name] = np.zeros(self.m)
        v = self.v_ji - self.v_ij
        v[self.name] = np.zeros(self.m)

        grad_bf = self.grad()
        self.x_i = self.x_i + np.dot(self.weight, x) - self.eta * self.v_i
        self.v_i = self.v_i + np.dot(self.weight, v) + (self.grad() - grad_bf)

        self.comp_threshold(k, self.n)

        #return self.x_i

    def trigger_x_ij_v_ij(self):
        return self.trigger_x_ij, self.trigger_v_ij

    # def optimal(self):
    #     x = 0
    #     y = 0
    #     for l in range(len(self.A_i)):
    #         if self.b_i[l] == 1:
    #             x += np.dot(self.x_i, self.A_i[l])
    #
    #         x2 = np.dot(self.x_i, self.A_i[l])
    #         #y += np.log(1 + np.exp(np.clip(x2, -744, 709)))
    #         y += np.log(1 + np.exp(x2))
    #         y += self.lam * np.linalg.norm(self.x_i) ** 2 / 2
    #
    #     L = y - x
    #     #print(x, y, L)
    #     return L

class TriggerSend(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.tilde_x_ij_send = np.zeros([n, m])
        self.tilde_v_ij_send = np.zeros([n, m])
        self.x_ij_trigger = np.zeros([n, 1])
        self.v_ij_trigger = np.zeros([n, 1])
        #self.x_ij_trigger_send = np.zeros([n, 1])
        #self.v_ij_trigger_send = np.zeros([n, 1])

    def x_v_encode(self, x_i, tilde_x_ij, threshold_x, v_i, tilde_v_ij, threshold_v, j):
        self.x_ij_trigger[j] = 0
        self.v_ij_trigger[j] = 0
        if np.linalg.norm(x_i - tilde_x_ij[j], 2) > threshold_x[j] or np.linalg.norm(v_i - tilde_v_ij[j], 2) > threshold_v[j]:
            tilde_x_ij[j] = x_i
            tilde_v_ij[j] = v_i
            self.x_ij_trigger[j] = 1
            self.v_ij_trigger[j] = 1
        self.tilde_x_ij_send[j] = tilde_x_ij[j]
        self.tilde_v_ij_send[j] = tilde_v_ij[j]

    def x_encode(self, x_i, tilde_x_ij, threshold_x, j):
        self.x_ij_trigger[j] = 0
        if np.linalg.norm(x_i - tilde_x_ij[j], 2) > threshold_x[j]:
            tilde_x_ij[j] = x_i
            self.x_ij_trigger[j] = 1
        #else:
            #self.x_ij_trigger[j].append(0)
            #self.x_ij_trigger[j] = 0
        self.tilde_x_ij_send[j] = tilde_x_ij[j]

    def v_encode(self, v_i, tilde_v_ij, threshold_v, j):
        self.v_ij_trigger[j] = 0
        if np.linalg.norm(v_i - tilde_v_ij[j], 2) > threshold_v[j]:
            tilde_v_ij[j] = v_i
            self.v_ij_trigger[j] = 1

        self.tilde_v_ij_send[j] = tilde_v_ij[j]

    def send_trigger_x_ij_v_ij(self, j):
        return self.x_ij_trigger[j], self.v_ij_trigger[j]

    def send_x_ij_v_ij(self, j, name):
        return (self.tilde_x_ij_send[j], self.tilde_v_ij_send[j]), name

    def send_x_ij_v_ij_all(self):
        return self.tilde_x_ij_send, self.tilde_v_ij_send


class TriggerRecieve(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        #self.x_D = np.zeros([n, m])
        #self.v_D = np.zeros([n, m])

        self.tilde_x_ij = np.zeros([n, m])
        self.tilde_v_ij = np.zeros([n, m])

    def get_x_ji_v_ji(self, state, j):
        #print('\n A')
        #print(j)
        self.tilde_x_ij[j] = state[0]
        self.tilde_v_ij[j] = state[1]

    def send_x_ji_v_ji(self):
        return self.tilde_x_ij, self.tilde_v_ij



##=======================================================================================================##
class Agent_subgradient(object):
    def __init__(self, n, m, A, b, s, R, weight, name):
        super(Agent_subgradient, self).__init__(n, m, A, b, weight, name)
        self.step = s
        self.R = R
        
    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x_j = np.zeros((self.n, self.m))
        self.x_i = np.random.rand(self.m)
#        print(self.x_i)


    def s(self, k):  # ステップ幅
        return self.step / (k + 1) ** 0.51

    def send(self, j):
        if self.weight[j] == 0:
            return None, self.name
        else:
            return self.x_i, self.name

    def receive(self, x_j, name):
        self.x_j[name] = x_j

    def grad(self):
        A_T = self.A_i.transpose()
        return 2.0 * np.dot(A_T, (np.dot(self.A_i, self.x_i) - self.b_i))

    def P_X(self, x):
        if np.linalg.norm(x) <= self.R:
            #             print(np.linalg.norm(x-self.c))
            return x
        else:
            # print('P')
            return self.R * x / np.linalg.norm(x)

    def update(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.weight[j] * (self.x_j[j] - self.x_i)
        z_i = self.x_i + sum - (self.s(k) * self.grad())
        self.x_i = self.P_X(z_i)


class Agent_subgradient_self_trigger(Agent_subgradient):
    def __init__(self, n, m, A, b, s, R, weight, name, th_pa):
        super(Agent_subgradient_self_trigger, self).__init__(n, m, A, b, s, R, weight, name)
        self.th_pattern = th_pa
        
    def initial_state(self):
        self.x_i = np.zeros(self.m)
        np.random.seed(0)
        self.x_i = np.random.rand(self.m)
        self.x_j = np.zeros((self.n, self.m))
        self.tildex_i = np.zeros((self.m))
        self.tildex_j = np.zeros((self.n, self.m))
        self.self_trigger_check = 1
        self.trigger_interval = 0
        self.trigger_count = 0
        self.neighbor_agents = np.sum(np.sign(self.weight)) - 1
        self.stopcheck = 0
        self.ave_f = []

    def threshold(self, k):
        if self.th_pattern == 0:
            return 0.0

        elif self.th_pattern == 1:
            return 100.0/((k + 1) ** 0.1)

        elif self.th_pattern == 2:
            return 50.0/((k + 1) ** 0.1)

    def T_i_count(self):
        if self.trigger_interval == 0:
            self.self_trigger_check = 1
        else:
            self.trigger_interval -= 1
            self.self_trigger_check = 0
    
    def trigger_estimate(self, k):
        self.tau_i = 5
        sum = 0.0
        for j in range(self.n):
            sum += self.weight[j] * np.linalg.norm(self.tildex_j[j] - self.tildex_i)
            T = np.linalg.norm(self.grad()) * self.s(k)
            if T == 0:
                T = 10 ** -323
        self.T_i = min(np.ceil(self.threshold(k + self.tau_i) / (2.0 * (sum + T))), self.tau_i)
        if self.T_i == 0.0:#E(k + tau_i, j) = 0 の場合
            self.T_i = 1
        self.trigger_interval = self.T_i - 1
    
    def send(self, j):
        if self.weight[j] == 0:
            return None, self.name
        else:
            if self.self_trigger_check == 1:
                self.trigger_count += 1
                self.tildex_i = self.x_i
                return self.tildex_i, self.name
            else:
                return None, self.name
    
    def receive(self, x_j, name):
        if x_j is None:
            return
        else:
            self.tildex_j[name] = x_j

    def update(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.weight[j] * (self.tildex_j[j] - self.tildex_i)
        z_i = self.x_i + sum - (self.s(k) * self.grad())
        self.x_i = self.P_X(z_i)
        if self.self_trigger_check == 1:
            self.trigger_estimate(k)
        self.T_i_count()

#     def centlized_solve(self, m, A, b, R):
#         x = cvx.Variable(m)
#         y1 = 0
#         x1 = 0
#         for l in range(len(A)):
#             if b[l] == 1:#target = 1
#                 x1 += x.T * A[l]
#             y1 += cvx.logistic(x.T * A[l])#all target
# #            print(x2)
# #            print(cvx.exp(x2))
#         y = y1 - x1
#         obj = cvx.Minimize(y)
#         constrains = [cvx.norm(x, 2) <= R]
#         pro = cvx.Problem(obj, constrains)
#         pro.solve(verbose = True)
#         return x.value, pro.value
    
    def optimal(self):
        x = 0
        y = 0
        for l in range(len(self.A_i)):
#            print("l = %d" % l)
            if self.b_i[l] == 1:
                x += np.dot(self.x_i, self.A_i[l])
#                print(x)
            x2 = np.dot(self.x_i, self.A_i[l])
            y += np.log(1 + np.exp(np.clip(x2, -744, 709)))
        L = y - x
#        print(x, y, L)
        return L        


class Agent_subgradient_event_trigger(Agent_subgradient):
    def __init__(self, n, m, A, b, s, R, weight, name, th_pa):
        super(Agent_subgradient_event_trigger, self).__init__(n, m, A, b, s, R, weight, name)
        self.th_pattern = th_pa
        
    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x_j = np.zeros((self.n, self.m))
        self.x_i = np.random.rand(self.m)
        self.tildex_i = np.zeros((self.m))
        self.tildex_j = np.zeros((self.n, self.m))
        self.trigger_check = 1
        self.trigger_count = 0
        self.neighbor_agents = np.sum(np.sign(self.weight)) - 1
        self.ave_f = []
        self.stopcheck = 0
        
    def threshold(self, k):
        if self.th_pattern == 0:
            return 0

        elif self.th_pattern == 3:
            return 1.0/((k + 1) ** 0.8)

        elif self.th_pattern == 4:
            return 40.0/((k + 1) ** 1)


    def trigger_judge(self, k):
        for j in range(self.n):
            if self.weight[j] > 0:
                if np.linalg.norm(self.x_i - self.tildex_i) >= self.threshold(k):
                    self.trigger_check = 1
                else:
                    self.trigger_check = 0               
                    
    def send(self, j):
        if self.weight[j] == 0:
            return None, self.name
        else:
            if self.trigger_check == 1:
                self.tildex_i = self.x_i
                self.trigger_count += 1
                return self.tildex_i, self.name
            else:
                return None, self.name

    def receive(self, x_j, name):
        if x_j is None:
            return
        else:
            self.tildex_j[name] = x_j

    def update(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.weight[j] * (self.tildex_j[j] - self.tildex_i)
        z_i = self.x_i + sum - (self.s(k) * self.grad())
        self.x_i = self.P_X(z_i)
        self.trigger_judge(k + 1)
        
#     def centlized_solve(self, m, A, b, R):
#         x = cvx.Variable(m)
#         y1 = 0
#         x1 = 0
#         for l in range(len(A)):
#             if b[l] == 1:#target = 1
#                 x1 += x.T * A[l]
#             y1 += cvx.logistic(x.T * A[l])#all target
# #            print(x2)
# #            print(cvx.exp(x2))
#         y = y1 - x1
#         obj = cvx.Minimize(y)
#         constraints = [cvx.norm(x, 2) <= R]
#         pro = cvx.Problem(obj,constraints)
#         pro.solve(verbose = True)
#         return x.value, pro.value
    
    def optimal(self):
        x = 0
        y = 0
        for l in range(len(self.A_i)):
#            print("l = %d" % l)
            if self.b_i[l] == 1:
                x += np.dot(self.x_i.T, self.A_i[l])
            y += np.log(1 + np.exp(np.dot(self.x_i.T, self.A_i[l])))
        L = y - x
#        print(x, y, L)
        return L


