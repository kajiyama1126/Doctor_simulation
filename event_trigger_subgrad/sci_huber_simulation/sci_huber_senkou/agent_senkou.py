# -*- coding: utf-8 -*-
import numpy as np
import copy
import scipy


class Agent: 
    stepsize = float()
    minimum_point = [[-3,3],[-2,1],[-2,2],[0,4],[-3,5]]
    a = float()
    b = float()
    stopsize = float()
    def E(self,k):
        return self.a*self.b**k
    
    
    def __init__(self,estimate,n,agent_number,graph):#初期状態
        self.n = n
        m = len(estimate)
        self.name = agent_number
        self.estimate = np.array(estimate,)
        self.estimate_l = np.array(estimate)
        self.estimate_hat =np.zeros_like(estimate)
        self.estimate_before = np.zeros_like(estimate)
        self.estimate_hat_before = np.zeros_like(estimate)
        self.d = np.array([0 for i in range(m)])
        self.estimate_receive = np.zeros([self.n,m])
        self.weight = np.zeros(n)
        self.psend = np.zeros(n)
        self.preceive = np.zeros(n)
        self.trigger_check = 1
        self.communication_graph = graph
        self.trigger_count=0
        self.minimum = self.minimum_point[agent_number]
        self.agent_receive_check = 0
        self.send_stop = 0
        self.stop = 0
        
        
        
    
    def make_p(self):#予定重みの作成
        self.psend = np.zeros(self.n)
        self.preceive = np.zeros(self.n)
        count = 0
        for i in range(len(self.communication_graph)):
            count += self.communication_graph[i]
        self.psend = [1.0/count for i in range(len(self.communication_graph))]
        #print self.psend
    
    def p_send(self,agent_number):#予定重みの送信
        #print self.count_psend
        return self.psend[agent_number],self.name
        
    
    def p_receive(self,receive,number):#予定重みの受け取り
        self.preceive[number] = receive 
        #print self.preceive
        
        
    def make_weight(self):#重みの作成
        self.weight = [0 for i in range(self.n)]
        sum1 = 0
        #print self.psend,self.preceive
        for i in range(self.n):
            if self.name != i:
                self.weight[i] = min(self.psend[i],self.preceive[i])
                sum1 += self.weight[i]
        self.weight[self.name] =  1.0-sum1          
        #print self.weight
        
 
    def subgradient(self):#劣勾配
        a = np.zeros_like(self.estimate)
        for i in range(len(self.estimate)):
            if abs(self.estimate[i]-self.minimum[i])<= 5.0:
                a[i] = 2.0*(self.estimate[i]-self.minimum[i])
            elif (self.estimate[i]-self.minimum[i]) < -5.0:
                a[i] = -10.0
            elif (self.estimate[i]-self.minimum[i]) > 5.0:
                a[i] = 10.0
        self.d = copy.copy(a)

    def koushin(self,n,k):#エージェントの更新式
        self.stop = 0
        if k == 2:
            self.estimate_hat = self.estimate_before
        if k>= 3:
            self.estimate_hat = (k-2.0)/(k-1.0)*self.estimate_hat + 1.0/(k-1.0)*self.estimate_before 
        self.estimate_before = copy.copy(self.estimate)
        self.subgradient()
        sum1 = np.zeros(2)
        for i in range(n):
            if i !=self.name:
                sum1 += self.weight[i]*self.estimate_receive[i]
                #print self.weight[i]*self.estimate_receive[i]
        self.estimate = sum1 + self.weight[self.name]*self.estimate_l - self.stepsize*self.d
        
        if k>=2:
            if np.linalg.norm(self.estimate_hat - self.estimate_hat_before) <= self.stopsize:
                self.stop = 1 
        if k >=2 :
            self.estimate_hat_before =copy.copy(self.estimate_hat)
        
        #print self.estimate

        
    def send(self):#状態の送信
        self.send_stop = 1
        self.estimate_l = self.estimate
        #self.count_send +=1
        return self.estimate_l,self.name
    
    def receive(self,get_receive,agent_nunbaer):#状態の受け取り
        self.agent_receive_check = 1
        self.estimate_receive[agent_nunbaer] = get_receive
        
    def error(self,k):
        e = np.linalg.norm(self.estimate_l - self.estimate)
        if e >= self.E(k):
            #print self.E(k)
            self.trigger_check = 1
            self.trigger_count +=1
        else:
            self.trigger_check = 0
        return e
    
''' 
    def make_D_graph(self,graph):
        D = np.zeros(graph.shape)
        D1 = np.zeros(len(graph))
        for i in range(len(graph)):
            for j in range(len(graph)):
                D1[i] += graph[i][j]
        D = np.diag(D1)
        return D
    
    def michi(self,graph):
        L = self.make_D_graph(graph)-graph
        I = np.matrix(np.identity(self.n)) 
        P = I-(1.0/self.n)*L
        print P
        for i in range(self.n - 2):
            P = P.dot(P)
            print P
        return P\
'''        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        