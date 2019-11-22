# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt

class Make_graph:
    color =['blue','red','green','cyan','black']
    agent_name=['agent1','agent2','agent3','agent4','agent5']
    def __init__(self,n,m):
        self.estimate_save = [[[]for j in range(m)]for i in range(n)]
        #print self.estimate_save
        self.n = n
        self.m = m
        self.count_trigger = [[]for i in range(n)]
        self.error_save = [[]for i in range(n)]
        self.costfunction = [[]for i in range(n)]
        
    def save_estimate(self,estimate,name):
        for i in range(self.m):
            self.estimate_save[name][i].append(estimate[i])
    def make_estimate_graph(self):
        graph_label = ['x1','x2','x3','x4','x5']
        line = ['^','--','-.',':','.']
        for i in range(self.n):
                plt.plot(self.estimate_save[i][0],line[i],c = self.color[i],lw = 2 ,label=graph_label[i])
        plt.legend(loc="center right")
        plt.xlabel("iteration")
        plt.show()
            
    def save_trigger(self,name,trigger):
        if trigger == 1:
            self.count_trigger[name].append(50-10*name)
        else:
            self.count_trigger[name].append(-50)
            
    def make_trigger_graph(self):
        for i in range(self.n):
            plt.plot(self.count_trigger[i],'o',c = self.color[i], label=self.agent_name[i])
        plt.ylim([0,(self.n + 1)*10])
        plt.xlim([0,1000])
        #plt.legend() 
        plt.xlabel("iteration")
        plt.yticks([(50-10*i) for i in range(self.n)],self.agent_name)
        #plt.tick_params(labelleft='off')    
        plt.show()
        
    def save_error(self,e):
        for i in range(self.n):
            self.error_save[i].append(e[i])
            
            
    def make_error_graph(self):
        for i in range(self.n):
            plt.plot(self.error_save[i],c=self.color[i],label=self.agent_name[i])
        plt.legend() 
        plt.xlabel("iteration")
        plt.ylabel("e")
        plt.show()
            
    
    def save_costfunction(self,estimate_hat,name,minimum):
        x1,x2 = 0,0
        for i in range(self.n):
            x1 += (estimate_hat[0]+minimum[i][0])**2
            x2 += (estimate_hat[1]+minimum[i][1])**2
        self.costfunction[name].append(x1+x2)
        
    def make_costfunction_graph(self):
        for i in range(self.n):
            plt.plot(self.costfunction[i],c= self.color[i],label=self.agent_name[i])
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("cost function")
        plt.show()