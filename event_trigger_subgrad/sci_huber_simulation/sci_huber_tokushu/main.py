# -*- coding: utf-8 -*-
from event_trigger_subgrad.sci_huber_simulation.sci_huber_tokushu.agent import Agent
from event_trigger_subgrad.sci_huber_simulation.sci_huber_tokushu.storage import Make_graph
import numpy as np
from numpy import float64, float32, double
import random

def f(n,m,x,minimum):
    fun = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if abs(x[j]-minimum[i][j])<= 5.0:
                fun[i][j] = (x[j]-minimum[i][j])**2.0
            else:
                fun[i][j] = 10*abs(x[j]-minimum[i][j])-25.0
    sum = 0.0
    for i in range(n):
        for j in range(m):
            sum += fun[i][j]
            
    return sum
    
        
    
n = 5
m = 2
set = 1

if set ==0:
    Agent.a = 0.2
    Agent.b = 0.99
else:
    Agent.a = Agent.b = 0
Agent.stepsize =0.02
Agent.stopsize = 0.0002
graph0 =np.array([[1,1,0,1,0],
                 [1,1,1,0,0],
                 [0,1,1,1,1],
                 [1,0,1,1,1],
                 [0,0,1,1,1]])

value_sum = 0
iteration_sum = 0
trigger_sum = 0

sendcount_before= 0
#エージェント定義
for test in range(100):
    agent0 = Agent([0,0],n,0,graph0[0])
    agent1 = Agent([0,0],n,1,graph0[1])
    agent2 = Agent([0,0],n,2,graph0[2])
    agent3 = Agent([0,0],n,3,graph0[3])
    agent4 = Agent([0,0],n,4,graph0[4])
    
    save_data = Make_graph(n,m)
    all_agent =[agent0,agent1,agent2,agent3,agent4]
    all_agent_stop = [agent0.stop,agent1.stop,agent2.stop,agent3.stop,agent4.stop]
    error = [0 for i in range(n)]
    
#     for i in range(n):
#         print 'initial value' ,i+1,all_agent[i].estimate
    for i in range(n):
        save_data.save_estimate(all_agent[i].estimate,i)
        
    
    
    for i in range(n):
        all_agent[i].make_p()#予定重みの作成
    for i in range(n):
        for j in range(n):
            if graph0[i][j] == 1:
                y,name = all_agent[i].p_send(j)
                all_agent[j].p_receive(y,name)
    
    
    for i in range(n):
        all_agent[i].make_weight()#重みの作成
        #print all_agent[i].weight
        
        
        
        
        
    
    #更新式
    for k1 in range(0,2000):
        for i in range(n):
            all_agent[i].send_stop = 0
        for i in range(n):#通信のあるエージェント間での状態の交換
            for j in range(n):
                if graph0[i][j] == 1 and all_agent[i].trigger_check == 1 and i!=j:
                    x,name = all_agent[i].send()
                    all_agent[j].receive(x,name)
        for i in range(n):
            all_agent[i].koushin(n, k1)#解の更新
            error[i] = np.array(all_agent[i].error(k1),double)           
            save_data.save_trigger(i,all_agent[i].trigger_check)
            save_data.save_costfunction(all_agent[i].estimate_hat,i,Agent.minimum_point)
        
        save_data.save_error(error) 
    
        
        if k1 >=2:
            for i in range(n):
                save_data.save_estimate(all_agent[i].estimate_hat,i)
        stopcount = 0
        sendcount = 0
        for i in range(n):
            stopcount += all_agent[i].stop 
            sendcount += all_agent[i].send_stop 
        #print sendcount
        if stopcount==5 and sendcount_before >= 1: 
            #print '反復回数',k1+1
            iteration_sum += k1+1
            break
        
        sendcount_before = sendcount
        
        
        '''        
        if k1%100 == 0:
            print error
            print agent0.E(k1)
        '''
    for i in range(n):
        print ('エージェントの',i+1,'の推定値x_hat',all_agent[i].estimate_hat)
    
    minimum_point = [[-3,3],[-2,1],[-2,2],[0,4],[-3,5]]
    sum3 = 0
    for i in range(n):
        sum3 += f(n,m,all_agent[i].estimate_hat,minimum_point)
    print (sum3/n)
#     sum=0
#     for i in range(n):
#         sum += 5*all_agent[i].estimate_hat[0]**2 + 20*all_agent[i].estimate_hat[0]+5*all_agent[i].estimate_hat[1]**2-30*all_agent[i].estimate_hat[1]+81
#     print sum/n
    value_sum += sum3/n
        
    trigger_sum += all_agent[0].trigger_count
            
#     for k in range(n):
#         print all_agent[k].trigger_count
print( '平均収束値', value_sum/(test+1.0))
print ('平均反復回数' ,iteration_sum/(test+1.0))
print ('平均トリガ回数',trigger_sum/(test+1.0))

#save_data.make_estimate_graph()
save_data.make_trigger_graph()
#save_data.make_error_graph()
#save_data.make_costfunction_graph()