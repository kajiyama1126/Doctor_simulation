# -*- coding: utf-8 -*-

from event_trigger_subgrad.sci_huber_simulation.sci_huber_compare.agent import Agent
from event_trigger_subgrad.sci_huber_simulation.sci_huber_compare.agent_senkou import Agent2
from event_trigger_subgrad.sci_huber_simulation.sci_huber_compare.storage import Make_graph
import numpy as np
import pylab as plt
from numpy import float64, float32, double
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
Optimal_value = 16
minimum_point = [[-3,3],[-2,1],[-2,2],[0,4],[-3,5]]
n = 5
m = 2
set = 1

f_time=[]
f_senkou=[]
f_event=[]

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


#エージェント定義
agent0 = Agent([0,0],n,0,graph0[0])
agent1 = Agent([0,0],n,1,graph0[1])
agent2 = Agent([0,0],n,2,graph0[2])
agent3 = Agent([0,0],n,3,graph0[3])
agent4 = Agent([0,0],n,4,graph0[4])
save_data = Make_graph(n,m)
all_agent =[agent0,agent1,agent2,agent3,agent4]
all_agent_stop = [agent0.stop,agent1.stop,agent2.stop,agent3.stop,agent4.stop]
error = [0 for i in range(n)]
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
for k1 in range(0,1000):
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

    
    if k1 >=1:
        sum3 = 0
        for i in range(n):
            sum3 += f(n,m,all_agent[i].estimate_hat,minimum_point)
        #print sum3/n
        for i in range(n):
            save_data.save_estimate(all_agent[i].estimate_hat,i)
        f_time.append(sum3/n-Optimal_value)
    #print sendcount
    #print k1+1

###############################################################################################

set = 0
if set ==0:
    Agent2.a = 0.2
    Agent2.b = 0.99
else:
    Agent2.a = Agent2.b = 0
Agent2.stepsize =0.02
Agent2.stopsize = 0.0002
graph0 =np.array([[1,1,0,1,0],
                 [1,1,1,0,0],
                 [0,1,1,1,1],
                 [1,0,1,1,1],
                 [0,0,1,1,1]])


#エージェント定義
agent0 = Agent2([0,0],n,0,graph0[0])
agent1 = Agent2([0,0],n,1,graph0[1])
agent2 = Agent2([0,0],n,2,graph0[2])
agent3 = Agent2([0,0],n,3,graph0[3])
agent4 = Agent2([0,0],n,4,graph0[4])
save_data = Make_graph(n,m)
all_agent =[agent0,agent1,agent2,agent3,agent4]
all_agent_stop = [agent0.stop,agent1.stop,agent2.stop,agent3.stop,agent4.stop]
error = [0 for i in range(n)]
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
for k1 in range(0,1000):
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

    
    if k1 >=1:
        sum3 = 0
        for i in range(n):
            sum3 += f(n,m,all_agent[i].estimate_hat,minimum_point)
        #print sum3/n
        for i in range(n):
            save_data.save_estimate(all_agent[i].estimate_hat,i)
        f_senkou.append(sum3/n-Optimal_value)
    #print sendcount
    #print k1+1

##################################################################################################################################################


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


#エージェント定義
agent0 = Agent([0,0],n,0,graph0[0])
agent1 = Agent([0,0],n,1,graph0[1])
agent2 = Agent([0,0],n,2,graph0[2])
agent3 = Agent([0,0],n,3,graph0[3])
agent4 = Agent([0,0],n,4,graph0[4])
save_data = Make_graph(n,m)
all_agent =[agent0,agent1,agent2,agent3,agent4]
all_agent_stop = [agent0.stop,agent1.stop,agent2.stop,agent3.stop,agent4.stop]
error = [0 for i in range(n)]
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
for k1 in range(0,1000):

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

    
    if k1 >=1:
        sum3 = 0
        for i in range(n):
            sum3 += f(n,m,all_agent[i].estimate_hat,minimum_point)
        #print sum3/n
        for i in range(n):
            save_data.save_estimate(all_agent[i].estimate_hat,i)
        f_event.append(sum3/n-Optimal_value)


x=np.arange(2,1001,1)
print (len(f_time))
plt.plot(x,f_time,label='Time-Triggered Algorithm [17]?_',lw='2')
plt.plot(x,f_senkou,'-.',label='Event-Triggered Algorithm [19]?',lw='2')
plt.plot(x,f_event,'--',label = 'Proposed Algorithm',lw='2')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('average cost')
plt.show()

save_data.make_trigger_graph()

# plt.plot(x,f_time,label='Time Trigger Algorithm')
# plt.plot(x,f_senkou,'-.',label='Previous Event Trigger Algortihm')
# plt.plot(x,f_event,'--',label = 'The Proposed Algortihm')
# plt.legend()
# plt.xlabel('iteration')
# plt.xlim([900,1000])
# plt.ylim([16,17.5])
# plt.show()
        
for k in range(n):
    print (all_agent[k].trigger_count)