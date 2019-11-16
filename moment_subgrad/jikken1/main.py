import numpy as np

from moment_subgrad.jikken1.paper_CDC_class import new_iteration_L1_paper,new_Agent_moment_CDC2017_paper2,new_iteration_L1_paper2,new_iteration_L1_paper_powerpoint
from iteration import new_iteration_L2_harnessing,new_iteration_L2,new_iteration_L1,new_iteration_Dist
from iteration import sub_new_iteration_L1
if __name__ == '__main__':
    #n = 50
    #m = 20
    n = 100
    m = 10
    #lamb = 0.1
    lamb = 0.001
    R = 4
    np.random.seed(0)  # ランダム値固定
    pattern = 8
    test = 4000
    #test = 200
    #step = [0.25, 0.25, 0.5, 0.5, 1., 1., 2., 2.]
    #step = [0.1, 0.1, 0.2, 0.2, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
    #step = [0.15, 0.15, 0.2, 0.2, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]
    #step = [0.15, 0.15, 0.2, 0.2, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    #step = [0.2, 0.2, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0] #<-- IEICE paper

    #実験2　gamma 比較用
    #step = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8]
    step = [0.4, 0.4, 0.4, 0.6, 0.4, 0.8, 0.4, 0.99] #<--IEICE paper
    # step = [2.0,0.2,5.0,0.5,10.0,1.0,20.0,2.0]
    # step = [0.5, 0.5, 0.5, 0.7, 0.5, 0.9, 0.5, 0.99]
    # step = [1.0,0.25,2.0,0.5,5.0,1.0,10.0,2.0]
    # step = [0.25,0.25,0.5,0.5,2.0,2.0,5.0,5.0]
    # step = [[0.1*(i+1),0.1*(i+1)] for i in range(pattern)]
    step = np.reshape(step,-1)
    print(n,m,lamb,R,test)
    if pattern != len(step):
        print('error')
        pass
    else:
        # step = np.array([[0.1 *(j+1) for i in range(2)] for j in range(10)])
        step = np.reshape(step, -1)
        # tmp = new_iteration_Dist(n, m, step, lamb, R, pattern, test)
        #tmp = new_iteration_L1_paper(n, m, step, lamb, R, pattern, test) #<--IEICE paper
        # tmp = new_iteration_L1_paper_powerpoint(n, m, step, lamb, R, pattern, test)
        tmp = new_iteration_L1_paper2(n, m, step, lamb, R, pattern, test) #<--IEICE paper
        # tmp = new_iteration_L2_harnessing(n, m, step, lamb, R, pattern, test)
        # tmp = new_iteration_L1(n, m, step, lamb, R, pattern, test)
        # tmp = new_iteration_Dist(n, m, step, lamb, R, pattern, test)
        # tmp = new_iteration_L2(n, m, step, lamb, R, pattern, test)
        # tmp = sub_new_iteration_L1(n, m, step, lamb, R, pattern, test)