import numpy as np

from moment_subgrad.jikken2.paper_CDC_class import new_iteration_L1_paper3, new_iteration_L1_paper,new_Agent_moment_CDC2017_paper2,new_iteration_L1_paper2,new_iteration_L1_paper_powerpoint
from moment_subgrad.jikken2.iteration import new_iteration_L2_harnessing,new_iteration_L2,new_iteration_L1,new_iteration_Dist
from moment_subgrad.jikken2.iteration import sub_new_iteration_L1
if __name__ == '__main__':
    #n = 50
    #m = 20
    n = 6
    m = 10
    #lamb = 0.1
    lamb = 0.001
    R = 4
    np.random.seed(0)  # ランダム値固定
    #pattern = 2
    pattern = 8
    #test = 4000
    test = 50000
    #step = [0.2, 0.2, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0] #<-- IEICE paper

    # #実験2　gamma 比較用
    # step = [0.4, 0.4, 0.4, 0.6, 0.4, 0.8, 0.4, 0.99] #<--IEICE paper
    # step = np.reshape(step,-1)
    # print(n,m,lamb,R,test)
    # if pattern != len(step):
    #     print('error')
    #     pass
    # else:
    #     step = np.reshape(step, -1)
    #     tmp = new_iteration_L1_paper2(n, m, step, lamb, R, pattern, test) #<--IEICE paper

    #実験3 B 比較用
    #step = [0.5, 0.5]
    #step = [1.2, 0.6]
    #step = [1.5, 0.9]
    step = [1.4, 1.0]
    #step = [1.5, 0.7]
    #B = [1, 1, 4, 4, 8, 8] #<--IEICE paper
    #B = [1, 1, 4, 4, 8, 8]  # <--IEICE paper
    B = [1, 1, 2, 2, 3, 3, 5, 5]
    #B = [8, 8]  # <--IEICE paper
    step = np.reshape(step,-1)
    print(n,m,lamb,R,test)
    if pattern != len(B):
        print('error')
        pass
    else:
        #step = np.reshape(step, -1)
        B = np.reshape(B, -1)
        #print(B)
        tmp = new_iteration_L1_paper3(n, m, step, B, lamb, R, pattern, test) #<--IEICE paper