import numpy as np

b = np.array([[1,2,3],[2,3,4]])
c= np.reshape(b,-1)
print(b,c)
# # A = np.array([[1 for i in range(5)] for j in range(100)])
# c = np.ones(20)
# d = np.reshape(c,(20,-1))
# A = np.kron(d,np.identity(5))
#
# print(A)
# # print(d)
# print(np.dot(A,b))
# A = np.array([[[i+j for i in range(3)] for j in range(3)] for k in range(5)])
# # ab = np.dot(A,b)
# # c= np.ones(5)
# # d= np.reshape(c,(5,-1))
# # e = np.kron(b,d)
# # print(c,d,e,ab)
# A1 = np.reshape(A,(-1,3))
# A2 = np.dot(A1,b)
# # print(A,A1,A2)