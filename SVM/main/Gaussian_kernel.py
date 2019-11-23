import numpy as np
import math
from sklearn.base import BaseEstimator


class GuassianKernelSVC(BaseEstimator):
    # def fit(self, X, y):
    #     self.coef_ = np.linalg.solve(
    #         np.dot(X.T, X), np.dot(X.T, y)
    #     )
    #     return self

    def predict(self, X, est, u, gamma, name, M):
        nd = len(X)
        gk = np.zeros(M)
        h_hat = np.zeros(nd)

        for i in range(nd):
            # h = 0
        #     for p in range(M):
        #         gk[p] = np.exp(-gamma * (np.linalg.norm(X[i] - u[p])) ** 2)
        #     h = h + np.dot(est[name * M: (name + 1) * M], gk) + est[-1]
            for p in range(M):
                gk[p] = np.exp(-gamma * (np.linalg.norm(X[i] - u[p])) ** 2)
            h = np.dot(est.T, gk)

            h_hat[i] = self.sgnf(h)
        #print(est)
        #print(h)
        #print(h_hat)
        #sys.exit()
        # np.savetxt('x_update.txt', est)
        # np.savetxt('h.txt', h)
        # np.savetxt('h_hat.txt', h_hat)

        return h_hat

    def sgnf(self, x):
        y = -1
        if x >= 0:
            y = 1
        return y
    # print(svm.support_vectors_)
    # print(svm.dual_coef_)
    # print(svm.intercept_)
    # print(svm.gamma)

class KernelSVC(BaseEstimator):
    # def fit(self, X, y):
    #     self.coef_ = np.linalg.solve(
    #         np.dot(X.T, X), np.dot(X.T, y)
    #     )
    #     return self

    def predict(self, X, est, u, gamma, name, M):
        nd = len(X)
        gk = np.zeros(M)
        h_hat = np.zeros(nd)

        for i in range(nd):
            # h = 0
        #     for p in range(M):
        #         gk[p] = np.exp(-gamma * (np.linalg.norm(X[i] - u[p])) ** 2)
        #     h = h + np.dot(est[name * M: (name + 1) * M], gk) + est[-1]
            for p in range(M):
                # gk[p] = np.exp(-gamma * (np.linalg.norm(X[i] - u[p])) ** 2)
                gk[p] = X[p]
            h = np.dot(est.T, gk)

            h_hat[i] = self.sgnf(h)
        #print(est)
        #print(h)
        #print(h_hat)
        #sys.exit()
        # np.savetxt('x_update.txt', est)
        # np.savetxt('h.txt', h)
        # np.savetxt('h_hat.txt', h_hat)

        return h_hat

    def sgnf(self, x):
        y = -1
        if x >= 0:
            y = 1
        return y
    # print(svm.support_vectors_)
    # print(svm.dual_coef_)
    # print(svm.intercept_)
    # print(svm.gamma)