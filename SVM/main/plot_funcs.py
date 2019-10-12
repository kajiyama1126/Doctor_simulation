import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, k, test_idx, resolution, x1_min, x1_max, x2_min, x2_max):
    # setup marker generator and color map
    markers = ('x', 'o', 's', '^', 'v')
    colors = ('b', 'r')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #cmap = 'binary'

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.figure()
    #plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.contour(xx1, xx2, Z, colors='g')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        #plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label="Label {}".format(cl))
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx], label="label {}".format(int(cl)))

    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("result_Gaussian_svm_{}.eps".format(k))
    plt.savefig("result_Gaussian_svm_{}.png".format(k))

#def plot_decision_regions2(X, y, classifier, k, resolution=0.01):
def plot_decision_regions2(X, y, classifier, k, est, u, gamma, name, M, resolution, x1_min, x1_max, x2_min, x2_max):
    # setup marker generator and color map
    markers = ('x', 'o', 's', '^', 'v')
    colors = ('b', 'r')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #cmap = 'binary'

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T, est, u, gamma, name, M)
    Z = Z.reshape(xx1.shape)

    plt.figure()
    #plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    #plt.contour(xx1, xx2, Z, cmap='gray_r')
    plt.contour(xx1, xx2, Z, colors='g')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        #plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label="Label {}".format(cl))
        # plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx], label="label {}".format(int(cl)))
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],  marker=markers[idx], label="label {}".format(int(cl)))

    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("result_Gaussian_svm_{}.eps".format(k+1))
    plt.savefig("result_Gaussian_svm_{}.png".format(k+1))


# def plot_decision_regions3(X, y, classifier, k, est, u, gamma, resolution, x1_min, x1_max, x2_min, x2_max):
#     # setup marker generator and color map
#     markers = ('x', 'o', 's', '^', 'v')
#     colors = ('b', 'r')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#     #cmap = 'binary'
#
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     #Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T, est, u, gamma, 0, m)
#     Z = Z.reshape(xx1.shape)
#
#     plt.figure()
#     #plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     #plt.contour(xx1, xx2, Z, cmap='gray_r')
#     plt.contour(xx1, xx2, Z, colors='g')
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     for idx, cl in enumerate(np.unique(y)):
#         #plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label="Label {}".format(cl))
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx], label="label {}".format(int(cl)))
#
#     plt.xlim([x1_min, x1_max])
#     plt.ylim([x2_min, x2_max])
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.savefig("result_Gaussian_svm_{}.eps".format(k+1))
#     plt.savefig("result_Gaussian_svm_{}.png".format(k+1))