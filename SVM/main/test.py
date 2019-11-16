#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:26:57 2019

@author: ueharakenyuu
"""
import pandas as pd
# import mglearn
# import seaborn as sns
import numpy as np
import matplotlib.pyplot  as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.datasets import load_iris

iris_dataset = load_iris()

############irisのデータの確認として###################
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print("First five columns of data: {}".format(iris_dataset['data'][:5]))
print("Shape of data: {}".format(iris_dataset['data'].shape))

print(iris_dataset['DESCR'][:193] + "\n...")
print("Feature names: {}".format(iris_dataset['feature_names']))
print("First five columns of target: {}".format(iris_dataset['target'][:5]))
print("Filename: {}".format(iris_dataset['filename']))
print("Target names: {}".format(iris_dataset['target_names']))

############pandasを使ったデータ全体の図示###################
# 使うときには，コメントアウトを外して！
"""
iris_dataframe = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
print(iris_dataframe.head())

iris_datalabel = pd.Series(data=iris_dataset.target)
print(iris_datalabel.head())
grr = pd.plotting.scatter_matrix(iris_dataframe, c=iris_datalabel, label=iris_dataset.feature_names, \
                                 figsize=(8,8), marker = 'o', hist_kwds = {'bins':20}, s=60, alpha=.8, cmap = mglearn.cm3)
plt.show()
"""
############seabornを使ったデータ全体の図示###################
# 使うときには，コメントアウトを外して！
"""
iris_dataset = sns.load_dataset("iris")
sns.pairplot(iris_dataset, hue='species', palette="husl").savefig('seaborn_iris.png')
"""
############irisデータの取り込み###################
# 特徴量のセットを変数Xに，ターゲットを変数yに
X = iris_dataset.data
y = iris_dataset.target  # 'setosa'=0 'versicolor'=1 'virginica'=2

X_sl = np.vstack((X[:, :1]))  # sepal lengthのみを取得
X_sw = np.vstack((X[:, 1:2]))  # sepal widthのみを取得
X_pl = np.vstack((X[:, 2:3]))  # petal lengthのみを取得
X_pw = np.vstack((X[:, 3:4]))  # petal widthのみを取得

X = np.hstack((X_sl, X_sw))  # sepal lengthとpetal widthのみを扱う

X = X[y != 2]  # 'virginica'=2を排除
y = y[y != 2]  # 'virginica'=2を排除

scaler = StandardScaler()  # 標準化を行います．
X = scaler.fit_transform(X)

############学習データとテストデータを分ける###################
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train stape: {}".format(X_train.shape))
print("y_train stape: {}".format(y_train.shape))
print("X_test stape: {}".format(X_test.shape))


# ----------グラフ描写用の関数---------------#
def plot_decision_function(model):
    _x0 = np.linspace(-3.0, 3.0, 100)
    _x1 = np.linspace(-3.0, 3.0, 100)
    x0, x1 = np.meshgrid(_x0, _x1)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = model.predict(X).reshape(x0.shape)
    y_decision = model.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap='spring', alpha=0.4)
    plt.contourf(x0, x1, y_decision, levels=10, alpha=0.2)


def plot_dataset(X, y):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "ro", ms=5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "go", ms=5)
    plt.xlabel("sepal_length")
    plt.ylabel("sepal_width")


# ---------------------------------------#

# ----4つのパターンを試す-------------#
plt.figure(figsize=(13, 9))
kernel_names = ['linear', 'rbf', 'poly', 'sigmoid']
i = 0
for kernel_name in kernel_names:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

    svm = SVC(kernel=kernel_name)
    svm.fit(X_train, y_train)

    R = svm.predict(X_test)
    total = len(X_test)
    success = sum(R == y_test)

    plt.subplot(221 + i)
    plot_decision_function(svm)
    plot_dataset(X_train, y_train)
    plt.title("kernel = {}, Accuracy:{:.1f}".format(kernel_name, 100.0 * success / total))
    i += 1

# plt.show()
plt.savefig('output.png', dpi=900)

# ----4つのパターンを試す-------------#