# -*- coding:utf-8 -*-
# Time: 2020/6/12 10:10
# Author: Taoran Liu
# File: LinearRegression.py
# IDE：PyCharm Community Edition


import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化多元线性回归的参数"""
        self.coefficient_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练集X_train, y_train 来训练多元线性回归模型 By 正规方程解"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train is the size of y_train"
        X_b = np.hstack( [np.ones( (len(y_train), 1) ) , X_train] )
        self._theta = np.linalg.inv( X_b.T.dot(X_b) ).dot(X_b.T).dot(y_train)
        self.coefficient_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self

    def fit_gd(self, X_train, y_train, learn_rate = 0.001, epoch = 10000, epsilon = 1e-8):
        """根据训练集X_train, y_train 来训练多元线性回归模型 By 梯度下降法"""

        def L(theta, X_b, y): # 损失函数
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dL(theta, X_b, y): # 梯度
            # print(X_b.dot(theta))
            gradient = X_b.T.dot(X_b.dot(theta) - y)
            return gradient * 2 / len(y)

        def gradient_decend(X_b, y, initial_theta, learn_rate, epoch, epsilon):
            theta = initial_theta
            while epoch > 0:
                epoch -= 1
                gradient = dL(theta, X_b, y)  # 求梯度
                last_theta = theta  # 记录上一次的 Theta
                theta = theta - learn_rate * gradient  # 去梯度的反方向
                # print(theta.shape)
                if np.absolute(L(last_theta, X_b, y) - L(theta, X_b, y)) < epsilon:
                    break
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 构造X_b矩阵
        initial_theta = np.zeros(  X_b.shape[1] ) # 初始化theta值
        self._theta = gradient_decend(X_b, y_train, initial_theta, learn_rate, epoch, epsilon)
        self.coefficient_ = self._theta[1:] # 系数
        self.interception_ = self._theta[0] # 截距

        return self

    def fit_sgd(self, X_train, y_train, epochs = 5, t0 = 5., t1 = 50.):
        """
        随机梯度下降法，每个batch为1
        :param X_train: 训练集
        :param y_train: 训练集label
        :param epochs: 轮数 默认 5
        :param t0: 默认 5.
        :param t1: 默认 50.
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert epochs >= 1, "the number of epochs is Error"

        def dL_sgd(theta, X_b_i, y_i):  # 梯度

            gradient = X_b_i.T.dot(X_b_i.dot(theta) - y_i)
            # print(gradient)
            return gradient * 2.

        def sgd(X_b, y, initial_theta, epochs, t0, t1):
            def learning_rate(t): # 动态控制学习率，使其越来越小，防止收敛之后又出去太多
                return t0 / (t + t1)
            theta = initial_theta
            m = len(y) # 样本的个数
            for epoch in range(epochs): # 轮数
                indexs = np.random.permutation(m) #索引乱序
                X_b_new = X_b[indexs] # 重新构造索引
                y_new = y[indexs] #
                for i in range(m): #
                    gradient = dL_sgd(theta, X_b_new[i], y_new[i])  # 求梯度
                    theta = theta - learning_rate(epoch * m + i) * gradient  # 去梯度的反方向
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 构造X_b矩阵
        initial_theta = np.zeros(X_b.shape[1])  # 初始化theta值
        self._theta = sgd(X_b, y_train, initial_theta, epochs, t0, t1)
        self.coefficient_ = self._theta[1:]  # 系数
        self.interception_ = self._theta[0]  # 截距

        return self

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coefficient_ is not None, "must fit before predict !"
        assert X_predict.shape[1] == len(self.coefficient_), "the feature number of X-predict must be equal to X-train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def  __repr__(self):
        return "LinearRegression()"


