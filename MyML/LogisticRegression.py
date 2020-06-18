# -*- coding:utf-8 -*-
# Time: 2020/6/15 16:16
# Author: Taoran Liu
# File: LogisticRegression.py
# IDE：PyCharm Community Edition

import numpy as np
from .metrics import accuracy_score
class LogisticRegression:

    def __init__(self):
        """初始化多元线性回归的参数"""
        self.coefficient_ = None
        self.interception_ = None
        self._theta = None

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def fit(self, X_train, y_train, learn_rate = 0.001, epoch = 10000, epsilon = 1e-8):
        """根据训练集X_train, y_train 来训练逻辑回归模型 By 梯度下降法"""

        def L(theta, X_b, y): # 损失函数
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y*np.log(y_hat) + (1- y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')

        def dL(theta, X_b, y): # 梯度
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

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

    def predict_proba(self, X_predict):
        assert self.interception_ is not None and self.coefficient_ is not None, "must fit before predict !"
        assert X_predict.shape[1] == len(self.coefficient_), "the feature number of X-predict must be equal to X-train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta)) # 相对于线性回归返回的是概率

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coefficient_ is not None, "must fit before predict !"
        assert X_predict.shape[1] == len(self.coefficient_), "the feature number of X-predict must be equal to X-train"
        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype="int")

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def  __repr__(self):
        return "LogisticRegression()"
