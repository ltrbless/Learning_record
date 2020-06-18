# -*- coding:utf-8 -*-
# Time: 2020/6/4 8:46
# Author: Taoran Liu
# File: SimpleLinearRegression.py
# IDE：PyCharm Community Edition

import numpy as np
from .metrics import r2_score

class SimpleLinearRegression:

    def __init__(self, method="vector"):
        """
        初始化 SimpleLinearRegression1 参数
        :param fun: ["scalar", "vector"] scalar 采用标量的运算方式，vector采用向量的运算方式
        """
        self.a_ = None
        self.b_ = None
        self.method = method

    def fit(self, x_train, y_train):

        assert x_train.ndim == 1, "Simple Linear Regressor can only solve single feature training data"
        assert len(x_train) == len(y_train), "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train) # 求均值
        y_mean = np.mean(y_train)

        num = 0.0 #初始化分子
        d = 0.0 #初始化分母

        if self.method == "scalar":
            for i in range(len(x_train)) :
                num += (x_train[i] - x_mean) * (y_train[i] - y_mean)
                d += (x_train[i] - x_mean) ** 2

        if self.method == "vector":
            num = (x_train - x_mean).dot(y_train - y_mean)
            d = (x_train - x_mean).dot(x_train - x_mean)

        # for x, y in zip(x_train, y_train):
        #     num += (x - x_mean) * (y - y_mean)
        #     d += (x - x_mean) ** 2

        self.a_ = num / d # 最小二乘法的公式去求解 a
        self.b_ = y_mean - self.a_ * x_mean #　把均值带入

        return self

    def predict(self, x_predict):
        """给定待预测数据集 x_predict，返回表示 x_predict的结果向量"""
        assert x_predict.ndim == 1, "Simple Linear Regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x):
        return self.a_ * x + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression1 a = %.2f b = %.2f" % (self.a_, self.b_)