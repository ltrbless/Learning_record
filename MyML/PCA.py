# -*- coding:utf-8 -*-
# Time: 2020/6/13 17:02
# Author: Taoran Liu
# File: PCA.py
# IDE：PyCharm Community Edition

import numpy as np

class PCA:

    def __init__(self, n_components):
        """
        初始化PCA
        :param n_components: 设置 n_components个主成分。
        """
        assert n_components >= 1, "n_components must be vaild."
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta = 0.01, n_iter = 1e4, epsilon=1e-8):
        """
        获得数据的前 n 个主成分
        :param X: m个样本，每个样本n个特征值
        :param eta: 学习率
        :param n_iter: 迭代次数
        :return: self
        """
        assert self.n_components <= X.shape[1], "n_components must be less than the feature number of X."
        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum(X.dot(w) ** 2) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)  # w = 0,求导一直为0，所以初始的时候不能为w不能为 0 向量

        def direction(w):
            return w / np.linalg.norm(w)  # linalg为numpy得线性代数库，np.linalg.norm(w)求解 w 向量的模。

        def first_component(X, initial_w, eta, n_iter, epsilon):
            w = direction(initial_w)
            while n_iter > 0:
                n_iter -= 1
                gradicent = df(w, X)
                last_w = w
                w = w + eta * gradicent  #
                w = direction(w)  ### 重置成为单位向量
                if abs(f(last_w, X) - f(w, X)) < epsilon:
                    break
            return w


        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iter, epsilon)
            self.components_[i, :] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w


        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1], "invaild"
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_component = %d)" % self.n_components

