# -*- coding:utf-8 -*-
# Time: 2020/6/2 15:12
# Author: Taoran Liu
# File: KNN.py
# IDE：PyCharm Community Edition
import math
import numpy as np
from collections import Counter
from .metrics import accuracy_score

class KNNClassify:

    def __init__(self, k, p=2):
        """
        初始化KNN分类器
        :param k:
        """
        assert k >= 1, "k must be vaild"
        self.k = k
        self.p = p
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """
        训练模型，KNN比较特殊，直接喂数据就形成了模型
        :param X_train: 训练集
        :param y_train: 标签
        :return:
        """
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """
        预测模型
        :param X_predict: 测试集
        :return:
        """
        assert self._X_train.shape[1] == X_predict.shape[1], "the feature number of test must be equal to X_train"
        assert 1 <= self.k <= self._X_train.shape[0], "k must be vaild"
        assert self._X_train.shape[0] == self._y_train.shape[0], "the size of X_train must equal to the size of y_train"

        y_predict = [self._predict(x_predict) for x_predict in X_predict] # 对每一个样本计算预测值
        return np.array(y_predict)

    def _predict(self, test):
        """
        预测每一个样本的值
        :param test: 一个样本的特征向量
        :return: 预测结果
        """
        distance = np.array([math.pow( np.sum((x_train - test) ** self.p), 1.0 /self.p ) for x_train in self._X_train])  # 计算距离，默认使用的是欧拉距离
        nearest = np.argsort(distance)[:self.k]  # 获得前K个最小的距离的索引值
        topK_y = self._y_train[nearest] # 取出来前 k 个最近的
        votes = Counter(topK_y) # 计算投票结果，format：{1:10,2:7,3:20} 答案为 3
        return votes.most_common(1)[0][0] # 取投票最多的

    def score(self, X_test, y_test):
        """根据测试集X_test与y_test来确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(K = %d, p = %d)" % (self.k, self.p)
