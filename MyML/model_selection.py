# -*- coding:utf-8 -*-
# Time: 2020/6/3 20:27
# Author: Taoran Liu
# File: model_selection.py
# IDE：PyCharm Community Edition

import numpy as np

def train_test_split(X, y, test_ratio=0.2, random_state=None):
    """
    通过train_test_split(X, y, test_ratio=0.2, random_state=None)将数据集切分成测试集比例占 test_ratio。
    :param X: 数据集
    :param y: 数据集的label
    :param test_ratio: 测试集的比例
    :param random_state: 随机种子
    :return:
    """
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_ratio must be vaild"

    if random_state:  # 是否设置随机种子
        np.random.seed(random_state)

    shuffle_index = np.random.permutation(len(X)) # 乱序的索引，从 0  -  len(x)-1

    test_size = int(test_ratio * len(X)) # 测试集大小
    test_indexs = shuffle_index[:test_size] # 得到测试集的索引
    train_indexs = shuffle_index[test_size:]

    X_train = X[train_indexs] # 通过索引得到数据集
    y_train = y[train_indexs]

    X_test = X[test_indexs] # 通过索引得到label
    y_test = y[test_indexs]

    return X_train, X_test, y_train, y_test