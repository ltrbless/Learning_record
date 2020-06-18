# -*- coding:utf-8 -*-
# Time: 2020/6/3 22:59
# Author: Taoran Liu
# File: metrics.py.py
# IDE：PyCharm Community Edition

import numpy as np

def accuracy_score(y_true, y_predict):
    """
    计算y_true与y_predict之间的准确率
    :param y_true: 真实的标签
    :param y_predict: 预测的标签
    :return: 准确率
    """
    assert y_true.shape[0] == y_predict.shape[0], "the size of true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """
    求均方误差 MSE
    :param y_ture:
    :param y_predict:
    :return:
    """
    assert len(y_true) == len(y_predict), "ths size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    """求均方根误差：RMSE(Root Mean Squared Error)"""
    assert len(y_true) == len(y_predict), "ths size of y_true must be equal to the size of y_predict"
    return np.sqrt(np.sum((y_true - y_predict) ** 2) / len(y_true))

def mean_absoluate_error(y_true, y_predict):
    """求平均绝对误差：MAE(Mean Absolute Error)"""
    assert len(y_true) == len(y_predict), "ths size of y_true must be equal to the size of y_predict"
    return np.sum( np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    assert len(y_true) == len(y_predict), "ths size of y_true must be equal to the size of y_predict"
    y_mean = np.mean(y_true)
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
    # return 1 - (y_predict - y_true).dot(y_predict - y_true)[0] / (y_mean - y_true).dot(y_mean - y_true)[0]



