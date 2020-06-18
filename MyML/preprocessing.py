# -*- coding:utf-8 -*-
# Time: 2020/6/3 23:26
# Author: Taoran Liu
# File: preprocessing.py
# IDE：PyCharm Community Edition

import numpy as np

class StandardScaler:

    def __init__(self):
        self.scale_ = None
        self.mean_ = None

    def fit(self, data):

        assert data.ndim == 2, "The dimention of data must be 2"

        self.mean_ = np.array( [ np.mean(data[:, column]) for column in range(data.shape[1]) ] )
        self.scale_ = np.array( [ np.std(data[:, column]) for column in range(data.shape[1]) ] )

        return self

    def transform_mydesign(self, data):
        assert data.ndim == 2, "The dimention of data must be 2"
        assert self.mean_ is not None and self.scale_ is not None, "must fit before transform!"
        assert data.shape[1] == len(self.mean_) and data.shape[1] == len(self.scale_), \
            "the feature number of data must be equal to mean_ and scale_."

        new_data = [ (data[:, column] - self.mean_[column]) / self.scale_[column] for column in range(data.shape[1]) ]
        # print("new_data : \n", new_data)
        return np.array(new_data).T

    def transform(self, data):
        assert data.ndim == 2, "The dimention of data must be 2"
        assert self.mean_ is not None and self.scale_ is not None, "must fit before transform!"
        assert data.shape[1] == len(self.mean_) and data.shape[1] == len(self.scale_), \
        "the feature number of data must be equal to mean_ and scale_."

        resX = np.empty(shape=data.shape, dtype=float)
        for column in range(data.shape[1]):
            resX[:, column] = (data[:, column] - self.mean_[column]) / self.scale_[column]
        return resX

    def __repr__(self):
        return "This StandardScaler object created by ltrbless."


