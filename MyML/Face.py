# -*- coding:utf-8 -*-
# Time: 2020/6/14 20:38
# Author: Taoran Liu
# File: Face.py
# IDE：PyCharm Community Edition

import numpy as np
import os
import cv2
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.decomposition import PCA

class Face:

    """
    将带不同的人脸放入不同的文件夹中，文件夹的名称为人脸真实名，并且把这些文件夹都放在 train 文件夹下
    将带预测的人脸放入到一个文件夹中，如果想计算模型得分，则需要把照片名称命名为真实的人脸的名字，并且要与 之前train中的文件夹得名一致
    """

    def __init__(self, size = (480, 640)):

        self.size = size # 图片大小  (宽，高)
        # self.n_component = n_component
        self.knn_clf = KNeighborsClassifier()
        self.pca = None

        self.names = None


    def load_train_images(self, floder_path):

        images = []
        labels = []
        names = []
        label = 0

        for path in floder_path:
            subdirname = path.split('\\')[-1]
            names.append(subdirname)
            for filename in os.listdir(path):
                imgpath = os.path.join(path, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)  # 读取灰度图片
                images.append(img)
                labels.append(label)
            label += 1

        # for subdirname in os.listdir(floder_path):  # 读取路径下所有的文件名称
        #     subjectpath = os.path.join(floder_path, subdirname)  # 合并成一个整个的文件名
        #     if os.path.isdir(subjectpath):  # os.path.isdir()用于判断对象是否为一个目录　
        #         names.append(subdirname)
        #         for filename in os.listdir(subjectpath):
        #             imgpath = os.path.join(subjectpath, filename)
        #             img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)  # 读取灰度图片
        #             images.append(img)
        #             labels.append(label)
        #         label += 1
                #     print(images)
        images = np.array(images)  # 将列表转到数组，一张图片对应数组一行
        labels = np.array(labels)
        names = np.array(names)
        self.names = names # 保留真实姓名
        images = self._prepare(images)
        return images, labels # images:[样本数m,像素的高度,像素的宽度] -> X_train  label：编码标签 -> y_train name:人物名字

    def load_test_images(self, floder_path):
        images = []
        names = []

        for subdirname in os.listdir(floder_path):  # 读取路径下所有的文件名称
            subjectpath = os.path.join(floder_path, subdirname)  # 合并成一个整个的文件名
            split_name = subdirname.split('.')[0] # 去掉文件类型
            names.append(split_name)
            img = cv2.imread(subjectpath, cv2.IMREAD_GRAYSCALE)  # 读取灰度图片
            images.append(img)
        images = np.array(images)  # 将列表转到数组，一张图片对应数组一行
        names = np.array(names)
        images = self._prepare(images)
        return images, names  # images:[样本数m,像素的高度,像素的宽度] -> X_train  label：编码标签 -> y_train name:人物名字

    def _prepare(self, images):
        '''
        图片的预处理，直方图均衡化
        images：训练集数据，灰度图片
        [m,height,width] m样本数 height高width宽
        return 处理后的数据[m,n]
        特征数 n = size[0] * size[1]
        '''
        new_images = []
        for image in images:
            re_img = cv2.resize(image, self.size )  # (宽，高)
            # 直方图均衡化  一般情况下直方图都是灰度图像，直方图x轴是灰度值（一般0~255），y轴就是图像中每一个灰度级对应的像素点的个数。
            # 直方图均衡化是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。
            # 这种方法对于背景和前景都太亮或者太暗的图像非常有用，这种方法尤其是可以带来X光图像中更好的骨骼结构显示以及曝光过度或者曝光不足照片中更好的细节。
            hist_img = cv2.equalizeHist(re_img)
            hist_img = re_img.reshape(1, -1)[0]  # 数组的一行是一个人脸，维度为 ： 宽 * 高
            new_images.append(hist_img)
        new_images = np.asarray(new_images)  # 列表变为数组
        return new_images

    def fit(self, X_train, y_train):

        # X_train = self._prepare(X_train)
        self.pca = PCA(0.9)  # 初始化的时候指定
        self.pca.fit(X_train)
        X_train_reduce = self.pca.transform(X_train)
        self.knn_clf.fit(X_train_reduce, y_train)

        return self

    def predict(self, X_test):
        # X_test = self._prepare(X_test)
        X_test_reduce = self.pca.transform(X_test)
        res = self.knn_clf.predict(X_test_reduce)
        return res

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)

        all = len(y_test)
        right = np.sum(y_test == y_predict)

        return right / all




