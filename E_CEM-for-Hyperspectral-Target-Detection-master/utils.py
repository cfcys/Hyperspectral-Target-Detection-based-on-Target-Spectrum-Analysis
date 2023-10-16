#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Some useful tools.
"""
Created on Oct 2018 @author: Rui ZHAO
The following code builds some useful classes and tools.

"""

import os.path as op
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import math


class Data(object):    # 数据类
    # snr 信噪比 指一个电子设备或者电子系统中信号与噪声的比例。
    def __init__(self, filepath, snr=None):
        self.name = op.splitext(op.split(filepath)[1])[0]     # 分离文件名与拓展名
        if snr: 
            self.name = self.name+'_noise'
        self.data = scio.loadmat(filepath)  # 加载数据
        self.img = np.array(self.data['X'], dtype=np.float64)  # image  图像
        self.tgt = np.array(self.data['d'], dtype=np.float64)  # target 目标
        self.grt = np.array(self.data['groundtruth'], dtype=np.float64)  # groundtruth 真实值
        if snr:
            self.add_noise(snr)  # 如果snr为真，那么直接去添加噪音

    def add_noise(self, snr):              # 添加噪声
        for i in range(self.img.shape[1]):   
            self.img[:, i] += wgn(self.img[:, i], snr)


class Detector(object):

    def __init__(self):
        self.data = []

    def load_data(self, img_data):   # 加载数据
        self.data = img_data
        self.name = img_data.name
        self.img = img_data.img
        self.tgt = img_data.tgt
        self.grt = img_data.grt

    def show(self, results, names):
        imgshow = [self.img[1,].reshape(self.grt.shape, order='F'), self.grt]
        # self.img[1,] mat的X的第二行的数据
        # .reshape(self.grt.shape, order='F') 按照列的方式排列成真实值的状态
        # imgshow中相当于有一行的图，一张目标的值
        nameshow = ['image(first band)', 'groundtruth'] + names
        # 展示的图片的名字，分别是跑出来的三张图片的效果
        for item in results:
            imgshow.append(item.reshape(self.grt.shape, order='F'))
        k = math.ceil(len(imgshow) / 3) * 100 + 31
        for i in range(len(imgshow)):  # show image
            plt.subplot(k + i)
            plt.axis('off')
            plt.imshow(imgshow[i], cmap='gray')
            plt.title(nameshow[i])
        plot_ROC(self.grt.reshape(-1, 1, order='F'), results, names)  # plot ROC curve
        return 0
        # 更改维度为1列 -1表示自动适应 order='F'表示按照列的顺序

def wgn(x, snr):
    # add white Gaussian noise to x with specific SNR
    # 在特定的信噪比下对x添加高斯白噪声
    snr = 10**(snr/10)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def dual_sigmoid(x):
    x = np.array(x)
    weights = 1.0 / (1.0 + np.exp(-x))
    return weights


def plot_ROC(test_labels, resultall, name):
    plt.subplots(num='ROC curve', figsize = [10,7])
    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(
         test_labels, resultall[i], pos_label=1)  # 计算误报率和检测概率 caculate False alarm rate and Probability of detection
        auc = "%.5f" % metrics.auc(fpr, tpr)     # caculate AUC (Area Under the Curve)
        print('%s_AUC: %s'%(name[i],auc))
        if not i: my_plot = plt.semilogx if metrics.auc(fpr, tpr) > 0.9 else plt.plot
        my_plot(fpr, tpr, label=name[i])
    plt.xlim([1e-5, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', facecolor='none', edgecolor='none')
    plt.title('ROC Curve')
    plt.show()   # show ROC curve
    return 0 
