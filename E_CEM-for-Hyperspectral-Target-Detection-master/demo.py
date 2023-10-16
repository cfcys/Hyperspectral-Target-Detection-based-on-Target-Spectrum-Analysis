#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Feb 2019 @author: Rui ZHAO
The following code shows how to run the E-CEM detector of our paper:
R Zhao, Z Shi, Z Zou, Z Zhang, Ensemble-Based Cascaded Constrained Energy Minimization for Hyperspectral Target Detection. Remote Sensing 2019.

"""

from utils import Data
from e_cem import ECEM


def main():
    data = Data('hyperspectral_data//san.mat')  # load data
    ecem = ECEM()
    ecem.parmset(**{'windowsize': [1/4, 2/4, 3/4, 4/4],   # window size 窗口大小
                    'num_layer': 10,  # the number of detection layers 检测层数
                    'num_cem': 6,  # the number of CEMs per layer   
                    'Lambda': 1e-6,  # the regularization coefficient
                    'show_proc': True})  # show the process or not
    result = ecem.detect(data, pool_num=4)  # detection (we recomemend to use multi-thread processing to speed up detetion)
    # data 与 检测装置是分离的，detect函数将他们连接起来
    ecem.show([result], ['E-CEM'])  # show


if __name__ == '__main__':
    main()
