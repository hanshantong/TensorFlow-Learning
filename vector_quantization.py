#!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

import numpy as np
from scipy.cluster import vq
import matplotlib.pyplot as plt

#创建数据
c1 = np.random.randn(100, 2) + 5
c2 = np.random.randn(30, 2) - 5
c3 = np.random.randn(50, 2)


#将数据合并成（100+30+50）X 2的矩阵
data = np.vstack([c1, c2, c3])

#使用kmeans计算它们的几何中心和方差
centroids, variance = vq.kmeans(data, 3)

identified, distance = vq.vq(data, centroids)

#获取上述三组数据的坐标
vqc1 = data[identified == 0]
vqc2 = data[identified == 1]
vqc3 = data[identified == 2]

#绘图
plt.scatter(vqc1[:,0],vqc1[:,1])
plt.scatter(vqc2[:,0],vqc2[:,1])
plt.scatter(vqc3[:,0],vqc3[:,1])
plt.show()

