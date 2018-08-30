#!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

import tensorflow as tf
import numpy as np

x_data = np.random.randn(5,10)
w_data = np.random.randn(10,3)

with tf.Graph().as_default():
	x = tf.placeholder(tf.float32,shape=(5,10))
	w = tf.placeholder(tf.float32,shape=(10,3))
	
	b = tf.fill((5,3),-1.) #定义一个 5 X 3 的矩阵，并用-1.0填充它
	xw = tf.matmul(x,w)
	
	xwb = xw + b
	s = tf.reduce_max(xwb) #求矩阵xwb中的最大值
	with tf.Session() as sess:
		outs = sess.run(s, feed_dict={x: x_data, w: w_data})
		
print('outs:{}'.format(outs))	


