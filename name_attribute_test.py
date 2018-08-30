#!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

import tensorflow as tf

with tf.Graph().as_default():
	c1 = tf.constant(5, dtype=tf.float64, name='c')
	c2 = tf.constant(6, dtype=tf.int32, name='c')
	
print(c1.name)
print(c2.name)