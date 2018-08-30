#!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

import tensorflow as tf

with tf.Graph().as_default():
	c1 = tf.constant(5, dtype=tf.float64, name='c')
	with tf.name_scope('apple'):
		c2 = tf.constant(6, dtype=tf.int16, name='c')
		c3 = tf.constant(7, dtype=tf.int32, name='c')
		
print(c1.name)
print(c2.name)
print(c3.name)