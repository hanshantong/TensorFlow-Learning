#!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

import tensorflow as tf 

a = tf.constant(4)
b = tf.constant(5)

c = a * b
d = a + b

f = c + d
e = c - d

g = f / e

#创建会话
sess = tf.Session()

#执行并返回结果
outs = sess.run(g)

#输出结果
print('outs = {}'.format(outs))
sess.close()