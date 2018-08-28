#!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(3.0)

c = a * b
d = tf.sin(c)
e = b / d

'''
#创建会话
sess = tf.Session()

#执行
outs = sess.run(e)

print('outs = %s' % (outs))

sess.close()
'''

with tf.Session() as sess:
	fetches = [a, b, c, d, e]
	outs = sess.run(fetches)

print('outs = {}'.format(outs))
print(type(outs[0]))