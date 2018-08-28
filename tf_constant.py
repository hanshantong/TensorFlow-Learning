#!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

#当导入tensorflow时，默认已经创建了一个计算图，
#之后定义的节点（变量）将自动关联到这个计算图
import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = a * b
e = b + c

f = d - e

#创建会话
sess = tf.Session()

outs = sess.run(f)

#关闭会话
sess.close()

print("outs = {}".format(outs))



