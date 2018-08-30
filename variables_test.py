  #!/usr/bin/env python3
#-*- coding:utf-8 -*-

__author__ = 'tongzi'

'''
在TensorFlow中，Variables在计算图中保持自己的状态不变，
一般它作为计算图中其它操作的输入数据。
创建Variables的步骤为两步：
（1）调用tf.Variable()创建一个Variable;
（2）调用tf.global_variables_initializer()初始化。


'''

import tensorflow as tf

init_val  = tf.random_normal((1,5),0,1)
var = tf.Variable(init_val, name='var')

print('Pre run:\n {}'.format(var))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	post_var = sess.run(var)
	
print('\nPost run:\n {}'.format(post_var))
