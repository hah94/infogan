import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		a1 = 0.5 * (1 + leak)
		a2 = 0.5 * (1 - leak)
		val = a1 * x + a2 * abs(x)
		return val

def gen_conv(z, size=7, channel=1, name='G_conv'):
	with tf.variable_scope(name) as scope:
		with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], activation_fn=tf.nn.relu,
	                    	normalizer_fn=slim.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02)):
			net = slim.fully_connected(z, size * size * 128, scope='fc')
			net = tf.reshape(net, (-1, size, size, 128))
			net = slim.conv2d_transpose(net, 64, 4, stride=2, padding='SAME', scope='deconv1')
			net = slim.conv2d_transpose(net, 1, 4, stride=2, activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope='deconv2')

			return net

def disc_conv(x, name='D_conv', reuse=False, size=64):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		with slim.arg_scope([slim.conv2d], activation_fn=lrelu, stride=2, kernel_size=4):
			net = slim.conv2d(x, size, scope='conv1')
			net = slim.conv2d(net, size * 2, normalizer_fn=slim.batch_norm, scope='conv2')
			net = slim.flatten(net) 
			d = slim.fully_connected(net, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02), scope='fc_d')
			q = slim.fully_connected(net, 128, actiavtion_fn=lrelu, normalizer_fn=slim.batch_norm, scope='fc_q1')
			q = slim.fully_connected(q, 10, activation_fn=None, scope='fc_q2') # class num : 10

			return d, q


