import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from models import *
import argparse
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/mnist')
parser.add_argument('--sample_dir', type=str, default='samples/mnist')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epoch', type=int, default=100000)
parser.add_argument('--train_ratio', type=int, default=2)

config = parser.parse_args()

#data = input_data.read_data_sets(config.data_dir, one_hot=True)

def sigmoid_ce_loss(a,b):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=a,labels= b))

def sample_c(m):
	return np.random.multinomial(1, 10*[0.1], size=m) 

class InfoGAN():
	def __init__(self, data):
		self.sess = tf.InteractiveSession()
		self.data = data
		self.size = 28 
		self.z_dim = 100
		self.c_dim = 10 #num of classes
		self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, 1])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
		self.c = tf.placeholder(tf.float32, shape=[None, self.c_dim])

		self.Gen = gen_conv(tf.concat(axis=1, values=[self.z, self.c]))
		self.Disc_real, _ = disc_conv(self.X)
		self.Disc_fake, self.Q_fake = disc_conv(self.Gen, reuse = True)

		self.Gen_loss = sigmoid_ce_loss(self.Disc_real, tf.ones_like(self.Disc_real))
		self.Disc_loss = sigmoid_ce_loss(self.Disc_fake, tf.ones_like(self.Disc_fake)) 
		self.Q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Q_fake,labels=self.c))

		self.optimizer = tf.train.AdamOptimizer()
		self.Gen_opt = self.optimizer.minimize(self.Gen_loss)
		self.Disc_opt = self.optimizer.minimize(self.Disc_loss)
		self.Q_opt = self.optimizer.minimize(self.Q_loss)


	def train(self, batch_size=config.batch_size, total_epochs=config.num_epoch, 
			  sample_dir=config.sample_dir, train_ratio=config.train_ratio):
		self.sess.run(tf.global_variables_initializer())
		if not os.path.exists(sample_dir):
			os.makedirs(sample_dir)
		
		check_dir = 'checkpoint/'
		if not os.path.exists(check_dir):
			os.makedirs(check_dir) 
	
		for epoch in range(total_epochs):
			batch_X, _ = self.data.train.next_batch(batch_size)
			batch_X = np.reshape(batch_X, (batch_size, self.size, self.size, 1))
			batch_z = np.random.uniform(-1., 1., size=[batch_size, self.z_dim])
			batch_c = sample_c(batch_size)

			self.sess.run(self.Disc_opt, feed_dict={self.z:batch_z, self.c:batch_c})
			self.sess.run(self.Gen_opt, feed_dict={self.X:batch_X})
			for _ in range(train_ratio):
				self.sess.run(self.Q_opt, feed_dict={self.z:batch_z, self.c:batch_c})

			if epoch % 1000 == 0 :
				Gen_current_loss = self.sess.run(self.Gen_loss, feed_dict={self.X:batch_X})
				Disc_current_loss, Q_current_loss = self.sess.run([self.Disc_loss, self.Q_loss], feed_dict={self.z:batch_z, self.c:batch_c})
				print('epoch : {}, G_loss : {}, D_loss : {}, Q_loss : {}'.format(epoch, Gen_current_loss, Disc_current_loss, Q_current_loss))

			if epoch % 10000 == 0 :
				z_s = np.random.uniform(-1., 1., size=[batch_size, self.z_dim])
				c_s = sample_c(batch_size)
				samples = self.sess.run(self.Gen, feed_dict={self.z: z_s, self.c: c_s})
				samples_array = tf.reshape(samples, (batch_size,28,28)) 
				samples_array = samples_array.eval()
				samples_array = samples_array.astype(np.uint8)
				for i in range(batch_size):
					im = Image.fromarray(samples_array[i])
					im.save('{}/{}_{}.png'.format(sample_dir, epoch, i))

			if epoch == (total_epochs-1) :
				tf.train.Saver().save(self.sess, 'checkpoint/infogan_final.ckpt')

if __name__ == '__main__':
	#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	#os.environ["CUDA_VISIBLE_DEVICES"]= "2"
	data = input_data.read_data_sets(config.data_dir, one_hot=True)
	Learning_model = InfoGAN(data)
	Learning_model.train()
	
