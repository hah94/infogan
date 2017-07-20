import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import models
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/mnist')
parser.add_argument('--sample_dir', type=str, default='samples/mnist')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=10e6)
parser.add_argument('--train_ratio', type=int, default=2)

config = parser.parse_args()

data = input_data.read_data_sets(config.data_dir, one_hot=True)

def sigmoid_ce_loss(a,b):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(a, b))

def sample_c(m, n, ind=-1):
	c = np.zeros([m,n])
	for i in range(m):
		if ind<0:
			int = np.random.randint(10)
		c[i,i%10] = 1
	return c

def generate_png(samples):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample)
	return fig

class InfoGAN():
	def __init__(self, data):
		self.data = data
		self.size = 28 
		self.z_dim = 100
		self.c_dim = 10 #num of classes
		self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
		self.c = tf.placeholder(tf.float32, shape=[None, self.c_dim])

		self.Gen = models.gen_conv(tf.concat([self.z, self.c],1))
		self.Disc_real, _ = models.disc_conv(self.X)
		self.Disc_fake, self.Q_fake = models.disc_conv(self.Gen, reuse = True)

		self.Gen_loss = sigmoid_ce_loss(self.Disc_real, tf.ones_like(self.Disc_real))
		self.Disc_loss = sigmoid_ce_loss(self.Disc_fake, tf.ones_like(self.Disc_fake)) 
		self.Q_loss = slim.losses.cross_entropy_loss(self.Q_fake, self.c)

		self.optimizer = tf.train.AdamOptimizer()
		self.Gen_opt = self.optimizer.minimize(self.Gen_loss)
		self.Disc_opt = self.optimizer.minimize(self.Disc_loss)
		self.Q_opt = self.optimizer.minimize(self.Q_loss)

		self.sess = tf.Session()

	def train(self, batch_size=config.batch_size, total_epochs=config.num_epoch, 
			  sample_dir=config.sample_dir, train_ratio=config.train_ratio):
		fig_count = 0
		self.sess.run(tf_global_variables_initializer())

		for epoch in range(total_epochs):
			batch_X, _ = self.data.train.next_batch(batch_size)
			batch_z = np.random.uniform(-1., 1., size=[batch_size, self.z_dim])
			batch_c = sample_c(batch_size, self.c_dim)

			fd = {self.z:batch_z, self.c:batch_c}
			self.sess.run(self.Disc_opt, feed_dict=fd)
			self.sess.run(self.Gen_opt, feed_dict=fd)
			for _ in range(train_ratio):
				self.sess.run(self.Q_opt, feed_dict=fd)

			if epoch % 1000 == 0 :
				Disc_current_loss = self.sess.run(self.Disc_loss, 
												  feed_dict={self.X:batch_x, self.z:batch_z, self.c:batch_c})
				Gen_current_loss, Q_current_loss = self.sess.run([self.Gen_loss, self.Q_loss], feed_dict=fd)
				print('epoch : {}, D_loss : {}, G_loss : {}, Q_loss : {}'.format(epoch, Disc_current_loss, Gen_current_loss, Q_current_loss))

			if epoch % 10000 == 0 :
				z_s = np.random.uniform(-1., 1., size=[16, self.z_dim])
				c_s = sample_c(16, self.c_dim, fig_count%10)
				samples = self.sess.run(self.Gen, feed_dict={self.z: z_s, self.c: c_s})
	 			
				figs = generate_png(samples)
				plt.savefig('{}/{}_{}.png'.format(sample_dir, epoch, str(fig_count%10)))
				fig_count += 1
				plt.close(fig)

if __name__ == '__main__':
	Learning_model = InfoGAN(data)
	Learning_model.train()
	
