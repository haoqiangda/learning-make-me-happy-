import tensorflow as tf 
from gan_layers import *
import numpy as np 

tf.reset_default_graph()
batch_size = 128
noise_dim = 96
#place the holder for the images from the training data 
x = tf.placeholder(tf.float32,[None,784])
z = sample_noise(batch_size,noise_dim)
G_sample = generator(z)

with tf.variable_scope("") as scope:

	logits_real = discriminator(preprocess_img(x))
	#Re-use discriminator weights on new inputs 
	scope.reuse_variables()
	logits_fake = discriminator(G_sample)

#get the list of variables for the discriminator and generator 
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

#get the solver
G_solver, D_solver = get_solvers()

#get the loss
D_loss, G_loss = gan_loss(logits_real, logits_fake)

D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
#train gan 
def run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step,\
				show_every=1000,print_every= 50,batch_size=128,num_epoch=10):
	max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
	for it in range(max_iter):
		if it % show_every ==0:
			samples = sess.run(G_sample)
			fig = show_images(samples[:16])
			plt.show()
			print()
		#run a batch of data through network
		minibatch,minibatch_y = mnist.train.next_batch(batch_size)
		_, D_loss_curr = sess.run([D_train_step,D_loss],feed_dict={x:minibatch})
		_, G_loss_curr = sess.run([G_train_step,G_loss])

		if it % print_every == 0:
			print('Iter: {},D:{},G:{}'.format(it,D_loss_curr,G_loss_curr))
	print('Final Images')
	samples = sess.run(G_sample)
	fig = plt.show_images(samples[:16])
	plt.show()

with get_session() as sess:
	sess.run(tf.global_variables_initializer())
	run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)