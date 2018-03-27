from gan_layers import *
import tensorflow as tf 
import numpy as np 
def test_leaky_relu(x,y_true):
	tf.reset_default_graph()
	with get_session() as sess:
		y_tf = leaky_relu(tf.constant(x))
		y = sess.run(y_tf)
		print('Maximum error :%g'%rel_error(y,y_true))
answers = np.load('gan-checks-tf.npz')
test_leaky_relu(answers['lrelu_x'],answers['lrelu_y'])

def test_sample_noise():
	batch_size = 3
	dim =4
	tf.reset_default_graph()
	with get_session() as sess:
		z = sample_noise(batch_size,dim)
		#Check z has the correct shape 
		assert z.get_shape().as_list() == [batch_size,dim]
		#Make sure z is a tensor and not a numpy array 
		assert isinstance(z,tf.Tensor)
		#Check that we get different noise for different evaluations 
		z1 = sess.run(z)
		z2 = sess.run(z)
		assert not np.array_equal(z1,z2)
		#Check that we get the corrct range 
		assert np.all(z1>=-1.0) and np.all(z1<=1.0)
		print("All tests passed")
test_sample_noise()

def test_generator(true_count=188320):
	tf.reset_default_graph()
	with get_session() as sess:
		y = generator(tf.ones((1,4)))
		cur_count = count_params()
		if cur_count != true_count:
			print('Incorrect number of parameters in generator.{0}instead of {1}.Check your achitecture.'.format(cur_count,true_count))
		else:
			print('Correct number of parameters in generator.')
test_generator()

def test_gan_loss(logits_real,logits_fake,d_loss_true,g_loss_true):
	tf.reset_default_graph()
	with get_session() as sess:
		d_loss,g_loss = sess.run(gan_loss(tf.constant(logits_real),tf.constant(logits_fake)))
		print("Maximun error in d_loss:%g"%rel_error(d_loss_true,d_loss))
		print('Maximum error in g_loss:%g'%rel_error(g_loss_true,g_loss))
test_gan_loss(answers['logits_real'],answers['logits_fake'],
				answers['d_loss_true'],answers['g_loss_true'])

