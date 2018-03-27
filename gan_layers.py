#!/usr/bin/env python
#coding:utf-8
#Copyright (C) dirlt
from __future__ import print_function,division 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
#set default size of plots 
plt.rcParams['figure.figsize'] = (10.0,8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] ='gray'

def show_images(images):
	images = np.reshape(images,[images.shape[0],-1])
	sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
	sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
	fig = plt.figure(figsize=(sqrtn,sqrtn))
	gs=gridspec.GridSpec(sqrtn, sqrtn)
	gs.update(wspace=0.05,hspace=0.05)
	for i ,img in enumerate(images):
		ax=plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(img.reshape([sqrtimg,sqrtimg]))
	return 
def preprocess_img(x):
	return 2*x-1.0
def deprocess_img(x):
	return (x+1.0)/2.0
def rel_error(x,y):
	return np.max(np.abs(x-y)/(np.maximum(1e-8,np.abs(x)+np.abs(y))))
def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config = config)
	return session
def count_params():
	param_count = np.sum([np.prod(x.get_shape().as_list())for x in tf.global_variables()])
	return param_count

#load minist image from tensorflow 
from tensorflow.examples.tutorials.mnist import input_data
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
mnist = input_data.read_data_sets('./datasets/MINIST_data',one_hot = False)
#show a batch 
show_images(mnist.train.next_batch(16)[0])



def leaky_relu(x,alpha=0.01):
	condition = tf.less(x,0)
	res = tf.where(condition,alpha*x,x)
	return res 

#load random noise 
def sample_noise(batch_size,dim):
	"""Generate random uniform noise from -1 to 1.
	Inputs:
	-batch_size:integer giving the batch size of noise to generate
	-dim :integer giving the dimension of the noise to generate 
	Returns:
	Tensorflow Tensor containing uniform noise in[-1,1] with shape [batch_size,dim]"""
	return tf.random_uniform([batch_size,dim],minval =-1,maxval=1)

def discriminator(x):
	with tf.variable_scope("discriminator"):
		fc1 = tf.layers.dense(x,units=256,use_bias=True,name="first_fc")
		leaky_relu1 = leaky_relu(fc1,alpha=0.01)
		fc2 = tf.layers.dense(leaky_relu1,units=256,use_bias=True,name="second_fc")
		leaky_relu2 = leaky_relu(fc2,alpha=0.01)
		logits = tf.layers.dense(leaky_relu2,units=1,name="logits")
		return logits

def generator(z):
	with tf.variable_scope("generator"):
		fc1 = tf.layers.dense(z,units=1024,use_bias=True)
		relu1 = tf.nn.relu(fc1)
		fc2 = tf.layers.dense(relu1,units=1024,use_bias=True)
		relu2 = tf.nn.relu(fc2)
		fc3 = tf.layers.dense(relu2,units=784,use_bias=True)
		img = tf.nn.tanh(fc3)
		return img

def gan_loss(logits_real,logits_fake):
	G_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake),logits = logits_fake)
	D_loss =tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),logits = logits_real,name = "discriminator_real_loss")+ \
					tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),logits=logits_fake,name="discriminator_fake_loss")
	D_loss = tf.reduce_mean(D_loss)
	G_loss = tf.reduce_mean(G_loss)
	return D_loss,G_loss

def get_solvers(learning_rate=1e-3,beta1=0.5):
	G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
	D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
	return G_solver,D_solver
