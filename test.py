# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:26:41 2017

@author: Pulu
"""


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import matplotlib # to plot images
# Force matplotlib to not use any X-server backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def weight_variable(shape):
    initial_value = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial_value)

def bias_variable(shape):
    initial_value = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial_value)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding="SAME")

def deconv2d(x,W,output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2,2,1], padding="SAME")

def DAE():
    
    x_image = tf.reshape(x,[-1,image_row,image_col,1])
    x_image_noise = tf.reshape(x_noise, [-1,28,28,1])
    tf.summary.scalar("imput",x_image_noise)
    with tf.name_scope("conv1_encoder"):
        with tf.name_scope("weight"):
            conv1_encoder_W = weight_variable([5,5,1,16])
        with tf.name_scope("weight"):
            conv1_encoder_b = bias_variable([16])
        with tf.name_scope("relu"):
            conv1_encoder_h = tf.nn.relu(tf.add(conv2d(x_image_noise,conv1_encoder_W),conv1_encoder_b))
    tf.summary.scalar("conv1_encoder",conv1_encoder_h)
    with tf.name_scope("conv2_encoder"):
        with tf.name_scope("weight"):
            conv2_encoder_W = weight_variable([5,5,16,32])
        with tf.name_scope("bias"):
            conv2_encoder_b = bias_variable([32])
        with tf.name_scope("relu"):
            conv2_encoder_h = tf.nn.relu(tf.add(conv2d(conv1_encoder_h,conv2_encoder_W),conv2_encoder_b))
    tf.summary.scalar("conv2_encoder",conv2_encoder_h)
    with tf.name_scope("conv3_encoder"):
        with tf.name_scope("weight"):
            conv3_encoder_W = weight_variable([5,5,32,64])
        with tf.name_scope("bias"):
            conv3_encoder_b = bias_variable([64])
        with tf.name_scope("relu"):
            conv3_encoder_h = tf.nn.relu(tf.add(conv2d(conv2_encoder_h,conv3_encoder_W),conv3_encoder_b))
    tf.summary.scalar("conv3_encoder",conv3_encoder_h)
    with tf.name_scope("hidden_layer"):
        hidden_layer = conv3_encoder_h
    tf.summary.scalar("hidden_layer",hidden_layer)
    with tf.name_scope("deconv1_decoder"):
        with tf.name_scope("weight"):
            conv1_decoder_W = weight_variable([5,5,32,64])
        with tf.name_scope("shape"):
            output_shape_decoder_conv1 = tf.stack([tf.shape(x)[0],7,7,32])
        with tf.name_scope("relu"):
            conv1_decoder_h = tf.nn.relu(deconv2d(conv3_encoder_h,conv1_decoder_W,output_shape_decoder_conv1))
    tf.summary.scalar("deconv1_decoder",conv1_decoder_h)
    with tf.name_scope("deconv2_decoder"):
        with tf.name_scope("weight"):
            conv2_decoder_W = weight_variable([5,5,16,32])
        with tf.name_scope("shape"):
            output_shape_decoder_conv2 = tf.stack([tf.shape(x)[0],14,14,16])
        with tf.name_scope("relu"):
            conv2_decoder_h = tf.nn.relu(deconv2d(conv1_decoder_h,conv2_decoder_W,output_shape_decoder_conv2))
    tf.summary.scalar("deconv2_decoder",conv2_decoder_h)
    with tf.name_scope("deconv3_decoder"):
        with tf.name_scope("weight"):
            conv3_decoder_W = weight_variable([5,5,1,16])
        with tf.name_scope("shape"):
            output_shape_decoder_conv3 = tf.stack([tf.shape(x)[0],28,28,1])
        with tf.name_scope("relu"):
            conv3_decoder_h = tf.nn.relu(deconv2d(conv2_decoder_h,conv3_decoder_W,output_shape_decoder_conv3))
    tf.summary.scalar("deconv3_decoder",conv3_decoder_h)
    x_image_reconstruct = conv3_decoder_h
    
    return x_image, hidden_layer, x_image_reconstruct

def vis(images, save_name):
    dim = images.shape[0]
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0/n_image_rows))
    gs = gridspec.GridSpec(n_image_rows,n_image_cols,top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    for g,count in zip(gs,range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count,:].reshape((28,28)))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_name + '_vis.png')
    
image_row = 28
image_col = 28

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape = [None,image_row*image_col])
with tf.name_scope('imput'):
    x_noise = tf.placeholder(tf.float32, shape = [None,image_row*image_col])
x_image, hidden_layer, x_image_reconstruct = DAE()
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.pow((x_image_reconstruct-x_image), 2))
tf.summary.scalar("loss",loss)
with tf.name_scope('train'):
    train_optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)


sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
batch_size = 50
print()
for i in range(1000):
    batch = mnist.train.next_batch(batch_size)
    batch_origin = batch[0]
    batch_noise = batch[0] + 0.5*np.random.randn(batch_size,image_row*image_col)
    if i%100 == 0:
        train_loss = loss.eval(feed_dict={x:batch_origin,x_noise:batch_noise})
        print("step %d, loss %g"%(i,train_loss))
    train_optimizer.run(feed_dict={x:batch_origin,x_noise:batch_noise})
print("final loss %g" % loss.eval(feed_dict={x: mnist.test.images, x_noise: mnist.test.images}))


test = mnist.test.next_batch(100)
test_origin = test[0]
test_noise = test[0] + 0.5*np.random.randn(100,image_row*image_col)
test_reconstuct = np.reshape(x_image_reconstruct.eval(feed_dict={x:test_origin,x_noise:test_noise}),[-1,28*28])
vis(test_origin,"origin")
vis(test_noise,"noise")
vis(test_reconstuct,"reconstuct")





