# coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST", one_hot=True)

learning_rate = 0.001
training_epochs=500
batch_size=200
display_size=1000
display_step=1
examples_to_show=10


n_hidden_1=32*32
n_input = 784

X=tf.placeholder(tf.float32,[None,n_input])
weight={'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'decoder_h1': tf.Variable(tf.random_normal([n_input,n_hidden_1]))}
biase={'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
       "decoder_b1":tf.Variable(tf.random_normal([n_input]))}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weight['encoder_h1']),biase['encoder_b1']))
    return layer_1

def decoder(x):
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(x,tf.transpose(weight['decoder_h1'])),biase['decoder_b1']))
    return layer_2

def log_func(x1,x2):
    return tf.multiply(x1,tf.log(tf.div(x1,x2)))

def KL_div(rho,rho_rate):
    inv_rho=tf.subtract(tf.constant(1.),rho)
    inv_rhohat=tf.subtract(tf.constant(1.),rho_rate)
    log_rho=log_func(rho,rho_rate)+log_func(inv_rho,inv_rhohat)
    return log_rho


def next_batch(batch_size):
    """
    :param batch_size:
    :return: 自变量和label
    """
    files="../data/"

    return var,Y
encoder_op=encoder(X)
decoder_op=decoder(encoder_op)
rho_hat=tf.reduce_mean(decoder_op,1)

y_pred=decoder_op
y_true=X

cost_m=tf.reduce_mean(tf.pow(y_true-y_pred,2))
cost_sparse=0.001*tf.reduce_mean(KL_div(0.2,rho_hat))
cost_reg=0.0001*(tf.nn.l2_loss(weight['decoder_h1'])+tf.nn.l2_loss(weight['encoder_h1']))
cost=tf.add(cost_reg,tf.add(cost_m,cost_sparse))

optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for i in range(100):
            batch_xs, batch_ys = next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:10]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # Store the Decoder and Encoder Weights
    dec = sess.run(weight['decoder_h1'])
    enc = sess.run(weight['encoder_h1'])

