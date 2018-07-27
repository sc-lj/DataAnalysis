# coding:utf-8
import tensorflow as tf


def tf_loss(pred,real):
    RMSE = tf.sqrt(tf.losses.mean_squared_error(pred, real))
    return RMSE

