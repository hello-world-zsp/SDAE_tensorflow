# -*- coding: utf8 -*-
import tensorflow as tf
from scipy.misc import imsave
import numpy as np
import os,sys

def solo_to_tuple(val, n=3):
    if type(val) in (list, tuple):
        return val
    else:
        return (val,)*n


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def mse(x,y):
    # return tf.reduce_mean(tf.pow(tf.subtract(x,y),2.0))
    return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(x,y),2.0),axis=1))


def loss_x_entropy(output, target):
    """Cross entropy loss
    Args:
        output: tensor of net output
        target: tensor of net we are trying to reconstruct
    Returns:
        Scalar tensor of cross entropy
    """
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'), target_tf),
                               tf.multiply(tf.log(1 - net_output_tf),(1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                     name='xentropy_mean')


def save_image(input, name,n_show):
    n_per_line = n_show ** 0.5                                              # 每行10张图
    n_lines = n_show//n_per_line
    h = input.shape[1] ** 0.5                                               # 图像大小
    w = h
    img_total = np.zeros([h * n_lines, w * n_per_line])                     # 灰度图
    for i in range(n_show):
        rec = (input[i] + 1) * 127                                          # 将(0,1)变换到（0,254）
        img = np.reshape(rec,[h,w])                                         # 行向量变图像
        row = i // n_lines
        col = i % n_per_line
        img_total[row * h:(row+1) * h,col*w:(col+1)*w] = img
    imsave(name, img_total)

def save_batch_data(name,input_data,is_New=False):
    save_name = os.path.join(sys.path[0], name)
    if not is_New:
        temp = np.load('./'+name).astype(np.float32)
        if temp.shape[0]>0:
            input_data = np.concatenate((temp,input_data), axis=0)

    np.save(save_name,input_data)