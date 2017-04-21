import tensorflow as tf
from scipy.misc import imsave
import numpy as np

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

def save_image(input, name,n_show):
    n_per_line = 10                                                         # 每行10张图
    n_lines = n_show//n_per_line
    h = input.shape[1] ** 0.5                                               # 图像大小
    w = h
    img_total = np.zeros([h * n_lines, w * n_per_line])                     # 灰度图
    for i in range(n_show):
        rec = (input[i] + 1) * 127
        img = np.reshape(rec,[h,w])                                         # 行向量变图像
        row = i // n_lines
        col = i % n_per_line
        img_total[row * h:(row+1) * h,col*w:(col+1)*w] = img
    imsave(name, img_total)