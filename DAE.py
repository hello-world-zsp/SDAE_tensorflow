import tensorflow as tf
from utils import *
import time
import numpy as np
from scipy.misc import imsave

class DAE(object):
    def __init__(self, sess, input_size, noise=0, units=20,layer=0, learning_rate=0.01,
                 n_epoch=100,is_training = True, input_dim = 1,batch_size = 20):

        self.sess = sess
        self.is_training = is_training
        self.units = units                  # 隐层节点数
        self.layer = layer                  # 是第几层
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.input_size = input_size        # 输入特征数
        self.input_dim = input_dim          # 输入是几通道，rgb是3

        self.lr = learning_rate
        self.stddev = 0.02                  # 初始化参数用的
        self.noise = noise                  # dropout水平，是数\

        self.build(self.is_training)

    # ------------------------- 隐层 -------------------------------------
    def hidden(self, input, units, noise, name = "default"):
        with tf.variable_scope(name):
            # dropout+fc+lrelu
            corrupt = tf.layers.dropout(input,rate= noise,training=self.is_training)
            fc1 = tf.layers.dense(corrupt,units,
                                 kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                bias_initializer=tf.constant_initializer(0.0),reuse=not self.is_training)
            act1 = lrelu(fc1)               #leaky relu
            character = tf.layers.batch_normalization(act1)
            fc = tf.layers.dense(act1, input.shape[1],
                                 kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                 bias_initializer=tf.constant_initializer(0.0),reuse=not self.is_training)
            act = lrelu(fc)
            out = tf.layers.batch_normalization(act)
        return character, out

    def build(self,is_training=True):
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.input_size],
                                    name="input")
        self.character,self.out = self.hidden(self.x,self.units,noise = self.noise,
                                    name = "hidden_layer" + str(self.layer))
        self.loss = mse(self.out,self.x)

    def train(self, x, train_vals,summ_writer, summ_loss):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,var_list=train_vals)
        n_batch = x.shape[0] // self.batch_size
        print("num_batch", n_batch)
        # ------------------------- 训练 ----------------------------------
        for epoch in range(self.n_epoch-1):
            for batch in range(n_batch):
                batch_x = x[batch*self.batch_size:(batch+1)*self.batch_size]
                _, loss,out,sum_loss_str = self.sess.run([self.optimizer,self.loss,self.out,summ_loss],
                                                         feed_dict={self.x:batch_x})
                summ_writer.add_summary(sum_loss_str,epoch*n_batch+batch)
            print("epoch ",epoch," train loss: ", loss)

        #----------- 最后一个epoch,除了训练，还要获取下一层训练的特征图 ------
        epoch = self.n_epoch - 1
        begin = time.time()
        characters = []
        outs = []
        for batch in range(n_batch):
            batch_x = x[batch * self.batch_size:(batch + 1) * self.batch_size]
            _, loss, out, character,sum_loss_str = self.sess.run([self.optimizer, self.loss, self.out,
                                                                  self.character,summ_loss],
                                                                feed_dict={self.x: batch_x})
            summ_writer.add_summary(sum_loss_str, epoch * n_batch + batch)
            characters.append(character)
            outs.append(out)
        self.next_x = np.concatenate(tuple(characters))
        self.rec = np.concatenate(tuple(outs))
        end = time.time()
        print("epoch ", epoch, " train loss: ", loss)
        print("time ",end-begin)
        # -------------------------------------------------------------------



