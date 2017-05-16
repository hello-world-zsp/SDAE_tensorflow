# -*- coding: utf8 -*-
import os
import tensorflow as tf
from utils import *

class MLPclassify(object):
    def __init__(self, sess, input_size, output_size, n_nodes=(180, 42, 10), learning_rate=1,
                 n_epochs=100,is_training=True, model_name = '',reg_lambda=0.1,batch_size = 20):

        self.sess = sess
        self.is_training = is_training
        self.name = model_name

        self.input_size = input_size        # 输入特征数
        self.output_size = output_size
        self.n_nodes = n_nodes              # 各层节点数
        self.n_layers = len(self.n_nodes)   # 层数
        self.n_epochs = n_epochs
        self.batch_size = batch_size


        self.lr = learning_rate
        self.dropout_keep_prob = 0.5
        self.reg_lambda = reg_lambda              # 还没用
        self.stddev = 0.02                  # 初始化参数用的

    # ------------------------- FC 层 -------------------------------------
    def hidden(self,input,output_filter,with_act = True,name = "hidden"):
        with tf.variable_scope(name):
            corrupt = tf.layers.dropout(input,rate= self.dropout_keep_prob,training=self.is_training)
            w = tf.get_variable('weights', shape=[input.shape[1], output_filter],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=self.stddev),
                                 regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))
            b = tf.get_variable('biases', shape=[1, output_filter],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32,
                                 regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))
            fc = tf.add(tf.matmul(corrupt, w), b)
            if with_act:
                bn = tf.layers.batch_normalization(fc)
                act = tf.nn.relu(bn)
            else:
                act = fc
        return act

    def build(self,is_training=True):
        self.x = tf.placeholder(tf.float32, [None,self.input_size],name="input")
        self.y = tf.placeholder(tf.float32, [None,self.output_size],name='label')

        input = self.x
        for i in range(self.n_layers-1):
            out = self.hidden(input, self.n_nodes[i], name=self.name+"hidden_layer" + str(i))
            input = out
        out = self.hidden(input, self.output_size, with_act = False, name=self.name + "output_layer")
        self.out = out

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=self.y, logits=out))
        self.train_vals = tf.trainable_variables()

    def train(self,x,y,valx,valy):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,var_list=self.train_vals)
        tf.global_variables_initializer().run()

        n_batch = x.shape[0] // self.batch_size
        print("num_batch", n_batch)
        for epoch in range(self.n_epochs):
            print("epoch: ", epoch)
            for batch in range(n_batch):
                batch_x = x[batch*self.batch_size:(batch+1)*self.batch_size]
                batch_y = y[batch*self.batch_size:(batch+1)*self.batch_size]
                _, loss,out = self.sess.run([self.optimizer,self.loss,self.out],feed_dict={self.x:batch_x,self.y:batch_y})
            # 每个epoch输出一下信息，loss用的是该epoch最后一个batch的loss,还有个准确率
            print ("train loss: ",loss)
            val_loss, val_out = self.sess.run([self.loss, self.out],feed_dict={self.x: valx, self.y: valy})
            right = tf.argmax(valy, 1).eval()
            prd = tf.argmax(val_out, 1).eval()
            correct_prediction = tf.equal(tf.argmax(valy, 1), tf.argmax(val_out, 1))     # argmax: one-hot to label
            acc_tf = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))              # 分类准确率
            acc = acc_tf.eval()
            # acc, right, prd = self.sess.run([acc_tf,right,prd], feed_dict={self.x: valx, self.y: valy})
            print ("val loss: ",val_loss,"val acc: ",acc )
            # print ("val loss: ", val_loss, "val acc: ", acc)

