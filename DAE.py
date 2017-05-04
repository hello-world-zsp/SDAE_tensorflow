import tensorflow as tf
from utils import *
import time
import numpy as np
from scipy.misc import imsave

class DAE(object):
    def __init__(self, sess, input_size, noise=0, units=20,layer=0, learning_rate=0.01,
                 n_epoch=100,is_training = True, input_dim = 1,batch_size = 20,decay=0.95,
                 summary_handle = None):

        self.sess = sess
        self.is_training = is_training
        self.units = units                  # 隐层节点数
        self.layer = layer                  # 是第几层
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.input_size = input_size        # 输入特征数
        self.input_dim = input_dim          # 输入是几通道，rgb是3

        self.lr_init = learning_rate
        self.stddev = 0.2                   # 初始化参数用的
        self.noise = noise                  # dropout水平，是数\
        self.reg_lambda = 0.0                  # 正则化系数,float
        self.dropout_p = 0.5                # dropout层保持概率
        self.lr_decay = decay
        self.change_lr_epoch = int(n_epoch*0.3) # 开始改变lr的epoch数

        self.summ_handle = summary_handle
        self.build(self.is_training)

    # ------------------------- 隐层 -------------------------------------
    def hidden(self, input, units, noise, name = "default"):
        input_size = int(input.shape[1])
        with tf.variable_scope(name):
            # mask噪声
            corrupt = tf.layers.dropout(input,rate= noise,training=self.is_training)
            # 加性高斯噪声
            # corrupt = tf.add(input,noise * tf.random_uniform(input.shape))
            ew = tf.get_variable('enc_weights',shape=[input_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0,stddev=self.stddev),
                                 regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))
            sew = tf.summary.histogram(name + '/enc_weights', ew)

            eb = tf.get_variable('enc_biases',shape=[1,units],
                                initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))
            seb = tf.summary.histogram(name+'/enc_biases',eb)
            fc1 = tf.add(tf.matmul(corrupt,ew),eb)
            # act1 = lrelu(fc1)               #leaky relu
            # act1 = tf.nn.relu(tf.layers.batch_normalization(fc1))
            act1 = tf.nn.sigmoid(tf.layers.batch_normalization(fc1))
            character = act1
            self.ew = ew
            self.eb = eb

            dw = tf.transpose(ew)
            db = tf.get_variable('dec_biases',shape=[1,input_size],
                                initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))
            sdb = tf.summary.histogram(name+'/dec_biases',db)
            self.summ_handle.add_summ(sew, seb,sdb)
            fc = tf.add(tf.matmul(act1,dw),db)

            # act = lrelu(fc)
            # act = lrelu(tf.nn.dropout(fc, self.dropout_p))
            out = tf.sigmoid(fc)

        return character, out

    def build(self,is_training=True):
        layer_name = "hidden_layer" + str(self.layer)
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.input_size],name="input")
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.character, self.out = self.hidden(self.x,self.units,noise = self.noise,
                                                name = layer_name)

        reg_losses = tf.losses.get_regularization_losses(layer_name)
        for loss in reg_losses:
            tf.add_to_collection('losses' + layer_name, loss)
        self.reg_losses = tf.get_collection('losses'+layer_name)

        mse_loss = mse(self.out, self.x)
        tf.add_to_collection('losses'+layer_name, mse_loss)
        self.loss = tf.add_n(tf.get_collection('losses'+layer_name))

    def train(self, x, train_vals, summ_writer, summ_handle):
        temp = set(tf.all_variables())
        self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9).minimize(self.loss,var_list=train_vals)
        # adam中有slot,需要初始化。
        tf.initialize_variables(set(tf.all_variables()) - temp).run()

        n_batch = x.shape[0] // self.batch_size
        print("num_batch", n_batch)
        counter = 0
        current_lr = self.lr_init
        begin_time = time.time()
        # ------------------------- 训练 ----------------------------------
        for epoch in range(self.n_epoch-1):
            if epoch > self.change_lr_epoch:
                current_lr = current_lr * self.lr_decay
            for batch in range(n_batch):
                counter += 1
                batch_x = x[batch*self.batch_size:(batch+1)*self.batch_size]
                _, loss, reg_loss, out, summ_loss= self.sess.run([self.optimizer, self.loss,self.reg_losses, self.out,
                                                                 summ_handle.summ_loss[0]],
                                                                feed_dict={self.x: batch_x, self.lr:current_lr})
                summ_writer.add_summary(summ_loss,epoch * n_batch + batch)
                if counter%50==0:
                # 记录w,b
                    summ_ew, summ_eb, summ_db = self.sess.run([summ_handle.summ_enc_w[0], summ_handle.summ_enc_b[0],
                                                                        summ_handle.summ_dec_b[0]],
                                                                        feed_dict={self.x: batch_x, self.lr:current_lr})
                    summ_writer.add_summary(summ_ew, epoch * n_batch + batch)
                    summ_writer.add_summary(summ_eb, epoch * n_batch + batch)
                    summ_writer.add_summary(summ_db, epoch * n_batch + batch)
            print("epoch ",epoch," train loss: ", loss," reg loss: ",reg_loss, " time:",str(time.time()-begin_time))

        #----------- 最后一个epoch,除了训练，还要获取下一层训练的特征图，记录wb的分布 ------
        epoch = self.n_epoch - 1
        characters = []
        outs = []
        for batch in range(n_batch):
            batch_x = x[batch * self.batch_size:(batch + 1) * self.batch_size]
            _, loss, out, character, self.ewarray,self.ebarray, summ_loss,summ_ew,summ_eb,summ_db\
                = self.sess.run([self.optimizer, self.loss, self.out,self.character,self.ew,self.eb,
                                summ_handle.summ_loss[0],summ_handle.summ_enc_w[0],summ_handle.summ_enc_b[0],
                                summ_handle.summ_dec_b[0]],
                                feed_dict={self.x: batch_x, self.lr:current_lr})
            summ_writer.add_summary(summ_loss, epoch * n_batch + batch)
            summ_writer.add_summary(summ_ew, epoch * n_batch + batch)
            summ_writer.add_summary(summ_eb, epoch * n_batch + batch)
            summ_writer.add_summary(summ_db, epoch * n_batch + batch)
            characters.append(character)
            outs.append(out)
        self.next_x = np.concatenate(tuple(characters))
        self.rec = np.concatenate(tuple(outs))
        print("epoch ", epoch, " train loss: ", loss, " time:",str(time.time()-begin_time))
        # -------------------------------------------------------------------



