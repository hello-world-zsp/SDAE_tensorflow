# -*- coding: utf8 -*-
import os
import tensorflow as tf
from utils import *
from DAE import *

class SummaryHandle():
    def __init__(self):
        self.summ_enc_w = []
        self.summ_enc_b = []
        self.summ_dec_b = []
        self.summ_loss = []

    def add_summ(self,e_w,e_b,d_b):
        self.summ_enc_w.append(e_w)
        self.summ_enc_b.append(e_b)
        self.summ_dec_b.append(d_b)


class SDAE(object):
    def __init__(self, sess, input_size, noise=0, n_nodes=(180, 42, 10), learning_rate=1,
                 n_epochs=100,is_training = True, input_dim = 1,
                 data_dir = None,batch_size = 20,num_show = 100,load_freq = (10,10,10)):

        self.sess = sess
        self.is_training = is_training
        self.n_nodes = n_nodes              # 各层节点数
        self.n_layers = len(self.n_nodes)   # 层数
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_size = input_size        # 输入特征数
        self.input_dim = input_dim          # 输入是几通道，rgb是3

        self.lr = learning_rate
        self.stddev = 0.02                  # 初始化参数用的
        self.noise = noise                  # dropout水平，是tuple

        self.checkpoint_dir = 'checkpoint'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.data_dir = data_dir
        self.n_show = num_show
        self.load_freq = load_freq

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(self.data_dir):
            print ("Error! No such data path.")

    def build(self,is_training=True):
        self.hidden_layers =[]
        self.loss = []
        self.summary_handles = []
        n_input = [self.input_size]+list(self.n_nodes[:-1])

        for i in range(self.n_layers):
            summary_handle = SummaryHandle()
            layer = DAE(self.sess, n_input[i], noise=self.noise[i],units=self.n_nodes[i],
                        layer=i, n_epoch=self.n_epochs[i], is_training=self.is_training,
                        input_dim=self.input_dim, batch_size=self.batch_size,learning_rate=self.lr[i],
                        summary_handle = summary_handle,load_freq=self.load_freq[i])
            self.loss.append(layer.loss)
            self.hidden_layers.append(layer)

            # self.summary_character.append(tf.summary.image(layer.next_x, "character" + str(i)))
            summary_handle.summ_loss.append(tf.summary.scalar('loss'+str(i),layer.loss))
            self.summary_handles.append(summary_handle)

        # ------------------------ 提取各层可训练参数 -----------------------------------
        self.train_vals = tf.trainable_variables()
        self.train_vals_layer =[]
        for i in range(self.n_layers):
            self.train_vals_layer.append ( [var for var in self.train_vals if str(i) in var.name.split("/")[0]])
        # ------------------------------------------------------------------------------

    def train(self, dataset = None,load_data_batch_func=None, shuffle=True):
        tf.global_variables_initializer().run()
        self.writer = tf.summary.FileWriter('./'+self.log_dir, self.sess.graph)

        read_data_path = self.data_dir+'/'+dataset

        x = load_data_batch_func(read_data_path)
        input = x
        if shuffle:
            r = np.random.permutation(len(x))
            input = x[r,:]
        # 保存原图
        save_image(input, name=self.result_dir + '/original' + str(0) + '.png', n_show=self.n_show)
        features = []
        imgs = []

        for layer in self.hidden_layers:
            idx = self.hidden_layers.index(layer)
            print("training layer: ",idx )
            save_data_path = self.data_dir + '/character' + str(idx) + '.npy'
            layer.train(input, read_data_path,save_data_path, self.train_vals_layer[idx],
                        summ_writer=self.writer, summ_handle=self.summary_handles[idx],
                        load_data_batch_func=load_data_batch_func)
            input = layer.next_x

            read_data_path = save_data_path
            # 保存重建图

            save_image(layer.rec,name = self.result_dir+'/rec'+str(idx)+'.png',n_show = self.n_show)
            features.append(layer.next_x)
            if idx == 0:
                img = np.add(layer.ewarray.T,
                                   np.dot(layer.ebarray.T,np.ones([1,self.input_size])))
            else:
                img = np.add(np.dot(layer.ewarray.T,imgs[idx-1]),
                                   np.dot(layer.ebarray.T, np.ones([1, self.input_size])))
            img = tf.sigmoid(img).eval()
            imgs.append(img)
            save_image(imgs[idx], name=self.result_dir + '/feature' + str(idx) + '.png', n_show=self.n_nodes[idx])

            save_image(tf.sigmoid(layer.next_x).eval(), name=self.result_dir + '/character' + str(idx) + '.png', n_show=self.n_show)
        features = np.concatenate(tuple(features[1:]),axis = 1)
        return features


