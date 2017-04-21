import os
import tensorflow as tf
from utils import *
from DAE import *

class SDAE(object):
    def __init__(self, sess, input_size, noise=0, n_nodes=(180, 42, 10), learning_rate=1,
                 n_epochs=100,is_training = True, input_dim = 1,
                 lambda1 = (.4, .05, .05),data_dir = None,batch_size = 20,num_show = 100):

        self.sess = sess
        self.is_training = is_training
        self.n_nodes = n_nodes              # 各层节点数
        self.n_layers = len(self.n_nodes)   # 层数
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_size = input_size        # 输入特征数
        self.input_dim = input_dim          # 输入是几通道，rgb是3

        self.lr = learning_rate
        self.lambda1 = lambda1              # 还没用
        self.stddev = 0.02                  # 初始化参数用的
        self.noise = noise                  # dropout水平，是tuple

        self.checkpoint_dir = 'checkpoint'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.n_show = num_show

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    def build(self,is_training=True):
        self.hidden_layers =[]
        self.loss = []
        self.summary_character = []
        self.summary_loss = []
        n_input = [self.input_size]+list(self.n_nodes[:-1])
        for i in range(self.n_layers):
            layer = DAE(self.sess, n_input[i], noise=self.noise[i],units=self.n_nodes[i],
                        layer=i, n_epoch=self.n_epochs[i], is_training=self.is_training,
                        input_dim=self.input_dim, batch_size=self.batch_size,learning_rate=self.lr[i])
            self.loss.append(layer.loss)
            self.hidden_layers.append(layer)
            # self.summary_character.append(tf.summary.image(layer.next_x, "character" + str(i)))
            self.summary_loss.append(tf.summary.scalar('loss',layer.loss))

        # ------------------------ 提取各层可训练参数 -----------------------------------
        self.train_vals = tf.trainable_variables()
        self.train_vals_layer =[]
        for i in range(self.n_layers):
            self.train_vals_layer.append ( [var for var in self.train_vals if str(i) in var.name.split("/")[0]])
        # ------------------------------------------------------------------------------


    def train(self,x,valx):
        tf.global_variables_initializer().run()
        self.writer = tf.summary.FileWriter('./'+self.log_dir, self.sess.graph)

        input = x
        # 保存原图
        save_image(x, name=self.result_dir + '/original' + str(0) + '.png', n_show=self.n_show)
        for layer in self.hidden_layers:
            idx = self.hidden_layers.index(layer)
            print("training layer: ",idx )
            layer.train(input,self.train_vals_layer[idx],
                        summ_writer=self.writer,summ_loss=self.summary_loss[idx])
            input = layer.next_x
            # 保存重建图

            save_image(layer.rec,name = self.result_dir+'/rec'+str(idx)+'.png',n_show = self.n_show)
            save_image(layer.next_x, name=self.result_dir + '/character' + str(idx) + '.png', n_show=self.n_show)
