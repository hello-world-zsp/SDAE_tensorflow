import os
import tensorflow as tf
from utils import *

class MLP(object):
    def __init__(self, sess, input_size, noise=0, n_nodes=(180, 42, 10), learning_rate=1,
                 n_epochs=100,is_training = True, input_dim = 1,
                 lambda1 = (.4, .05, .05),data_dir = None,batch_size = 20):

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

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    # ------------------------- 编码层 -------------------------------------
    def encoder(self,input,output_filter,noise,name = "default"):
        with tf.variable_scope(name):
            # dropout+fc+lrelu
            corrupt = tf.layers.dropout(input,rate= noise,training=self.is_training)
            fc = tf.layers.dense(corrupt,output_filter,
                                 kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                bias_initializer=tf.constant_initializer(0.0))
            act = lrelu(fc) #leaky relu
        return act

    # ------------------------- 解码层 -------------------------------------
    def decoder(self,input,output_filter,name="default"):
        with tf.variable_scope(name):
            # fc+lrelu
            fc = tf.layers.dense(input, output_filter,
                                 kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                 bias_initializer=tf.constant_initializer(0.0))
            act = lrelu(fc)
        return act

    def build(self,is_training=True):
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.input_size],
                                    name="input")
        self.y = tf.placeholder(tf.float32,[self.batch_size,self.n_nodes[-1]],name='label')
        # --------------------------- 建立编码层 -----------------------------------------
        self.enc_layers = []
        input = self.x
        for i in range(self.n_layers):
            out = self.encoder(input,self.n_nodes[i],noise = self.noise[i],
                               name = "encode_layer" + str(i))
            self.enc_layers.append(out)
            input = out
        self.enc_out = out
        # -------------------------------------------------------------------------------
        #----------------------------建立解码层-------------------------------------------
        self.dec_layers = []
        input = self.enc_out
        dec_nodes = list(self.n_nodes[:self.n_layers-1])        # 解码器各层节点数，与编码器对应
        dec_nodes.reverse()
        dec_nodes.append(self.input_size)
        for i in range(self.n_layers):
            out = self.decoder(input,dec_nodes[i],
                               name = "decode_layer" + str(i))
            self.dec_layers.append(out)
            input = out
        self.rec = out  # 重建图
        # ------------------------------------------------------------------------------
        # ------------------------ 定义loss --------------------------------------------
        self.loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=self.y,logits=self.enc_out))
        self.loss2 = mse(self.rec,self.x)
        self.loss = self.loss1 + self.loss2
        # ------------------------------------------------------------------------------
        # ------------------------ 提取各层可训练参数 -----------------------------------
        self.train_vals = tf.trainable_variables()
        self.train_vals_layer =[]
        for i in range(self.n_layers):
            self.train_vals_layer.append ( [var for var in self.train_vals if str(i) in var.name.split("/")[0]])
        # ------------------------------------------------------------------------------

    def train(self,x,y,valx,valy):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr[0]).minimize(self.loss,var_list=self.train_vals)
        tf.global_variables_initializer().run()

        n_batch = x.shape[0] // self.batch_size
        print("num_batch", n_batch)
        for epoch in range(self.n_epochs[0]):
            print("epoch: ", epoch)
            for batch in range(n_batch):
                batch_x = x[batch*self.batch_size:(batch+1)*self.batch_size]
                batch_y = y[batch*self.batch_size:(batch+1)*self.batch_size]

                _, loss,out = self.sess.run([self.optimizer,self.loss,self.enc_out],feed_dict={self.x:batch_x,self.y:batch_y})
            # 每个epoch输出一下信息，loss用的是该epoch最后一个batch的loss,还有个准确率
            print ("train loss: ",loss)
            correct_prediction = tf.equal(tf.argmax(batch_y, 1), tf.argmax(out, 1))     # argmax: one-hot to label
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))               # 分类准确率
            print ("train acc: ",self.sess.run(acc,feed_dict={self.x:batch_x,self.y:batch_y}) )

