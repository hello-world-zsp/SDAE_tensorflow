# -*- coding: utf8 -*-
import tensorflow as tf
import sys
import ast
import os

from utils import *
from MLP import *
from MLPrec import *
from SDAE import *
from readData import *

## 都改成flag吧！
flags = tf.app.flags
# flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
# FLAGS = flags.FLAGS
sdae_args = {
        "noise"     : .1,
        "n_nodes"   : (225, 100, 49),
        "learning_rate": (.01, 0.01, 0.001),
        "n_epochs"  : (300, 150, 150),
        "data_dir": 'data',
        "batch_size": 50,
        "num_show"  :100,
        "rho"       :(0.05, 0.02, 0.05),
        "reg_lambda":0.0,
        "sparse_lambda":1
}

mlp_args = {
        "noise"     : .1,
        "n_nodes"   : (225,100),
        "learning_rate": .001,
        "n_epochs"  : 200,
        "data_dir": 'data',
        "batch_size": 50,
}

def main():
    # ------------------ 读参数、数据 ---------------------
    if len(sys.argv) < 2:
        print("Using defaults:\n".format(sys.argv[0]))
    for arg in sys.argv[1:]:        # argv[0]代表代码本身文件路径，因此要从第二个开始取参数。
        k, v = arg.split('=', 1)
        sdae_args[k] = ast.literal_eval(v)

    for k in ('learning_rate', 'n_epochs', 'n_nodes', 'noise', 'rho'):
        sdae_args[k] = solo_to_tuple(sdae_args[k])
    for k in ('learning_rate', 'n_epochs', 'n_nodes', 'noise'):
        mlp_args[k] = solo_to_tuple(mlp_args[k])
    print("Stacked DAE arguments: ")
    for k in sorted(sdae_args.keys()):
        print("\t{:15}: {}".format(k, sdae_args[k]))

    if not os.path.isdir(sdae_args["data_dir"]):
        os.makedirs(sdae_args["data_dir"])
    trainX, trainY,valX, valY, testX, testY = load_data(sdae_args["data_dir"])
    # ----------------------------------------------------
    # ----------------- 模型初始化 ------------------------
    with tf.Session() as sess:
        print("Initializing...")
        sdae = SDAE(sess,trainX.shape[1],is_training = True,**sdae_args)
        print("build model...")
        sdae.build(is_training = True)
        print("training...")
        features = sdae.train(trainX,valX,shuffle=False)

        # mlp = MLPrec(sess, trainX.shape[1], is_training=True, **mlp_args)  # 一个多FC层的 enc-dec结构的网络
        # mlp.build(is_training=True)
        # mlp.train(trainX)


        # mlp = MLP(sess,features.shape[1],is_training = True,**mlp_args)  # 一个多FC层的 enc-dec结构的网络
        # mlp.build(is_training = True)
        # mlp.train(features,trainY[:features.shape[0]],valX,valY)


if __name__ == '__main__':
    main()