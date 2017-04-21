# -*- coding: utf8 -*-
import tensorflow as tf
import sys
import ast
import os

from utils import *
from MLP import *
from SDAE import *
from readData import *

## 都改成flag吧！
flags = tf.app.flags
# flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
# FLAGS = flags.FLAGS
sdae_args = {
        "noise"     : .1,
        "n_nodes"   : (225, 100, 49),
        "learning_rate": .01,
        "n_epochs"  : 15,
        "data_dir": 'data',
        "lambda1"   : (.4, .05, .05),
        "batch_size": 50,
        "num_show"  :100
}

def main():
    # ------------------ 读参数、数据 ---------------------
    if len(sys.argv) < 2:
        print("Using defaults:\n".format(sys.argv[0]))
    for arg in sys.argv[1:]:        # argv[0]代表代码本身文件路径，因此要从第二个开始取参数。
        k, v = arg.split('=', 1)
        sdae_args[k] = ast.literal_eval(v)

    for k in ('learning_rate', 'n_epochs', 'n_nodes', 'noise', 'lambda1'):
        sdae_args[k] = solo_to_tuple(sdae_args[k])
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
        #mlp = MLP(sess,trainX.shape[1],is_training = True,**sdae_args)  # 一个多FC层的 enc-dec结构的网络
        sdae = SDAE(sess,trainX.shape[1],is_training = True,**sdae_args)
        print("build model...")
        #mlp.build(is_training = True)
        sdae.build(is_training = True)
        print("training...")
        #mlp.train(trainX,trainY,valX,valY)
        sdae.train(trainX,valX)


if __name__ == '__main__':
    main()