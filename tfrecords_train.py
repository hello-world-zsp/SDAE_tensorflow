# -*- coding: utf8 -*-
import tensorflow as tf
import sys
import ast
import os

from utils import *
from MLPclassify import *
from tfrecords_SDAE import *
from readData import *
import kmeans

# SDAE网络参数
sdae_args = {
        "noise"     : .1,                               # 噪声水平0-1.0
        "n_nodes"   : (512, 256, 128),                   # 各DAE层节点数
        "learning_rate": (.0001, 0.001, 0.001),        # 各层学习率
        "n_epochs"  : (200, 150, 150),                     # 各层迭代次数
        "rho"       :(0.05, 0.02, 0.02),                 # 各层稀疏水平
        "data_dir": './data/',                         # 数据存储和读取的路径
        "batch_size": 50,                               # batch_size
        "reg_lambda":0.0,                               # 正则化系数
        "sparse_lambda":1.0                             # 稀疏项系数
}

# 用于检验无监督学习特征的分类网络，MLP
mlp_args = {
        "n_nodes"   : (80,),
        "learning_rate": .001,
        "n_epochs"  : 100,
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
        sdae_args[k] = solo_to_tuple(sdae_args[k], n=3)
    print("Stacked DAE arguments: ")
    for k in sorted(sdae_args.keys()):
        print("\t{:15}: {}".format(k, sdae_args[k]))
    if not os.path.isdir(sdae_args["data_dir"]):
        os.makedirs(sdae_args["data_dir"])

    # 获取一下数据信息：有多少条训练数据和验证集数据
    trainX, valX,trainIdx,valIdx,trainY,valY = load_goods_data(train_ratio=0.8,use_cat=False)
    # trainX, valX = load_users_data(train_ratio=0.8)
    # ----------------------------------------------------
    # ----------------- 模型初始化、训练 ------------------------
    with tf.Session() as sess:
        print("Initializing...")
        sdae = SDAE(sess, trainX.shape, valX.shape, is_training=True, **sdae_args)
        print("training...")
        features,val_features = sdae.train()


if __name__ == '__main__':
    main()