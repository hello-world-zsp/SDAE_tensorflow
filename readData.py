# -*- coding: utf8 -*-

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import csv
from sklearn.preprocessing import normalize


def load_data(path,shuffle=False):
    mnist = input_data.read_data_sets(path, one_hot=True)
    trainX = mnist.train.images     # ndarray
    trainY = mnist.train.labels
    trainY = trainY.astype('float32')
    valX = mnist.validation.images
    valY = mnist.validation.labels
    valY = valY.astype('float32')
    testX = mnist.test.images
    testY = mnist.test.labels
    testY = testY.astype('float32')
    if shuffle:
        r = np.random.permutation(len(trainY))
        trainX = trainX[r,:]
        trainY = trainY[r,:]

        r = np.random.permutation(len(valY))
        valX = valX[r,:]
        valY = valY[r,:]

        r = np.random.permutation(len(testY))
        testX = testX[r,:]
        testY = testY[r,:]

    return trainX, trainY, valX, valY, testX, testY


def load_data_batch(path,start=0,batch_size=None,shuffle=False):
    if batch_size == None:
        X = np.load("./" + path).astype(np.float32)[:, :, :, None]
    else:
        X = np.load("./" + path).astype(np.float32)[start:start+batch_size, :]
    X = X/255
    if len(X.shape)>2:
        X = np.reshape(X,[X.shape[0],X.shape[1]*X.shape[2]])
    if shuffle:
        r = np.random.permutation(X.shape[0])
        X = X[r,:]
    trainX = X

    return trainX

def load_data_zi(path,shuffle=False,val_ratio=0.8):
    X = np.load("./" + path + "/jg.npy").astype(np.float32)[:, :, :, None]  # 2994*80*80,没有归一化
    # X = np.load("./"+path+"/simsun80.npy").astype(np.float32)[:,:,:,None]    # 2994*80*80,没有归一化
    X = X/255
    X = np.reshape(X,[X.shape[0],X.shape[1]*X.shape[2]])
    trainX = X[:np.round(val_ratio*X.shape[0])]
    valX = X[np.round(val_ratio * X.shape[0]):]
    return trainX, valX


def load_goods_data(train_ratio = 0.9, shuffle=True, use_cat = True):
    X=np.loadtxt(open("./data/goods_vectors_new.csv","rb"),delimiter=",",skiprows=0)
    if use_cat:
        X = X[:,3:]     # 前3列是id，后15列是one-hot的分类信息
    else:
        Y = X[:, -15:]
        X = X[:, 3:-15]
    nan_loc = np.argwhere(np.isnan(X))
    if len(nan_loc)>0:
        X = np.delete(X,nan_loc[0],axis=0)

    X = normalize(X,axis=1)
    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X = X[idx,:]
        Y = Y[idx,:]
    else :
        idx = range(len(X))
    trainX = X[:np.round(train_ratio * X.shape[0])]
    trainidx = idx[:np.round(train_ratio * X.shape[0])]
    trainY = Y[:np.round(train_ratio * X.shape[0])]
    valX = X[np.round(train_ratio * X.shape[0]):]
    validx = idx[np.round(train_ratio * X.shape[0]):]
    valY = Y[np.round(train_ratio * X.shape[0]):]
    return trainX,valX,trainidx,validx,trainY,valY