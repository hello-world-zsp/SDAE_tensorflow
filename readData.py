import numpy
from tensorflow.examples.tutorials.mnist import input_data

def load_data(path):
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

    return trainX, trainY, valX, valY, testX, testY