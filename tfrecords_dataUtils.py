# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
from readData import load_goods_data,load_users_data


def write_tfrecords(path, name, data):
    """
    把数据转存为tfrecord格式
    :param path: 文件存放路径，str
    :param name: 文件存储名称, str
    :param data: 要被保存的数据, ndarray,会保存为np.float32
    :return: None
    """
    tfrecords_filename = path+name+'.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    data = data.astype(np.float32)
    data_raw = data.tostring()                  # 要改成string，作为一个object而不是很多行，才能存
    example = tf.train.Example(features=tf.train.Features(feature={     # 数据填入到Example协议内存块(protocol buffer)，
        'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))}))
    writer.write(example.SerializeToString())                           # 将协议内存块序列化为一个字符串,写入到TFRecords文件
    writer.close()
    print ('tfrecords saved')

# trainX,valX,trainidx,validx,trainY,valY  = load_goods_data(train_ratio=0.8,use_cat=False)
# write_tfrecords(path='./data/', name='goods_vectors_train', data=trainX)
# write_tfrecords(path='./data/', name='goods_vectors_val', data=valX)
# trainX, valX = load_users_data(train_ratio=0.8)
# write_tfrecords(path='./data/', name='user_data_train', data=trainX)
# write_tfrecords(path='./data/', name='user_data_val', data=valX)

def read_and_decode_tfrecords(tfrecords_filename, shape=None):
    """
    读tfrecords格式的文件
    :param tfrecords_filename: 被读文件名称，string
    :param shape: 数据形状, tuple or list
    :return: 读出来的数据, 大小为shape的ndarray，tf.float32
    """
    filename_queue = tf.train.string_input_producer([tfrecords_filename])  # 根据文件名生成一个队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(                                 # 将Example协议内存块解析为张量
        serialized_example,
        features={
            'data_raw': tf.FixedLenFeature([], tf.string),
        })
    data_rec2_tf = tf.decode_raw(features['data_raw'], tf.float32)
    if shape is not None:
        data_rec2_tf = tf.reshape(data_rec2_tf, shape)
    return data_rec2_tf


def data_generator_tfrecords(filename, shape, nb_batch, batch_size=None, shuffle=True):
    """
    训练数据生成器（实际上由于tf.train.batch本身会不断产生batch，这里的yield可以改成return）
    :param filename: 文件名称，从tfrecords文件中读取数据, string
    :param shape: 数据形状, tuple or list
    :param nb_batch: 总batch数目, int
    :param batch_size: batch size, int
    :param shuffle: 是否重排, bool
    :return: 每个batch产生一组训练数据, ndarry, (batch_size x data_dim)
    """
    data = read_and_decode_tfrecords(filename, shape)

    if shuffle:
        r = np.random.permutation(shape[0]).astype(np.int32)
        data = tf.gather(data,r)

    # 如果没定义batch_size,就产生所有数据
    if batch_size is None:
        batch_data = tf.train.batch([data[:]], batch_size=shape[0], enqueue_many=True)
        yield batch_data
    batch = 0
    while batch <= nb_batch:
        batch_data = tf.train.batch([data[:]], batch_size=batch_size, enqueue_many=True)
        batch += 1
        yield batch_data
