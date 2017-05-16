# -*- coding: utf8 -*-
from readData import *
from sklearn.cluster import KMeans
import numpy as np
import itertools

def kmeans_compare(X1=None,X2=None,n_clusters=2):

    assert len(X1) == len(X2), 'number of X1 and X2 must be equal.'
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X1)
    labels_original = kmeans.labels_
    centres_original = np.zeros([n_clusters,len(X1)])   # 每个中心对应哪几个样本
    for i in range(n_clusters):
        centres_original[i] = [1 if label==i else 0 for label in labels_original ]

    kmeans2 = KMeans(n_clusters=n_clusters, random_state=0).fit(X2)
    labels = kmeans2.labels_
    centres = np.zeros([n_clusters,len(X1)])
    for i in range(n_clusters):
        centres[i] = [1 if label==i else 0 for label in labels ]

    most_sim = []
    similarity = 0
    for c in centres_original:
        sim = [sum(np.array(c)*np.array(c2)) for c2 in centres ] # 两个list相同元素个数
        maxidx = np.argmax(sim)
        most_sim.append(maxidx)
        # A交B/A并B
        similarity += sum(np.array(c)*np.array(centres[maxidx]))/sum(np.sign(np.array(c)+np.array(centres[maxidx])))
    similarity /= n_clusters
    print most_sim
    print similarity

    return similarity


# _, X1, _, _ = load_goods_data(shuffle=True)
# _, X2, _, _ = load_goods_data(shuffle=True)
# kmeans_compare(X1,X2,n_clusters=15)