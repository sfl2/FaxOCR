import numpy as np
import cv2
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

import six
import os
import sys
import math
from sklearn.decomposition import PCA
import net
import idx2numpy as i2n



xp = np

img_size = 48

train_size = 1711
test_size = 249
N = train_size
N_test = test_size

train_path = "./data/numbers-proceed"
test_path = "./data/mustread-proceed"


x_train = i2n.convert_from_file('./data/new/faxocr-training-48_train_images.idx3')
y_train = i2n.convert_from_file('./data/new/faxocr-training-48_train_labels.idx1').astype('int32')

x_test = i2n.convert_from_file('./data/new/faxocr-mustread-48_train_images.idx3')
y_test = i2n.convert_from_file('./data/new/faxocr-mustread-48_train_labels.idx1').astype('int32')


print x_train.shape
def reshape(data):
    shape = data.shape
    n_d = np.zeros((shape[0],1,shape[1],shape[2]),dtype="float32")
    size = shape[0]
    for i in range(size):
        n_d[i][0] = data[i]
    return n_d

x_train = reshape(x_train)
x_test = reshape(x_test)



"""
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
"""

batchsize = 1
n_epoch = 20

model = L.Classifier(net.VGG())
serializers.load_npz("./cnn.model", model)
#serializers.load_npz("./cnn.state", optimizer)


batchsize = 1

sum_accuracy = 0.0
for i in range(0, N_test, batchsize):
    #print "batch first index : {}".format(i)
    x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                         volatile='on')
    t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                         volatile='on')
    model.predictor.train = False

    #print "{} : {}".format(i,  model.predictor(x).data.reshape(len(t.data),-1))
    #print t.data

    for j in range(len(t.data)):

        if np.argmax(model.predictor(x).data.reshape(batchsize,10,-1)[j]) == t.data[j]:
            continue
        print "{} : {}, {}".format(i+j,  np.argmax(model.predictor(x).data.reshape(len(t.data),-1)[j]),t.data[j])

        for h in range(img_size):
            for w in range(img_size):
                if x_test[i+j][0][h][w]>0:
                    print 1,
                else :
                    print " ",
            print

    loss = model(x, t)
    sum_accuracy += float(model.accuracy.data) * len(t.data)
    #print model.accuracy.data
    #print len(t.data)


    print('accuracy={}/{}'.format(sum_accuracy , N_test))
