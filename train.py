import numpy as np
import cv2
import chainer

import chainer.links as L
from chainer import optimizers
from chainer import serializers

import six
import os
import sys

import net

xp = np

img_size = 48

train_size = 1711
test_size = 249
N = train_size
N_test = test_size

train_path = "./data/numbers-proceed"
test_path = "./data/mustread-proceed"

def getData(path, size, l):
    data = np.zeros((size, 1, l, l), dtype="float32")
    label = np.zeros((size), dtype="int32")
    for root, _,files in os.walk(path):
        i = 0
        for f in files:
            pic = cv2.imread(root+"/"+f)
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            for h in six.moves.range(0,l):
                for w in six.moves.range(0,l):
                    pic[h][w] = (pic[h][w] > 0)
            data[i][0] = pic
            l = int(f.split('-')[0])
            label[i] = l
            i +=1


    return data, label


x_train, y_train = getData(train_path, train_size, img_size)
x_test, y_test = getData(test_path, test_size, img_size)

print y_train

batchsize = 100
n_epoch = 20

model = L.Classifier(net.VGG())
optimizer = optimizers.Adam()
optimizer.setup(model)


for epoch in range(1, n_epoch+1):
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        arr = np.asarray(x_train[perm[i:i+batchsize]])
        x = chainer.Variable(arr)
        t = chainer.Variable(np.asarray(y_train[perm[i:i+batchsize]]))
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))
    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}/{}'.format(
        sum_loss / N_test, math.floor(sum_accuracy) , N_test))



# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
