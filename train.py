import numpy as np
import cv2
import chainer
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import six
import os
import sys
import math
from sklearn.decomposition import PCA
import net
import idx2numpy as i2n

#TODO normalize input
#TODO resnet,nin



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

batchsize = 59

n_epoch = 30
model = L.Classifier(net.VGG())
optimizer = optimizers.Adam()
optimizer.setup(model)
"""
serializers.load_npz("./smlp.model", model)
serializers.load_npz("./smlp.state", optimizer)
"""
for epoch in range(1, n_epoch+1):
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    model.predictor.train = True
    for i in range(0, N, batchsize):
        arr = np.asarray(x_train[perm[i:i+batchsize]])
        x = chainer.Variable(arr)
        t = chainer.Variable(np.asarray(y_train[perm[i:i+batchsize]]))
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    print('epoch : {}'.format(epoch))
    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))
    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_test, 1):
            x = chainer.Variable(xp.asarray(x_test[i:i+1]),
                                 volatile='on')
            t = chainer.Variable(xp.asarray(y_test[i:i + 1]),
                                 volatile='on')
            model.predictor.train = False
            loss = model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
    if sum_accuracy > 224:#epoch 19 made the best score(cnn.model)

        # Save the model and the optimizer
        print('save the model')
        serializers.save_npz(str(epoch)+'mlp.model', model)
        print('save the optimizer')
        serializers.save_npz(str(epoch)+'mlp.state', optimizer)
        sys.exit()
    print('test  mean loss={}, accuracy={}/{}'.format(
        sum_loss / N_test, sum_accuracy , N_test))



# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
