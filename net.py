import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            b0 = L.BatchNormalization(1),
            c1_1 = L.Convolution2D(1, 32, ksize=3, stride=1, pad=1),
            b1_1 = L.BatchNormalization(32),
            c1_2 = L.Convolution2D(32, 32, ksize=3, stride=1, pad=1),
            b1_2 = L.BatchNormalization(32),
            c2_1 = L.Convolution2D(32, 32, ksize=3, stride=1, pad=1),
            b2_1 = L.BatchNormalization(32),
            c2_2 = L.Convolution2D(32, 32, ksize=3, stride=1, pad=1),
            b2_2 = L.BatchNormalization(32),
            c3_1 = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1),
            b3_1 = L.BatchNormalization(64),
            c3_2 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1),
            b3_2 = L.BatchNormalization(64),
            f1 = L.Linear(6*6*64, 100),
            f2 = L.Linear(100, 10),
        )
        self.train = True

    def __call__(self, x):
        h = self.b0(x,test=not self.train)
        h = self.b1_1(F.elu(self.c1_1(h)),test=not self.train)
        h = self.b1_2(F.elu(self.c1_2(h)),test=not self.train)
        h = F.max_pooling_2d(h,2,stride=2)
        h = self.b2_2(F.elu(self.c2_1(h)),test=not self.train)
        h = self.b2_2(F.elu(self.c2_2(h)),test=not self.train)
        h = F.max_pooling_2d(h,2,stride=2)
        h = self.b3_1(F.elu(self.c3_1(h)),test=not self.train)
        h = self.b3_2(F.elu(self.c3_2(h)),test=not self.train)
        h = F.max_pooling_2d(h,2,stride=2)
        #print h.data
        h = F.dropout(F.elu(self.f1(h)),train=self.train)
        #print h.data
        return self.f2(h)
