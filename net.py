import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
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
            f = L.Linear(6*6*64,10)
        )

    def __call__(self, x):
        h = self.b1_1(F.elu(self.c1_1(x)))
        h = self.b1_2(F.elu(self.c1_2(h)))
        h = F.max_pooling_2d(h,2,stride=2)
        h = self.b2_2(F.elu(self.c2_1(h)))
        h = self.b2_2(F.elu(self.c2_2(h)))
        h = F.max_pooling_2d(h,2,stride=2)
        h = self.b3_1(F.elu(self.c3_1(h)))
        h = self.b3_2(F.elu(self.c3_2(h)))
        h = F.max_pooling_2d(h,2,stride=2)
        return self.f(h)
