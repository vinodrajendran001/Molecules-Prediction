import cPickle
import itertools
import gzip
import time

import numpy as np
import theano
import theano.tensor as T

import climin.stops
import climin.initialize
import climin.project
import climin.schedule

import breze.arch.util

import breze.learn.base
from breze.learn.data import one_hot

from breze.learn import cnn

import matplotlib.pyplot as plt

datafile = 'mnist.pkl.gz'
# Load data.

with gzip.open(datafile,'rb') as f:
    train_set, val_set, test_set = cPickle.load(f)

X, Z = train_set
VX, VZ = val_set
TX, TZ = test_set

Z = one_hot(Z, 10)
VZ = one_hot(VZ, 10)
TZ = one_hot(TZ, 10)

image_height, image_width = image_dims = 28, 28
print X.shape
print Z.shape
X = X.reshape((-1, 1, image_height, image_width))
VX = VX.reshape((-1, 1, image_height, image_width))
TX = TX.reshape((-1, 1, image_height, image_width))

print X.shape

X, Z, VX, VZ, TX, TZ = [breze.learn.base.cast_array_to_local_type(i) for i in (X, Z, VX, VZ, TX, TZ)]

max_passes = 400
batch_size = 100
max_iter = max_passes * X.shape[0] / batch_size
n_report = X.shape[0] / batch_size

stop = climin.stops.AfterNIterations(max_iter)
pause = climin.stops.ModuloNIterations(n_report)


optimizer = 'adadelta', {'step_rate': 0.001}
#optimizer = dropout_optimizer_conf(steprate_0=1, n_repeats=1)
m = cnn.Lenet(image_height, image_width,
              1,
              n_hiddens_conv=[10, 10],
              filter_shapes=[(5, 5), (5, 5)],
              pool_shapes=[(2, 2), (2, 2)],
              n_hiddens_full=[500],
              n_output=10,
              hidden_transfers_conv=['rectifier', 'rectifier'],
              hidden_transfers_full=['rectifier'],
              out_transfer='softmax',
              loss='cat_ce', optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)

m.parameters.data[...] = np.random.normal(0, 0.01, m.parameters.data.shape)


def f_n_wrong(x, z):
    y = m.predict(x)
    return (y.argmax(axis=1) != z.argmax(axis=1)).sum()


losses = []
print 'max iter', max_iter

start = time.time()
# Set up a nice printout.
keys = '#', 'seconds', 'loss', 'val loss', 'train emp', 'test emp'
max_len = max(len(i) for i in keys)
header = '\t'.join(i for i in keys)
print header
print '-' * len(header)

for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
    print 'entering for loop'
    if info['n_iter'] % n_report != 0:
        continue
    passed = time.time() - start
    losses.append((info['loss'], info['val_loss']))

    info.update({
        'time': passed,
        'train_emp': f_n_wrong(X, Z),
        'test_emp': f_n_wrong(TX, TZ),
    })
    row = '%(n_iter)i\t%(time)g\t%(loss)g\t%(val_loss)g\t%(train_emp)g\t%(test_emp)g' % info
    print row