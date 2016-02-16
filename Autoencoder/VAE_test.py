import cPickle
import gzip

from breze.learn.data import one_hot
from breze.learn.base import cast_array_to_local_type
from breze.learn.utils import tile_raster_images

import climin.stops


import climin.initialize

from breze.learn.sgvb import mlp
from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np

#import fasttsne


import theano
theano.config.compute_test_value = 'ignore'#'raise'

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

image_dims = 28, 28

X, Z, VX, VZ, TX, TZ = [cast_array_to_local_type(i) for i in (X, Z, VX,VZ, TX, TZ)]


batch_size = 100
#optimizer = 'rmsprop', {'step_rate': 1e-4, 'momentum': 0.95, 'decay': .95, 'offset': 1e-6}
#optimizer = 'adam', {'step_rate': .5, 'momentum': 0.9, 'decay': .95, 'offset': 1e-6}
optimizer = 'gd'



fast_dropout = True

if fast_dropout:
    class MyVAE(mlp.FastDropoutVariationalAutoEncoder,
                mlp.FastDropoutMlpGaussLatentVAEMixin,
                mlp.FastDropoutMlpBernoulliVisibleVAEMixin):
        pass
    kwargs = {
        'p_dropout_inpt': .1,
        'p_dropout_hiddens': [.2],
    }
    print 'yeah'

else:
    class MyVAE(mlp.VariationalAutoEncoder,
                mlp.GaussLatentVAEMixin,
                mlp.BernoulliVisibleVAEMixin,
                ):
        pass
    kwargs = {}


# This is the number of random variables NOT the size of
# the sufficient statistics for the random variables.
n_latents = 10
n_hidden = 512

X = X[:100]
m = MyVAE(X.shape[1], [n_hidden], n_latents, [n_hidden], ['rectifier'] * 1, ['rectifier'] * 1,
          optimizer=optimizer, batch_size=batch_size,
          **kwargs)

#m.exprs['loss'] += 0.001 * (m.parameters.enc_in_to_hidden ** 2).sum() / m.exprs['inpt'].shape[0]

climin.initialize.randomize_normal(m.parameters.data, 0, 1e-2)

#climin.initialize.sparsify_columns(m.parameters['enc_in_to_hidden'], 15)
#climin.initialize.sparsify_columns(m.parameters['enc_hidden_to_hidden_0'], 15)
#climin.initialize.sparsify_columns(m.parameters['dec_hidden_to_out'], 15)

#f_latent_mean = m.function(['inpt'], 'latent_mean')
#f_sample = m.function([('gen', 'layer-0-inpt')], 'output')
#f_recons = m.function(['inpt'], 'output')

m.estimate_nll(X[:10])

max_passes = 250
max_iter = 1000
n_report = X.shape[0] / batch_size

stop = climin.stops.AfterNIterations(max_iter)
pause = climin.stops.ModuloNIterations(n_report)

print 'X shape:', X.shape

for i, info in enumerate(m.powerfit((X,), (VX,), stop, pause)):
    print i, info['loss'], info['val_loss']


m.parameters.data[...] = info['best_pars']


#f_sample = m.function([m.recog_sample], m.vae.gen.sample())
#f_recons = m.function(['inpt'], m.vae.gen.sample())


f_L = m.function([m.vae.inpt], m.vae.recog.stt)
L = f_L(X)

print 'Reconstructed shape:', L.shape
