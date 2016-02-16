__author__ = 'vinod'
import pickle
import cPickle as cp
import gzip
import time

import numpy as np
import theano.tensor as T

import climin.stops

import climin.initialize
import climin.project
import climin.schedule
import climin.mathadapt as ma

from breze.learn.autoencoder_rc import ContractiveAutoEncoder, DenoisingAutoEncoder
from sklearn.preprocessing import scale
from breze.learn.data import one_hot
import breze.learn.base
import os


import matplotlib.pyplot as plt
seed = 3453
np.random.seed(seed)

#datafile = '/home/vinod/Downloads/qm7b_bob/qm7b_bob.pkl'
datafile = 'qm7.pkl'
dataset = pickle.load(open(datafile, 'rb'))
split = 1
#P = np.hstack(dataset['P'][range(0, split)+ range(split+1, 5)].flatten())
P = dataset['P'][range(0, split)+ range(split+1, 5)].flatten()
X = dataset['X'][P]
Z = dataset['T'][P]
#Z = Z[:, 0]
Z = Z.reshape(Z.shape[0], 1)

train_labels = Z
Ptest = dataset['P'][split]
TX = dataset['X'][Ptest]
TZ = dataset['T'][Ptest]
#TZ = TZ[:, 0]
TZ = TZ.reshape(TZ.shape[0], 1)
test_labels = TZ
Z = scale(Z, axis=0)
TZ = scale(TZ, axis=0)
weights = []

'''
mean = X.mean(axis=0)
std = (X - mean).std()
X = (X - mean) / std
TX = (TX - mean)/ std
'''

batch_size = 25
#max_iter = max_passes * X.shape[ 0] / batch_size
max_iter = 1000
n_report = X.shape[0] / batch_size


stop = climin.stops.AfterNIterations(max_iter)
pause = climin.stops.ModuloNIterations(n_report)



optimizer = 'gd', {'step_rate': 0.001, 'momentum': 0}

typ = 'dae'
if typ == 'cae':
    m = ContractiveAutoEncoder(X.shape[1], [2700], X,
            hidden_transfers=['tanh'], out_transfer='identity', loss='squared', optimizer=optimizer,
                               batch_size=batch_size, max_iter=max_iter, c_jacobian=1)

elif typ == 'dae':
    m = DenoisingAutoEncoder(X.shape[1], [2500], X,
            hidden_transfers=['tanh'], out_transfer='identity', loss='squared', optimizer=optimizer,
                             batch_size=batch_size, max_iter=max_iter, noise_type='gauss', c_noise=.2)

#climin.initialize.randomize_normal(m.parameters.data, 0, 1 / np.sqrt(m.n_inpt))


#m.init_weights()
#Transform the test data
#TX = m.transformedData(TX)
TX = np.array([m.transformedData(TX) for _ in range(10)]).mean(axis=0)
print TX.shape

losses = []
print 'max iter', max_iter



X, TX = [breze.learn.base.cast_array_to_local_type(i) for i in (X, TX)]

for layer in m.mlp.layers:
    weights.append(m.parameters[layer.weights])


weight_decay = ((weights[0]**2).sum()
                    + (weights[1]**2).sum()
                   )


weight_decay /= m.exprs['inpt'].shape[0]
m.exprs['true_loss'] = m.exprs['loss']
c_wd = 0.1
m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

#m.parameters.data[...] = np.random.normal(0, 0.01, m.parameters.data.shape)


# Transform the test data
#TX = m.transformedData(TX)
#m.init_weights()
#TX = np.array([TX for _ in range(10)]).mean(axis=0)
print TX.shape

losses = []
print 'max iter', max_iter

X, TX = [breze.learn.base.cast_array_to_local_type(i) for i in (X, TX)]



start = time.time()
# Set up a nice printout.
keys = '#', 'seconds', 'loss', 'val loss', 'mae_train', 'rmse_train', 'mae_test', 'rmse_test'
max_len = max(len(i) for i in keys)
header = '\t'.join(i for i in keys)
print header
print '-' * len(header)
results = open('result_hp.txt', 'a')
results.write(header + '\n')
results.write('-' * len(header) + '\n')
results.close()

'''
EXP_DIR = os.getcwd()
base_path = os.path.join(EXP_DIR, "pars_hp"+str(counter)+".pkl")
n_iter = 0

if os.path.isfile(base_path):
    with open("pars_hp"+str(counter)+".pkl", 'rb') as tp:
        n_iter, best_pars, best_loss = cp.load(tp)
        m.parameters.data[...] = best_pars
'''
for i, info in enumerate(m.powerfit((X, ), (TX, ), stop, pause)):
    if info['n_iter'] % n_report != 0:
        continue

    passed = time.time() - start
    losses.append((info['loss'], info['val_loss']))
    info.update({
        'time': passed,

    })
    best_pars = info['best_pars']
    best_loss = info['best_loss']

    row = '%(n_iter)i\t%(time)g\t%(loss)f\t%(val_loss)f' % info
    #results = open('result_hp.txt', 'a')
    print row
    #results.write(row + '\n')
    #results.close()
    '''
    with open("pars_hp"+str(counter)+".pkl", 'wb') as fp:
        cp.dump((info['n_iter'], info['best_pars'], info['best_loss']), fp)
    with open("hps"+str(counter)+".pkl", 'wb') as tp:
        cp.dump((step, momentum, decay, n_hidden, hidden_transfers, par_std, batch_size, counter, info['n_iter']), tp)
    '''

m.parameters.data[...] = best_pars

f_reconstruct = m.function([m.exprs['inpt']], m.mlp.layers[0].output)
TR = f_reconstruct(TX)
print TR.shape


'''
cp.dump((best_pars, best_loss), open("best_pars"+str(counter)+".pkl", 'wb'))
f_L = m.function([m.vae.inpt], m.vae.recog.stt)
#train data
L = f_L(X)
#test data
LT = f_L(TX)
mydict = {'X': np.array(L), 'Z': Z, 'TZ': TZ, 'TX': np.array(LT), 'P': CV}
output = open("autoencoder"+str(counter)+".pkl", 'wb')
cp.dump(mydict, output)
output.close()
'''


'''
import cPickle
import gzip

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T

import climin.stops
import climin.initialize
import math


from breze.learn import autoencoder


def scale_cols_to_unit_interval(arr):
    arr = arr.copy()
    arr -= arr.min(axis=0)
    arr /= arr.max(axis=0)
    arr *= 255
    return arr.astype('uint8')


datafile = 'mnist.pkl.gz'
# Load data.

with gzip.open(datafile,'rb') as f:
    train_set, val_set, test_set = cPickle.load(f)

X, Z = train_set
VX, VZ = val_set
TX, TZ = test_set

image_dims = 28, 28

method = 'denoising'     # One of: basic, sparse, contractive, rica, denoising

n_features = 32
max_passes = 250

feature_dims = int(math.ceil(np.sqrt(n_features))), int(math.ceil(np.sqrt(n_features)))

batch_size = 250
hidden_transfer = 'sigmoid'
out_transfer = 'sigmoid'
loss = 'squared'
optimizer = 'rmsprop', {'steprate': 0.001, 'momentum': 0.9, 'decay': 0.9}
#optimizer = 'gd', {'steprate': 0.01, 'momentum': 0.95, 'momentum_type': 'nesterov'}
#optimizer = 'lbfgs'
sparsify = False
par_std = 1e-1
tied_weights = True

losses = []


if method == 'denoising':
    c_noise = .2
    noise_type = 'mask'
    batch_size = 250

    fe = autoencoder.DenoisingAutoEncoder(X.shape[1], [n_features],
                              hidden_transfers=[hidden_transfer], out_transfer=out_transfer,
                              c_noise=c_noise, noise_type=noise_type,
                              batch_size=batch_size, optimizer=optimizer, tied_weights=tied_weights)


'''



