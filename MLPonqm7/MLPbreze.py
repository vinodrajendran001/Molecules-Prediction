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

from breze.learn.mlp import Mlp, FastDropoutNetwork
from sklearn.preprocessing import scale
from breze.learn.data import one_hot
import breze.learn.base
import os


import matplotlib.pyplot as plt


datafile = 'qm7.pkl'
dataset = pickle.load(open(datafile, 'r'))
split = 1
P = dataset['P'][range(0, split)+ range(split+1, 5)].flatten()
X = dataset['X'][P]
Z = dataset['T'][P]
Z = Z.reshape(Z.shape[0], 1)
train_labels = Z
Ptest = dataset['P'][split]
TX = dataset['X'][Ptest]
TZ = dataset['T'][Ptest]
TZ = TZ.reshape(TZ.shape[0], 1)
test_labels = TZ
Z = scale(Z, axis=0)
TZ = scale(TZ, axis=0)
weights = []


batch_size = 25
#max_iter = max_passes * X.shape[ 0] / batch_size
max_iter = 2000
n_report = X.shape[0] / batch_size


stop = climin.stops.AfterNIterations(max_iter)
pause = climin.stops.ModuloNIterations(n_report)



optimizer = 'gd', {'step_rate': 0.001, 'momentum': 0}

typ = 'plain'
if typ == 'plain':
    m = Mlp(2099, [400, 100], 1, X, Z,
            hidden_transfers=['tanh', 'tanh'], out_transfer='identity', loss='squared', optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)
elif typ == 'fd':
    m = FastDropoutNetwork(2099, [400, 100], 1, X, Z, TX, TZ,
            hidden_transfers=['tanh', 'tanh'], out_transfer='identity', loss='squared',
            p_dropout_inpt=.1,
            p_dropout_hiddens=.2,
            optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)


#climin.initialize.randomize_normal(m.parameters.data, 0, 1 / np.sqrt(m.n_inpt))


m.init_weights()
#Transform the test data
#TX = m.transformedData(TX)
TX = np.array([m.transformedData(TX) for _ in range(10)]).mean(axis=0)
print TX.shape

losses = []
print 'max iter', max_iter



X, Z, TX, TZ = [breze.learn.base.cast_array_to_local_type(i) for i in (X, Z, TX, TZ)]

for layer in m.mlp.layers:
    weights.append(m.parameters[layer.weights])


weight_decay = ((weights[0]**2).sum()
                    + (weights[1]**2).sum()
                    + (weights[2]**2).sum())


weight_decay /= m.exprs['inpt'].shape[0]
m.exprs['true_loss'] = m.exprs['loss']
c_wd = 0.1
m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay



mae = T.abs_((m.exprs['output'] * np.std(train_labels) + np.mean(train_labels))- m.exprs['target']).mean()
f_mae = m.function(['inpt', 'target'], mae)

rmse = T.sqrt(T.square((m.exprs['output'] * np.std(train_labels) + np.mean(train_labels))- m.exprs['target']).mean())
f_rmse = m.function(['inpt', 'target'], rmse)



start = time.time()
# Set up a nice printout.
keys = '#', 'seconds', 'loss', 'val loss', 'mae_train', 'rmse_train', 'mae_test', 'rmse_test'
max_len = max(len(i) for i in keys)
header = '\t'.join(i for i in keys)
print header
print '-' * len(header)
results = open('result.txt', 'a')
results.write(header + '\n')
results.write('-' * len(header) + '\n')
results.close()


EXP_DIR = os.getcwd()
base_path = os.path.join(EXP_DIR, "pars.pkl")
base_path1 = os.path.join(EXP_DIR, "best_pars.pkl")
n_iter = 0

if os.path.isfile(base_path):
    with open('pars.pkl', 'rb') as tp:
        n_iter, best_pars = cp.load(tp)
        m.parameters.data[...] = best_pars

for i, info in enumerate(m.powerfit((X, Z), (TX, TZ), stop, pause)):

    if info['n_iter'] % n_report != 0:
        continue

    passed = time.time() - start
    losses.append((info['loss'], info['val_loss']))


    info.update({
        'time': passed,
        'mae_train': f_mae(m.transformedData(X), train_labels),
        'rmse_train': f_rmse(m.transformedData(X), train_labels),
        'mae_test': f_mae(TX, test_labels),
        'rmse_test': f_rmse(TX, test_labels)

    })

    #if os.path.isfile(base_path1):
    info['n_iter'] += n_iter

    row = '%(n_iter)i\t%(time)g\t%(loss)f\t%(val_loss)f\t%(mae_train)g\t%(rmse_train)g\t%(mae_test)g\t%(rmse_test)g' % info
    results = open('result.txt','a')
    print row
    results.write(row + '\n')
    results.close()
    with open('pars.pkl', 'wb') as fp:
        cp.dump((info['n_iter'], info['best_pars']), fp)

    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(Z.shape, Z.min(), Z.max()))



m.parameters.data[...] = info['best_pars']
with open('best_pars.pkl', 'wb') as bp:
    cp.dump(info['best_pars'], bp)



Y = m.predict(m.transformedData(X))
TY = m.predict(TX)

output_train = Y * np.std(train_labels) + np.mean(train_labels)
output_test = TY * np.std(train_labels) + np.mean(train_labels)


print 'TRAINING SET\n'
print('MAE:  %5.2f kcal/mol'%np.abs(output_train - train_labels).mean(axis=0))
print('RMSE: %5.2f kcal/mol'%np.square(output_train - train_labels).mean(axis=0) ** .5)


print 'TESTING SET\n'
print('MAE:  %5.2f kcal/mol'%np.abs(output_test - test_labels).mean(axis=0))
print('RMSE: %5.2f kcal/mol'%np.square(output_test - test_labels).mean(axis=0) ** .5)


mae_train = np.abs(output_train - train_labels).mean(axis=0)
rmse_train = np.square(output_train - train_labels).mean(axis=0) ** .5
mae_test = np.abs(output_test - test_labels).mean(axis=0)
rmse_test = np.square(output_test - test_labels).mean(axis=0) ** .5


results = open('result.txt', 'a')
results.write('Training set:\n')
results.write('MAE:\n')
results.write("%5.2f" %mae_train)
results.write('\nRMSE:\n')
results.write("%5.2f" %rmse_train)
results.write('\nTesting set:\n')
results.write('MAE:\n')
results.write("%5.2f" %mae_test)
results.write('\nRMSE:\n')
results.write("%5.2f" %rmse_test)


results.close()



