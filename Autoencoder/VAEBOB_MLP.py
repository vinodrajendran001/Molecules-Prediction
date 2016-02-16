__author__ = 'vinod'
import pickle
import cPickle as cp

import time
import os
import numpy as np
import theano.tensor as T

import climin.stops

import climin.initialize
import climin.project
import climin.schedule
import math

from sklearn.preprocessing import scale
import breze.learn.base
import warnings
#from breze.learn.base import UnsupervisedModel
import signal
from breze.learn import sgvb
from breze.learn.sgvb import mlp
#from breze.learn.sgvb.mlp import VariationalAutoEncoder

from matplotlib import pyplot as plt

def run_vae(step, momentum, decay, n_hidden, hidden_transfers, par_std, batch_size, counter, X, TX):

    print step, momentum, decay, n_hidden, hidden_transfers, par_std, batch_size, counter
    seed = 3453
    np.random.seed(seed)
    batch_size = batch_size
    #max_iter = max_passes * X.shape[ 0] / batch_size
    max_iter = 5000000
    n_report = X.shape[0] / batch_size
    weights = []
    #X = X.reshape(X.shape[0], 23*23)
    #TX = TX.reshape(TX.shape[0], 23*23)

    input_size = X.shape[1]
    print input_size

    #Normalize
    mean = X.mean(axis=0)
    std = (X - mean).std()
    X = (X - mean) / std
    TX = (TX - mean)/ std

    stop = climin.stops.AfterNIterations(max_iter)
    pause = climin.stops.ModuloNIterations(n_report)


    optimizer = 'gd', {'step_rate': step, 'momentum': momentum, 'decay': decay}


    fast_dropout = False

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
                    mlp.MlpGaussLatentVAEMixin,
                    mlp.MlpBernoulliVisibleVAEMixin,
                    ):
            pass
        kwargs = {}

    print 'test:', input_size, n_hidden[0], n_hidden[1], n_hidden[2], hidden_transfers[0], hidden_transfers[2]
    m = MyVAE(input_size, n_hidden[0], n_hidden[1], n_hidden[2], hidden_transfers[0], hidden_transfers[2],
          optimizer=optimizer, batch_size=batch_size,
          **kwargs)



    climin.initialize.randomize_normal(m.parameters.data, 0, par_std)
    #m.parameters.data[...] = np.random.normal(0, 0.01, m.parameters.data.shape)


    # Transform the test data
    #TX = m.transformedData(TX)
    #m.init_weights()
    #TX = np.array([TX for _ in range(10)]).mean(axis=0)
    print TX.shape

    losses = []
    print 'max iter', max_iter

    X, TX = [breze.learn.base.cast_array_to_local_type(i) for i in (X, TX)]


    #mae = T.abs_((m.exprs['output'] * np.std(train_labels) + np.mean(train_labels))- m.exprs['target']).mean()
    #f_mae = m.function(['inpt', 'target'], mae)

    #rmse = T.sqrt(T.square((m.exprs['output'] * np.std(train_labels) + np.mean(train_labels))- m.exprs['target']).mean())
    #f_rmse = m.function(['inpt', 'target'], rmse)

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

    EXP_DIR = os.getcwd()
    base_path = os.path.join(EXP_DIR, "pars_hp"+str(counter)+".pkl")
    n_iter = 0

    if os.path.isfile(base_path):
        with open("pars_hp"+str(counter)+".pkl", 'rb') as tp:
            n_iter, best_pars, best_loss = cp.load(tp)
            m.parameters.data[...] = best_pars

    for i, info in enumerate(m.powerfit((X,), (TX,), stop, pause)):
        if info['n_iter'] % n_report != 0:
            continue
        if info['val_loss'] < 0.:
            break

        passed = time.time() - start
        losses.append((info['loss'], info['val_loss']))
        info.update({
            'time': passed,
            #'rmse_train': f_rmse(X, train_labels),
            #'mae_test': f_mae(TX, test_labels),
            #'rmse_test': f_rmse(TX, test_labels)

        })
        best_pars = info['best_pars']
        best_loss = info['best_loss']
        info['n_iter'] += n_iter

        row = '%(n_iter)i\t%(time)g\t%(loss)f\t%(val_loss)f' % info
        results = open('result_hp.txt', 'a')
        print row
        results.write(row + '\n')
        results.close()
        with open("pars_hp"+str(counter)+".pkl", 'wb') as fp:
            cp.dump((info['n_iter'], info['best_pars'], info['best_loss']), fp)
        with open("hps"+str(counter)+".pkl", 'wb') as tp:
            cp.dump((step, momentum, decay, n_hidden, hidden_transfers, par_std, batch_size, counter, info['n_iter']), tp)


    m.parameters.data[...] = best_pars
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



if __name__ == '__main__':

    #datafile = 'qm7.pkl'
    datafile = '/home/vinod/Downloads/MLPonTheano/data/qm7b_bob.pkl'
    dataset = pickle.load(open(datafile, 'rb'))
    split = 1
    CV = dataset['P']
    #P = dataset['P'][range(0, split)+ range(split+1, 5)].flatten()
    P = np.hstack(dataset['P'][range(0, split)+ range(split+1, 5)])
    X = dataset['B'][P]
    Z = dataset['T'][P]
    Ptest = dataset['P'][split]
    TX = dataset['B'][Ptest]
    TZ = dataset['T'][Ptest]

    weights = []

    step_rate = [0.0001, 0.0005, 0.005,0.001,0.00001,0.00005]
    #momentum = [0.99, 0.995, 0.9, 0.95]
    momentum = [0]
    decay = [0.9, 0.95, 0.99]
    step_rate_max = 0.05
    step_rate_min = 1e-7
    #n_hidden = [[[200,200],10,[200,200]],[[500,500],10,[500,500]],[[1000,1000],10,[1000,1000]],[[700,700],10,[700,700]],[[100,100],10,[100,100]],[[50,50],10,[50,50]]]

    n_hidden = [[[200, 200], 64, [200, 200]],[[400, 400], 64, [400, 400]]]

    #n_hidden = [[[200,200],10,[200,200]],[[100,100],10,[100,100]],[[50,50],10,[50,50]]]

    #hidden_transfers = [[['sigmoid', 'sigmoid'], 'identity', ['sigmoid', 'sigmoid']], [['tanh', 'tanh'], 'identity', ['tanh', 'tanh']], [['rectifier', 'rectifier'],'identity', ['rectifier','rectifier']]]

    hidden_transfers = [[['tanh', 'tanh'], 'identity', ['tanh', 'tanh']]]

    par_std = [1.5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_size = [25]


    counter = 0
    counter1 = 0

    step_rate1 = [0]
    momentum1 = [0]
    decay1 = [0]
    n_hidden1 = [[0, 0]]
    hidden_transfers1 = [['', ''], '', ['', '']]
    par_std1 = [0]
    batch_size1 = [0]

    while 1==1:

        for i in range(1, 100):
            if os.path.isfile(os.path.join(os.getcwd(),"hps"+str(i)+".pkl")):
                with open("hps"+str(i)+".pkl", 'rb') as tp:
                    step_rate1, momentum1, decay1, n_hidden1, hidden_transfers1, par_std1, batch_size1, counter1, n_iter1 = cp.load(tp)
                if (n_iter1 > 0) and (n_iter1 <= 10000):
                    run_vae(step_rate1, momentum1, decay1, n_hidden1, hidden_transfers1, par_std1, batch_size1, counter1, X, TX)

        step_rate_ind = int(np.random.random_sample() * len(step_rate))
        momentum_ind = int(np.random.random_sample() * len(momentum))
        decay_ind = int(np.random.random_sample() * len(decay))
        n_hidden_ind = int(np.random.random_sample() * len(n_hidden))
        hidden_transfers_ind = int(np.random.random_sample() * len(hidden_transfers))
        par_std_ind = int(np.random.random_sample() * len(par_std))
        batch_size_ind = int(np.random.random_sample() * len(batch_size))

        counter = counter1+ 1
        print counter

        if((step_rate[step_rate_ind] != step_rate1) or (momentum[momentum_ind] != momentum1) or (decay[decay_ind] != decay1) or (n_hidden[n_hidden_ind] != n_hidden1) or (hidden_transfers[hidden_transfers_ind] != hidden_transfers1) or (par_std[par_std_ind] != par_std1) or (batch_size[batch_size_ind] != batch_size1)
           ):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = open('result_hp.txt', 'a')
                results.write('testing: %s,%s,%s,%s,%s,%s,%s\n' %(step_rate[step_rate_ind], momentum[momentum_ind], decay[decay_ind], n_hidden[n_hidden_ind], hidden_transfers[hidden_transfers_ind], par_std[par_std_ind], batch_size[batch_size_ind]))
                results.close()
                run_vae(step_rate[step_rate_ind], momentum[momentum_ind], decay[decay_ind], n_hidden[n_hidden_ind], hidden_transfers[hidden_transfers_ind], par_std[par_std_ind], batch_size[batch_size_ind], counter, X, TX)










