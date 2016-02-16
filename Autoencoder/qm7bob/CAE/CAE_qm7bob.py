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

from breze.learn.autoencoder import ContractiveAutoEncoder, DenoisingAutoEncoder
from sklearn.preprocessing import scale
from breze.learn.data import one_hot
import breze.learn.base
import os
import warnings
import math


import matplotlib.pyplot as plt


def run_ae(step, momentum, decay, n_hidden, hidden_transfers, par_std, batch_size, counter, X, TX):

    print step, momentum, decay, n_hidden, hidden_transfers, par_std, batch_size, counter
    seed = 3453
    np.random.seed(seed)
    batch_size = batch_size
    max_iter = 5000000
    n_report = X.shape[0] / batch_size
    weights = []


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

    typ = 'cae'
    if typ == 'cae':
        m = ContractiveAutoEncoder(X.shape[1], n_hidden, X,
                hidden_transfers=hidden_transfers, out_transfer='identity', loss='squared', optimizer=optimizer,
                                   batch_size=batch_size, max_iter=max_iter, c_jacobian=1)

    elif typ == 'dae':
        m = DenoisingAutoEncoder(X.shape[1], n_hidden, X,
                hidden_transfers=hidden_transfers, out_transfer='identity', loss='squared', optimizer=optimizer,
                                 batch_size=batch_size, max_iter=max_iter, noise_type='gauss', c_noise=.2)


    climin.initialize.randomize_normal(m.parameters.data, 0, par_std)


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


    start = time.time()
    # Set up a nice printout.
    keys = '#', 'seconds', 'loss', 'val loss'
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

        if 0 <= info['val_loss'] < 1:
            with open("pars_bestloss.pkl", 'wb') as fp:
                cp.dump((info['n_iter'], info['best_pars'], info['best_loss']), fp)
            m.parameters.data[...] = info['best_pars']
            f_L = m.function([m.vae.inpt], m.vae.recog.stt)
            L = f_L(X)
            LT = f_L(TX)
            mydict = {'X': np.array(L), 'Z': Z, 'TZ': TZ, 'TX': np.array(LT), 'P': CV}
            output = open("autoencoder_best.pkl", 'wb')
            cp.dump(mydict, output)
            output.close()

        if math.isnan(info['loss']) == True:
            n_iter = info['n_iter']
            break

        passed = time.time() - start
        losses.append((info['loss'], info['val_loss']))
        info.update({
            'time': passed,


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
    i = m.mlp.layers.__len__() - 2
    f_reconstruct = m.function([m.exprs['inpt']], m.mlp.layers[i].output)
    L = f_reconstruct(X)
    LT = f_reconstruct(TX)
    print L.shape, LT.shape
    mydict = {'X': np.array(L), 'Z': Z, 'TZ': TZ, 'TX': np.array(LT), 'P': CV}
    output = open("autoencoder"+str(counter)+".pkl", 'wb')
    cp.dump(mydict, output)
    output.close()



if __name__ == '__main__':

    datafile = '/home/hpc/pr63so/ga93yih2/Dataset/qm7b_bob.pkl'
    dataset = pickle.load(open(datafile, 'rb'))
    split = 1
    CV = dataset['P']
    P = np.hstack(dataset['P'][range(0, split)+ range(split+1, 5)])
    X = dataset['B'][P]
    Z = dataset['T'][P]
    Ptest = dataset['P'][split]
    TX = dataset['B'][Ptest]
    TZ = dataset['T'][Ptest]

    weights = []

    step_rate = [0.0001, 0.0005, 0.005,0.001,0.00001,0.00005]
    momentum = [0, 0.9, 0.09]
    decay = [0.9, 0.95, 0.99]
    step_rate_max = 0.05
    step_rate_min = 1e-7

    n_hidden = [[100],[200],[300]]


    hidden_transfers = [['tanh']]

    par_std = [1.5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_size = [25]


    counter = 0
    counter1 = 0

    step_rate1 = [0]
    momentum1 = [0]
    decay1 = [0]
    n_hidden1 = [[0, 0]]
    hidden_transfers1 = [['']]
    par_std1 = [0]
    batch_size1 = [0]

    while 1==1:

        for i in range(1, 100):
            if os.path.isfile(os.path.join(os.getcwd(),"hps"+str(i)+".pkl")):
                with open("hps"+str(i)+".pkl", 'rb') as tp:
                    step_rate1, momentum1, decay1, n_hidden1, hidden_transfers1, par_std1, batch_size1, counter1, n_iter1 = cp.load(tp)
                if (n_iter1 > 0) and (n_iter1 <= 5000000):
                    run_ae(step_rate1, momentum1, decay1, n_hidden1, hidden_transfers1, par_std1, batch_size1, counter1, X, TX)

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
                run_ae(step_rate[step_rate_ind], momentum[momentum_ind], decay[decay_ind], n_hidden[n_hidden_ind], hidden_transfers[hidden_transfers_ind], par_std[par_std_ind], batch_size[batch_size_ind], counter, X, TX)










