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


from breze.learn.mlpbobformatted import Mlp, FastDropoutNetwork
from sklearn.preprocessing import scale

import breze.learn.base
import warnings


def run_mlp(func, step, momentum, X, Z, TX, TZ, wd, opt, counter, arch, batches):

    print func, step, momentum, wd, opt, counter, arch, batches
    seed = 3453
    np.random.seed(seed)
    batch_size = batches
    #max_iter = max_passes * X.shape[ 0] / batch_size
    max_iter = 500
    n_report = X.shape[0] / batch_size
    weights = []
    input_size = len(X[0])

    stop = climin.stops.AfterNIterations(max_iter)
    pause = climin.stops.ModuloNIterations(n_report)


    optimizer = opt, {'step_rate': step, 'momentum': momentum}

    typ = 'plain'
    if typ == 'plain':
        m = Mlp(input_size, arch, 1, X, Z, hidden_transfers=func, out_transfer='identity', loss='squared', optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)

    elif typ == 'fd':
        m = FastDropoutNetwork(2099, arch, 1, X, Z, TX, TZ,
                hidden_transfers=['tanh', 'tanh'], out_transfer='identity', loss='squared',
                p_dropout_inpt=.1,
                p_dropout_hiddens=.2,
                optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)


    #climin.initialize.randomize_normal(m.parameters.data, 0, 1 / np.sqrt(m.n_inpt))


    # Transform the test data
    #TX = m.transformedData(TX)
    TX = np.array([m.transformedData(TX) for _ in range(10)]).mean(axis=0)
    print TX.shape

    losses = []
    print 'max iter', max_iter

    m.init_weights()

    X, Z, TX, TZ = [breze.learn.base.cast_array_to_local_type(i) for i in (X, Z, TX, TZ)]

    for layer in m.mlp.layers:
        weights.append(m.parameters[layer.weights])


    weight_decay = ((weights[0]**2).sum()
                        + (weights[1]**2).sum()
                        + (weights[2]**2).sum()
                    )


    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = wd
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
    results = open('result_hp.txt', 'a')
    results.write(header + '\n')
    results.write('-' * len(header) + '\n')
    results.close()

    EXP_DIR = os.getcwd()
    base_path = os.path.join(EXP_DIR, "pars_hp"+str(counter)+".pkl")
    n_iter = 0

    if os.path.isfile(base_path):
        with open("pars_hp"+str(counter)+".pkl", 'rb') as tp:
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

        info['n_iter'] += n_iter

        row = '%(n_iter)i\t%(time)g\t%(loss)f\t%(val_loss)f\t%(mae_train)g\t%(rmse_train)g\t%(mae_test)g\t%(rmse_test)g' % info
        results = open('result_hp.txt','a')
        print row
        results.write(row + '\n')
        results.close()
        with open("pars_hp"+str(counter)+".pkl", 'wb') as fp:
            cp.dump((info['n_iter'], info['best_pars']), fp)
        with open("hps"+str(counter)+".pkl", 'wb') as tp:
            cp.dump((func, step, momentum, wd, opt, counter, info['n_iter'], arch, batches), tp)



    m.parameters.data[...] = info['best_pars']
    cp.dump(info['best_pars'], open('best_pars.pkl', 'wb'))

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


if __name__ == '__main__':


    datafile = '/home/vinod/Downloads/MLPonTheano/data/qm7b_bob_formatted.pkl'
    dataset = pickle.load(open(datafile, 'rb'))
    split = 1
    P = np.hstack(dataset['P'][range(0, split)+ range(split+1, 5)])

    X = dataset['B'][P]
    Z = dataset['T'][P]
    Z = Z[:,0]
    Z = Z.reshape(Z.shape[0], 1)

    Ptest = dataset['P'][split]
    TX = dataset['B'][Ptest]
    TZ = dataset['T'][Ptest]
    TZ = TZ[:,0]
    TZ = TZ.reshape(TZ.shape[0], 1)
    train_labels = Z
    test_labels = TZ

    Z = scale(Z, axis=0)
    TZ = scale(TZ, axis=0)


    archs = [[400, 300], [400, 100], [400, 200]]
    funcs = [['tanh', 'tanh'], ['rectifier','rectifier']]
    steps =[0.1, 0.01, 0.001, 0.0001, 0.00001]
    momentum = [0.00001, 0.0001, 0.01, 0.001, 0.1]
    batches = [25, 50, 75, 100]
    wds = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    opts = ['gd','rmsprop']

    counter = 0
    counter1 = 0



    func_ind1 = [['', '']]
    step_ind1 = [0]
    momentum_ind1 = [0]
    wd_ind1 = [0]
    opt_ind1 = ['']
    arch_ind1 = [[0, 0]]
    batch_ind1 = [0]




    while 1==1:


        for i in range(1, 60):
            if os.path.isfile(os.path.join(os.getcwd(),"hps"+str(i)+".pkl")):
                with open("hps"+str(i)+".pkl", 'rb') as tp:
                    func_ind1, step_ind1, momentum_ind1, wd_ind1, opt_ind1, counter1, n_iter1, archs1, batches1 = cp.load(tp)
                if (n_iter1 > 0) and (n_iter1 <= 500):
                    run_mlp(func_ind1, step_ind1, momentum_ind1, X, Z, TX, TZ, wd_ind1, opt_ind1, counter1, archs1, batches1)


        arch_ind = int(np.random.random_sample() * len(archs))
        func_ind = int(np.random.random_sample() * len(funcs))
        step_ind = int(np.random.random_sample() * len(steps))
        momentum_ind = int(np.random.random_sample() * len(momentum))
        batch_ind = int(np.random.random_sample() * len(batches))
        wd_ind = int(np.random.random_sample() * len(wds))
        opt_ind = int(np.random.random_sample() * len(opts))
        counter = counter1+ 1
        print counter


        if (func_ind1 != funcs[func_ind]) or (step_ind1 != steps[step_ind]) or (momentum_ind1 != momentum[momentum_ind]) or (wd_ind1 != wds[wd_ind]) or (opt_ind1 != opts[opt_ind]) or (arch_ind1 != archs[arch_ind]) or (batch_ind1 != batches[batch_ind]):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = open('result_hp.txt', 'a')
                results.write('testing: %s,%s,%s,%s,%s,%s,%s\n' %(funcs[func_ind], steps[step_ind], momentum[momentum_ind], wds[wd_ind], opts[opt_ind], archs[arch_ind], batches[batch_ind]))
                results.close()
                run_mlp(funcs[func_ind], steps[step_ind], momentum[momentum_ind], X, Z, TX, TZ, wds[wd_ind], opts[opt_ind], counter, archs[arch_ind], batches[batch_ind])



