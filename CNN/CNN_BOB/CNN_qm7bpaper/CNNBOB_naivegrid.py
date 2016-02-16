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


from breze.learn.cnn import SimpleCnn2d, Lenet
from sklearn.preprocessing import scale
import breze.learn.base
import warnings


def run_mlp(step, momentum, decay, n_hidden_full, n_hidden_conv, hidden_full_transfers, hidden_conv_transfers, filter_shapes, pool_size, par_std, batch_size, opt, L2 , counter, X, Z, TX, TZ, image_height, image_width, nouts):

    print step, momentum, decay, n_hidden_full, n_hidden_conv, hidden_full_transfers, hidden_conv_transfers, filter_shapes, pool_size, par_std, batch_size, opt, L2, counter, image_height, image_width, nouts
    seed = 3453
    np.random.seed(seed)
    batch_size = batch_size
    #max_iter = max_passes * X.shape[ 0] / batch_size
    max_iter = 25000000
    n_report = X.shape[0] / batch_size
    weights = []
    #input_size = len(X[0])
    #Normalize
    mean = X.mean(axis=0)
    std = (X - mean).std()
    X = (X - mean) / std
    TX = (TX - mean)/ std

    stop = climin.stops.AfterNIterations(max_iter)
    pause = climin.stops.ModuloNIterations(n_report)


    optimizer = opt, {'step_rate': step, 'momentum': momentum, 'decay': decay}


    typ = 'Lenet'

    if typ == 'Lenet':
        m = Lenet(image_height, image_width, 1, X, Z, n_hiddens_conv=n_hidden_conv, filter_shapes=filter_shapes, pool_shapes=pool_size, n_hiddens_full=n_hidden_full, n_output=nouts, hidden_transfers_conv=hidden_conv_transfers, hidden_transfers_full=hidden_full_transfers, out_transfer='identity', loss='squared', optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)

    elif typ == 'SimpleCnn2d':
        m = SimpleCnn2d(2099, [400, 100], 1, X, Z, TX, TZ,
                hidden_transfers=['tanh', 'tanh'], out_transfer='identity', loss='squared',
                p_dropout_inpt=.1,
                p_dropout_hiddens=.2,
                optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)


    climin.initialize.randomize_normal(m.parameters.data, 0, par_std)
    #m.parameters.data[...] = np.random.normal(0, 0.01, m.parameters.data.shape)


    # Transform the test data
    #TX = m.transformedData(TX)
    m.init_weights()
    TX = np.array([TX for _ in range(10)]).mean(axis=0)
    print TX.shape

    losses = []
    print 'max iter', max_iter



    X, Z, TX, TZ = [breze.learn.base.cast_array_to_local_type(i) for i in (X, Z, TX, TZ)]


    for layer in m.lenet.mlp.layers:
        weights.append(m.parameters[layer.weights])


    weight_decay = ((weights[0]**2).sum()
                        + (weights[1]**2).sum()
                        #+ (weights[2]**2).sum()
                    )


    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = L2
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
            'mae_train': f_mae(X, train_labels),
            'rmse_train': f_rmse(X, train_labels),
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
            cp.dump((step, momentum, decay, n_hidden_full, n_hidden_conv, hidden_full_transfers, hidden_conv_transfers, filter_shapes, pool_size, par_std, batch_size, opt, L2, counter, info['n_iter']), tp)




    m.parameters.data[...] = info['best_pars']
    cp.dump(info['best_pars'], open('best_pars.pkl', 'wb'))

    Y = m.predict(X)
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


    datafile = '/home/hpc/pr63so/ga93yih2/Dataset/qm7b_bob_formatted.pkl'
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
    weights = []
    image_height, image_width = image_dims = 21, 21
    nouts = 1
    X = X.reshape((-1, 1, image_height, image_width))
    TX = TX.reshape((-1, 1, image_height, image_width))

    step_rate = [0.0001, 0.0005, 0.005,0.001,0.00001,0.00005]
    momentum = [0, 0.5, 0.9]
    decay = [0.8,0.95,0.5]
    step_rate_max = 0.05
    step_rate_min = 1e-7
    n_hidden_full = [[200,200],[500,500],[1000,1000],[50,50]]
    #n_hidden_full = [[200]]
    n_hidden_conv = [[8,16],[16,32],[32,64]]
    #n_hidden_conv = [[10,10]]
    hidden_full_transfers = [['tanh','tanh'], ['rectifier','rectifier']]
    #hidden_full_transfers = [['tanh', 'tanh']]
    hidden_conv_transfers = [['tanh','tanh'], ['rectifier','rectifier']]
    #hidden_conv_transfers = [['tanh', 'tanh']]
    #filter_shapes = [[(5,5),(5,5)],[(6,6),(5,5)],[(5,5),(4,4)]]
    filter_shapes = [[(6,6), (5,5)]]
    pool_size = [[(2,2),(2,2)]]
    #pool_size = [[(2,2),(2,2)]]
    par_std = [1e-1, 1e-2,1e-3,1e-4,1e-5]
    batch_size = [25,50]
    optimizer = ['rmsprop','gd']
    L2 = [0.1, 0.01, 0.001, .0001, .00001, .05, .005, .0005]


    counter = 0
    counter1 = 0


    step_rate1 = [0]
    momentum1 = [0]
    decay1 = [0]
    n_hidden_full1 = [[0,0]]
    n_hidden_conv1 = [[0,0]]
    hidden_full_transfers1 = [['','']]
    hidden_conv_transfers1 = [['','']]
    filter_shapes1 = [[(0,0),(0,0)]]
    pool_size1 = [[(0,0),(0,0)]]
    par_std1 = [0]
    batch_size1 = [0]
    optimizer1 = ['']
    L21 = [0]


    while 1==1:

        for i in range(1, 100):
            if os.path.isfile(os.path.join(os.getcwd(),"hps"+str(i)+".pkl")):
                with open("hps"+str(i)+".pkl", 'rb') as tp:
                    step_rate1, momentum1, decay1, n_hidden_full1, n_hidden_conv1, hidden_full_transfers1, hidden_conv_transfers1, filter_shapes1, pool_size1, par_std1, batch_size1, optimizer1, L21, counter1, n_iter1 = cp.load(tp)
                if (n_iter1 > 0) and (n_iter1 <= 25000000):
                    run_mlp(step_rate1, momentum1, decay1, n_hidden_full1, n_hidden_conv1, hidden_full_transfers1, hidden_conv_transfers1, filter_shapes1, pool_size1, par_std1, batch_size1, optimizer1, L21, counter1, X, Z, TX, TZ, image_height, image_width, nouts)

        step_rate_ind = int(np.random.random_sample() * len(step_rate))
        momentum_ind = int(np.random.random_sample() * len(momentum))
        decay_ind = int(np.random.random_sample() * len(decay))
        n_hidden_full_ind = int(np.random.random_sample() * len(n_hidden_full))
        n_hidden_conv_ind = int(np.random.random_sample() * len(n_hidden_conv))
        hidden_full_transfers_ind = int(np.random.random_sample() * len(hidden_full_transfers))
        #hidden_conv_transfers_ind = int(np.random.random_sample() * len(hidden_conv_transfers))
        hidden_conv_transfers_ind = hidden_full_transfers_ind
        filter_shapes_ind = int(np.random.random_sample() * len(filter_shapes))
        pool_size_ind = int(np.random.random_sample() * len(pool_size))
        par_std_ind = int(np.random.random_sample() * len(par_std))
        batch_size_ind = int(np.random.random_sample() * len(batch_size))
        optimizer_ind = int(np.random.random_sample() * len(optimizer))
        L2_ind = int(np.random.random_sample() * len(L2))


        counter = counter1+ 1
        print counter

        if((step_rate[step_rate_ind] != step_rate1) or (momentum[momentum_ind] != momentum1) or (decay[decay_ind] != decay1) or (n_hidden_full[n_hidden_full_ind] != n_hidden_full1) or (n_hidden_conv[n_hidden_conv_ind] != n_hidden_conv1) or (hidden_full_transfers[hidden_full_transfers_ind] != hidden_full_transfers1) or (hidden_conv_transfers[hidden_conv_transfers_ind] != hidden_conv_transfers1) or (filter_shapes[filter_shapes_ind] != filter_shapes1) or (pool_size[pool_size_ind] != pool_size1) or (par_std[par_std_ind] != par_std1) or (batch_size[batch_size_ind] != batch_size1) or (optimizer[optimizer_ind] != optimizer1) or
        (L2[L2_ind] != L21)
        ):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = open('result_hp.txt', 'a')
                results.write('testing: %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' %(step_rate[step_rate_ind], momentum[momentum_ind], decay[decay_ind], n_hidden_full[n_hidden_full_ind], n_hidden_conv[n_hidden_conv_ind], hidden_full_transfers[hidden_full_transfers_ind], hidden_conv_transfers[hidden_conv_transfers_ind], filter_shapes[filter_shapes_ind], pool_size[pool_size_ind], par_std[par_std_ind], batch_size[batch_size_ind], optimizer[optimizer_ind], L2[L2_ind]))
                results.close()
                run_mlp(step_rate[step_rate_ind], momentum[momentum_ind], decay[decay_ind], n_hidden_full[n_hidden_full_ind], n_hidden_conv[n_hidden_conv_ind], hidden_full_transfers[hidden_full_transfers_ind], hidden_conv_transfers[hidden_conv_transfers_ind], filter_shapes[filter_shapes_ind], pool_size[pool_size_ind], par_std[par_std_ind],batch_size[batch_size_ind], optimizer[optimizer_ind], L2[L2_ind],counter, X, Z, TX, TZ, image_height, image_width, nouts)










