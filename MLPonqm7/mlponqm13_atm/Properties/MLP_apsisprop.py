__author__ = 'vinod'

import pickle
import cPickle as cp
import gzip
import time
import os
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
from apsis.models.parameter_definition import *
from apsis.assistants.lab_assistant import ValidationLabAssistant, BasicLabAssistant
from apsis.utilities.benchmark_functions import branin_func
from apsis.utilities.logging_utils import get_logger
import breze.learn.base
import warnings
import dill
from apsis.utilities.plot_utils import plot_lists
import matplotlib.pyplot as plt
import math

logger = get_logger("apsis.qm7")

start_time = None


def load_qm7():
    datafile = '/home/hpc/pr63so/ga93yih2/gdb13/gdb13_atm.pkl'
    dataset = pickle.load(open(datafile, 'rb'))
    split = 1
    P = dataset['P'][range(0, split)+ range(split+1, 5)].flatten()
    X = dataset['X'][P]
    Z = dataset['Z'][P]
    #Z = Z.reshape(Z.shape[0], 1)
    train_labels = Z

    Ptest = dataset['P'][split]
    TX = dataset['X'][Ptest]
    TZ = dataset['Z'][Ptest]
    #TZ = TZ.reshape(TZ.shape[0], 1)
    test_labels = TZ
    Z = scale(Z, axis=0)
    TZ = scale(TZ, axis=0)


    return X, Z, TX, TZ, train_labels, test_labels


def do_one_eval(X, Z, TX, TZ, test_labels, train_labels, step_rate, momentum, decay, c_wd, counter, opt):
    seed = 3453
    np.random.seed(seed)
    max_passes = 200
    batch_size = 25
    max_iter = 50000
    n_report = X.shape[0] / batch_size
    weights = []
    optimizer = 'gd', {'step_rate': step_rate, 'momentum': momentum, 'decay': decay}


    stop = climin.stops.AfterNIterations(max_iter)
    pause = climin.stops.ModuloNIterations(n_report)
    # This defines our NN. Since BayOpt does not support categorical data, we just
    # use a fixed hidden layer length and transfer functions.

    typ = 'plain'
    if typ == 'plain':
        m = Mlp(2100, [800, 800], 15, X, Z,
                hidden_transfers=['tanh', 'tanh'], out_transfer='identity', loss='squared', optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)
    elif typ == 'fd':
        m = FastDropoutNetwork(2100, [800, 800], 14, X, Z,
                hidden_transfers=['tanh', 'tanh'], out_transfer='identity', loss='squared',
                p_dropout_inpt=.1,
                p_dropout_hiddens=.2,
                optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)

    #climin.initialize.randomize_normal(m.parameters.data, 0, 1e-3)


    # Transform the test data
    #TX = m.transformedData(TX)
    TX = np.array([m.transformedData(TX) for _ in range(10)]).mean(axis=0)
    losses = []
    print 'max iter', max_iter

    m.init_weights()

    for layer in m.mlp.layers:
        weights.append(m.parameters[layer.weights])


    weight_decay = ((weights[0]**2).sum()
                        + (weights[1]**2).sum()
                        + (weights[2]**2).sum())

    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = c_wd
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay


    mae = T.abs_((m.exprs['output'] * np.std(train_labels, axis=0) + np.mean(train_labels, axis=0))- m.exprs['target']).mean(axis=0)
    f_mae = m.function(['inpt', 'target'], mae)

    rmse = T.sqrt(T.square((m.exprs['output'] * np.std(train_labels, axis=0) + np.mean(train_labels, axis=0))- m.exprs['target']).mean(axis=0))
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
    results.write("%f %f %f %f %s" %(step_rate, momentum, decay, c_wd, opt))
    results.write('\n')
    results.close()

    EXP_DIR = os.getcwd()
    base_path = os.path.join(EXP_DIR, "pars_hp_"+opt+str(counter)+".pkl")
    n_iter = 0

    if os.path.isfile(base_path):
        with open("pars_hp_"+opt+str(counter)+".pkl", 'rb') as tp:
            n_iter, best_pars = dill.load(tp)
            m.parameters.data[...] = best_pars

    if n_iter == 0:
        print 'am here'
        with open("pars.pkl", 'rb') as tp:
            n_iter_dummy, best_pars = dill.load(tp)
            m.parameters.data[...] = best_pars

    for i, info in enumerate(m.powerfit((X, Z), (TX, TZ), stop, pause)):
        if info['n_iter'] % n_report != 0:
            continue
        passed = time.time() - start
        if math.isnan(info['loss']) == True:
            info.update({'mae_test': f_mae(TX, test_labels)})
            n_iter = info['n_iter']
            break
        losses.append((info['loss'], info['val_loss']))
        info.update({
            'time': passed,
            'mae_train': f_mae(m.transformedData(X), train_labels),
            'rmse_train': f_rmse(m.transformedData(X), train_labels),
            'mae_test': f_mae(TX, test_labels),
            'rmse_test': f_rmse(TX, test_labels)

        })
        info['n_iter'] += n_iter
        #row = '%(n_iter)i\t%(time)g\t%(loss)f\t%(val_loss)f\t%(mae_train)g\t%(rmse_train)g\t%(mae_test)g\t%(rmse_test)g' % info
        row = '%(n_iter)i\t%(time)g\t%(loss)f\t%(val_loss)f\t%(mae_train)s\t%(rmse_train)s\t%(mae_test)s\t%(rmse_test)s' % info
        results = open('result.txt','a')
        print row
        results.write(row + '\n')
        results.close()
        with open("pars_hp_"+opt+str(counter)+".pkl", 'wb') as fp:
            dill.dump((info['n_iter'], info['best_pars']), fp)
        with open("apsis_pars_"+opt+str(counter)+".pkl", 'rb') as fp:
            LAss, opt, step_rate, momentum, decay, c_wd, counter, n_iter1, result1 = dill.load(fp)
        n_iter1 = info['n_iter']
        result1 = info['mae_test']
        with open("apsis_pars_"+opt+str(counter)+".pkl", 'wb') as fp:
            dill.dump((LAss, opt, step_rate, momentum, decay, c_wd, counter, n_iter1, result1), fp)


    return info['mae_test'], info['n_iter']


def do_evaluation(LAss, opt, X, Z, TX, TZ, test_labels, train_labels, counter):
    """
    Evaluates opt on a certain parameter set.
    Parameters
    ----------
    LAss : ValidationLabAssistant
        The LAss to use for updates.
    opt : string
        The name of the experiment to use here.
    X, Z : matrix
        Feature and Target matrices of the training set, one-hot encoded.
    VX, VZ : matrix
        Feature and Target matrices of the validation set, one-hot encoded.
    """
    to_eval = LAss.get_next_candidate(opt)
    step_rate = to_eval.params["step_rate"]
    momentum = to_eval.params["momentum"]
    decay = to_eval.params["decay"]
    c_wd = to_eval.params["c_wd"]
    print opt,step_rate, momentum, decay, c_wd
    with open("apsis_pars_"+opt+str(counter)+".pkl", 'wb') as fp:
        dill.dump((LAss, opt, step_rate, momentum, decay, c_wd, counter, 0, 0), fp)
    result, n_iter = do_one_eval(X, Z, TX, TZ,test_labels, train_labels, step_rate, momentum, decay, c_wd, counter, opt)
    to_eval.result = result
    LAss.update(opt, to_eval)
    with open("apsis_pars_"+opt+str(counter)+".pkl", 'wb') as fp:
        dill.dump((LAss, opt, step_rate, momentum, decay, c_wd, counter, n_iter, result), fp)


def init(random_steps, steps, cv=1):
    """
    Parameters
    ----------
    random_steps : int
        The number of initial random steps. This is shared by all optimizers.
    steps : int
        The number of total steps. Must be greater than random_steps.
    cv : int
        The number of crossvalidations for evaluation.
    """
    X, Z, TX, TZ, train_labels, test_labels = load_qm7()
    # The current parameter definitions are the more successful ones.
    # However, this can be compared by changing the lines commented.
    param_defs = {
        #"step_rate": MinMaxNumericParamDef(0, 1),
        "step_rate": AsymptoticNumericParamDef(0, 1),
        #"momentum": MinMaxNumericParamDef(0, 1),
        "momentum": AsymptoticNumericParamDef(0, 1),
        'decay': MinMaxNumericParamDef(0, 1),
        "c_wd": MinMaxNumericParamDef(0, 1)
    }

    #LAss = ValidationLabAssistant(cv=cv)
    LAss = BasicLabAssistant()
    #experiments = ["random_qm7", "bay_mnist_ei_L-BFGS-B"]#, "bay_mnist_ei_rand"]
    #LAss.init_experiment("random_qm7", "RandomSearch", param_defs, minimization=True)

    experiments = ["RandomSearch", "BayOpt"]
    optimizer_arguments= [{}, {"initial_random_runs": random_steps}]

    exp_ids = []
    for i, o in enumerate(experiments):
        exp_id = LAss.init_experiment(o, o, param_defs,
                         minimization=True, optimizer_arguments=optimizer_arguments[i])#{"multiprocessing": "none"})
        exp_ids.append(exp_id)


    global start_time
    start_time = time.time()
    #First, the random steps
    for i in range(steps*cv):
        for opt in exp_ids:
            print opt

            if os.path.isfile(os.path.join(os.getcwd(),"apsis_pars_"+opt+str(i)+".pkl")):
                with open("apsis_pars_"+opt+str(i)+".pkl", 'rb') as fp:
                    LAss, opt, step_rate, momentum, decay, c_wd, counter, n_iter, result = dill.load(fp)
                    print n_iter
                if counter > steps*cv:
                    break
                if n_iter >= 50000:
                    continue
                else:
                    print step_rate, momentum, decay, c_wd, counter
                    do_one_eval(X, Z, TX, TZ,test_labels, train_labels, step_rate, momentum, decay, c_wd, counter, opt)
            else:
                print("%s\tBeginning with random initialization. Step %i/%i" %(str(time.time()-start_time), i, steps*cv))
                do_evaluation(LAss, opt, X, Z, TX, TZ, test_labels, train_labels, i)
                with open("apsis_pars_"+opt+str(i)+".pkl", 'rb') as fp:
                    a, b, c, d, e, f, g, h, result = dill.load(fp)
            #plots = open('plot.txt', 'a')
            #plots.write("%d %s %f" %(i+1, opt, result))
            #plots.write("\n")
            #plots.close()


    '''
    for opt in exp_ids:
        logger.info("Best %s score:  %s" %(opt, LAss.get_best_candidate(opt)))
        print("Best %s score:  %s" %(opt, LAss.get_best_candidate(opt)))
        results = open('result.txt', 'a')
        results.write("Best %s score:  %s" %(opt, LAss.get_best_candidate(opt)))
        results.close()

    f = 'plot.txt'
    steps = np.array([x.split(' ')[0] for x in open(f).readlines()])
    filter1 = [line.strip() for line in open(f) if exp_ids[0] in line]
    mae1 = np.array([x.split(' ')[2] for x in filter1])
    filter2 = [line.strip() for line in open(f) if exp_ids[1] in line]
    mae2 = np.array([x.split(' ')[2] for x in filter2])
    plt.plot(steps, mae1)
    plt.plot(steps, mae2)
    plt.show()
    '''

if __name__ == '__main__':
    init(20, 50, 1)
