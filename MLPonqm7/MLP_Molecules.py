# -*- coding: utf-8 -*-

"""Module for learning various types of multilayer perceptrons."""


import itertools

import climin
import climin.util
import climin.gd
from climin.util import minibatches
import numpy as np
import theano
import theano.tensor as T

from breze.arch.component.varprop import loss as vp_loss
from breze.arch.construct import neural
from breze.arch.util import lookup

from breze.arch.construct.simple import SupervisedLoss
from breze.arch.util import ParameterSet
from breze.learn.base import SupervisedModel
import inspect
import itertools
import random
import warnings


# TODO Mlp docs are loss missing

class Mlp(SupervisedModel):
    """Multilayer perceptron class.

    This implementation uses a stack of affine mappings with a subsequent
    non linearity each.

    Parameters
    ----------

    n_inpt : integer
        Dimensionality of a single input.

    n_hiddens : list of integers
        List of ``k`` integers, where ``k`` is thenumber of layers. Each gives
        the size of the corresponding layer.

    n_output : integer
        Dimensionality of a single output.

    hidden_transfers : list, each item either string or function
        Transfer functions for each of the layers. Can be either a string which
        is then used to look up a transfer function in
        ``breze.component.transfer`` or a function that given a Theano tensor
        returns a tensor of the same shape.

    out_transfer : string or function
        Either a string to look up a function in ``breze.component.transfer``
        or a function that given a Theano tensor returns a tensor of the same
        shape.

    optimizer : string, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : integer, None
        Number of examples per batch when calculting the loss
        and its derivatives. None means to use all samples every time.

    imp_weight : boolean
        Flag indicating whether importance weights are used.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.
    """

    def __init__(self, n_inpt, n_hiddens, n_output, train_data, labels, test_data, test_labels,
                 hidden_transfers, out_transfer, loss,
                 imp_weight=False,
                 optimizer='adam',
                 batch_size=None,
                 max_iter=1000, verbose=False):


        self.step = 1.0
        self.noise = 1.0
        self.triuind = (np.arange(23)[:, np.newaxis] <= np.arange(23)[np.newaxis, :]).flatten()
        self.max = 0
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.imp_weight = imp_weight

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.X = train_data
        self.Z = labels
        self.TX = test_data
        self.TZ = test_labels


        for _ in range(10):
            self.max = np.maximum(self.max, self.realize(self.X).max(axis=0))
        print 'max shape', self.max.shape

        self.dim_max = self.transformedData(self.X)
        self.dim_max_test = self.transformedData(self.TX)


        self.mean = self.dim_max.mean(axis=0)
        self.std = (self.dim_max - self.mean).std()
        self.dim_max = (self.dim_max - self.mean)/self.std


        self.mean_test = self.dim_max_test.mean(axis=0)
        self.std_test = (self.dim_max_test - self.mean_test).std()
        self.dim_max_test = (self.dim_max_test - self.mean_test)/self.std_test


        self.n_inpt = self.dim_max.shape[1]

        self.scaling = np.sqrt(self.n_inpt / self.n_output)

        self._init_exprs()

        self.argss = self._make_args(train_data, labels)

    def _make_args(self, X, Z, imp_weight=None):
        from breze.learn.base import cast_array_to_local_type
        batch_size = getattr(self, 'batch_size', None)
        if batch_size is None:
            X, Z = cast_array_to_local_type(X), cast_array_to_local_type(Z)
            if imp_weight is not None:
                imp_weight = cast_array_to_local_type(imp_weight)
                data = itertools.repeat([X, Z, imp_weight])
            else:
                data = itertools.repeat([X, Z])
        elif batch_size < 1:
            raise ValueError('need strictly positive batch size')
        else:
            if imp_weight is not None:
                data = self.iter_minibatches([self.dim_max, self.Z, imp_weight], self.batch_size,
                                        list(self.sample_dim) + [self.sample_dim[0]])
                data = ((cast_array_to_local_type(x),
                         cast_array_to_local_type(z),
                         cast_array_to_local_type(w)) for x, z, w in data)
            else:
                data = self.iter_minibatches([self.X, self.Z], self.batch_size,
                                        self.sample_dim)
                #data = self.iter_minibatches([self.dim_max, self.Z], self.batch_size,
                #                        self.sample_dim)
                data = ((cast_array_to_local_type(x),
                         cast_array_to_local_type(z)) for x, z in data)
        args = ((i, {}) for i in data)
        return args



    def iter_minibatches(self, lst, batch_size, dims, n_cycles=False, random_state=None):
        batches = [minibatches(i, batch_size, d) for i, d in zip(lst, dims)]
        if len(batches) > 1:
            if any(len(i) != len(batches[0]) for i in batches[1:]):
                raise ValueError("containers to be batched have different lengths")
            counter = itertools.count()
            if random_state is not None:
                random.seed(random_state.normal())
            while True:
                indices = [i for i, _ in enumerate(batches[0])]
                while True:
                    random.shuffle(indices)
                    for i in indices:
                        yield (self.transformedData(batches[0][i]), batches[1][i])
                    count = counter.next()
                    if n_cycles and count >= n_cycles:
                        raise StopIteration()


    def _init_exprs(self):
        inpt = T.matrix('inpt')
        target = T.matrix('target')
        parameters = ParameterSet()
        if theano.config.compute_test_value:
            inpt.tag.test_value = np.empty((2, self.n_inpt))
            target.tag.test_value = np.empty((2, self.n_output))

        self.mlp = neural.Mlp(
            inpt,
            self.n_inpt, self.n_hiddens, self.n_output,
            self.hidden_transfers, self.out_transfer,
            declare=parameters.declare)

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
            if theano.config.compute_test_value:
                imp_weight.tag.test_value = np.empty((2, self.n_output))
        else:
            imp_weight = None

        self.loss_layer = SupervisedLoss(
            target, self.mlp.output, loss=self.loss_ident,
            imp_weight=imp_weight,
            declare=parameters.declare,
        )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=self.mlp.output,
                                 loss=self.loss_layer.total,
                                 parameters=parameters)
        self.exprs['imp_weight'] = imp_weight

    def transformedData(self, traindata):
        X = self.expand(self.realize(traindata))
        return X

    def realize(self, X):
        def _realize_(x):
            inds = np.argsort(-(x**2).sum(axis=0)**.5+np.random.normal(0, self.noise, x[0].shape))
            x = x[inds,:][:,inds]*1
            x = x.flatten()[self.triuind]
            return x
        return np.array([_realize_(z) for z in X])

    def expand(self, X):
        Xexp = []
        for i in range(X.shape[1]):
            for k in np.arange(0, self.max[i] + self.step, self.step):
                Xexp += [np.tanh((X[:, i] - k) / self.step)]
        return np.array(Xexp).T




def dropout_optimizer_conf(
        steprate_0=1, steprate_decay=0.998, momentum_0=0.5,
        momentum_eq=0.99, n_momentum_anneal_steps=500,
        n_repeats=500):
    """Return a dictionary suitable for climin.util.optimizer which
    specifies the standard optimizer for dropout mlps."""
    steprate = climin.gd.decaying(steprate_0, steprate_decay)
    momentum = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)

    # Define another time for steprate calculcation.
    momentum2 = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)
    steprate = ((1 - j) * i for i, j in itertools.izip(steprate, momentum2))

    steprate = climin.gd.repeater(steprate, n_repeats)
    momentum = climin.gd.repeater(momentum, n_repeats)

    return 'gd', {
        'steprate': steprate,
        'momentum': momentum,
    }


class DropoutMlp(Mlp):
    """Class representing an MLP that is trained with dropout [D]_.

    The gist of this method is that hidden units and input units are "zerod out"
    with a certain probability.

    References
    ----------
    .. [D] Hinton, Geoffrey E., et al.
           "Improving neural networks by preventing co-adaptation of feature
           detectors." arXiv preprint arXiv:1207.0580 (2012).


    Attributes
    ----------

    Same attributes as an ``Mlp`` object.

    p_dropout_inpt : float
        Probability that an input unit is ommitted during a pass.

    p_dropout_hidden : float
        Probability that an input unit is ommitted during a pass.

    max_length : float
        Maximum squared length of a weight vector into a unit. After each
        update, the weight vectors will projected to be shorter.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 p_dropout_inpt=.2, p_dropout_hiddens=.5,
                 max_length=None,
                 optimizer='adam',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        """Create a DropoutMlp object.


        Parameters
        ----------

        Same attributes as an ``Mlp`` object.

        p_dropout_inpt : float
            Probability that an input unit is ommitted during a pass.

        p_dropout_hiddens : list of floats
            List of which each item gives the probability that a hidden unit
            of that layer is omitted during a pass.

        """
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        super(DropoutMlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss,
            optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
            verbose=verbose)


class FastDropoutNetwork(SupervisedModel):
    """Class representing an MLP that is trained with fast dropout [FD]_.

    This method employs a smooth approximation of dropout training.


    References
    ----------
    .. [FD] Wang, Sida, and Christopher Manning.
            "Fast dropout training."
            Proceedings of the 30th International Conference on Machine
            Learning (ICML-13). 2013.


    Attributes
    ----------

    Same attributes as an ``Mlp`` object.

    p_dropout_inpt : float
        Probability that an input unit is ommitted during a pass.

    p_dropout_hiddens : list of floats
        Each item constitues the probability that a hidden unit of the
        corresponding layer is ommitted during a pass.

    inpt_var : float
        Assumed variance of the inputs. "quasi zero" per default.
    """

    def __init__(self, n_inpt, n_hiddens, n_output, train_data, labels, test_data, test_labels,
                 hidden_transfers, out_transfer, loss,
                 imp_weight=False,
                 optimizer='adam',
                 batch_size=None,
                 p_dropout_inpt=.2,
                 p_dropout_hiddens=.5,
                 max_iter=1000, verbose=False):
        """Create a FastDropoutMlp object.

        Parameters
        ----------

        Same parameters as an ``Mlp`` object.

        p_dropout_inpt : float
            Probability that an input unit is ommitted during a pass.

        p_dropout_hidden : float
            Probability that an input unit is ommitted during a pass.

        max_length : float or None
            Maximum squared length of a weight vector into a unit. After each
            update, the weight vectors will projected to be shorter.
            If None, no projection is performed.
        """
        self.p_dropout_inpt = p_dropout_inpt
        if isinstance(p_dropout_hiddens, float):
            self.p_dropout_hiddens = [p_dropout_hiddens] * len(n_hiddens)
        else:
            self.p_dropout_hiddens = p_dropout_hiddens

        p_dropouts = [p_dropout_inpt] + self.p_dropout_hiddens
        if not all(0 < i < 1 for i in p_dropouts):
            raise ValueError('dropout rates have to be in (0, 1)')


        self.step = 1.0
        self.noise = 1.0
        self.triuind = (np.arange(23)[:, np.newaxis] <= np.arange(23)[np.newaxis, :]).flatten()
        self.max = 0
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.imp_weight = imp_weight
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.X = train_data
        self.Z = labels
        self.TX = test_data
        self.TZ = test_labels
        for _ in range(10):
            self.max = np.maximum(self.max, self.realize(self.X).max(axis=0))
        print 'max shape', self.max.shape

        self.dim_max = self.transformedData(self.X)
        self.dim_max_test = self.transformedData(self.TX)

        self.mean = self.dim_max.mean(axis=0)
        self.std = (self.dim_max - self.mean).std()

        self.dim_max =(self.dim_max-self.mean)/self.std
        self.n_inpt = self.dim_max.shape[1]
        self._init_exprs()
        self.argss = self._make_args(train_data, labels)

    def _make_args(self, X, Z, imp_weight=None):
        from breze.learn.base import cast_array_to_local_type
        batch_size = getattr(self, 'batch_size', None)
        if batch_size is None:
            X, Z = cast_array_to_local_type(X), cast_array_to_local_type(Z)
            if imp_weight is not None:
                imp_weight = cast_array_to_local_type(imp_weight)
                data = itertools.repeat([X, Z, imp_weight])
            else:
                data = itertools.repeat([X, Z])
        elif batch_size < 1:
            raise ValueError('need strictly positive batch size')
        else:
            if imp_weight is not None:
                data = self.iter_minibatches([self.dim_max, self.Z, imp_weight], self.batch_size, list(self.sample_dim) + [self.sample_dim[0]])
                data = ((cast_array_to_local_type(x),
                         cast_array_to_local_type(z),
                         cast_array_to_local_type(w)) for x, z, w in data)
            else:
                data = self.iter_minibatches([self.X, self.Z], self.batch_size,
                                        self.sample_dim)
                #data = self.iter_minibatches([self.dim_max, self.Z], self.batch_size,self.sample_dim)
                data = ((cast_array_to_local_type(x),
                         cast_array_to_local_type(z)) for x, z in data)
        args = ((i, {}) for i in data)
        return args


    def iter_minibatches(self, lst, batch_size, dims, n_cycles=False, random_state=None):
        batches = [minibatches(i, batch_size, d) for i, d in zip(lst, dims)]
        if len(batches) > 1:
            if any(len(i) != len(batches[0]) for i in batches[1:]):
                raise ValueError("containers to be batched have different lengths")
            counter = itertools.count()
            if random_state is not None:
                random.seed(random_state.normal())
            while True:
                indices = [i for i, _ in enumerate(batches[0])]
                while True:
                    random.shuffle(indices)
                    for i in indices:
                        yield (self.transformedData(batches[0][i]), batches[1][i])
                        count = counter.next()
                    if n_cycles and count >= n_cycles:
                        raise StopIteration()


    def _init_exprs(self):
        inpt = T.matrix('inpt')
        target = T.matrix('target')
        parameters = ParameterSet()

        if theano.config.compute_test_value:
            inpt.tag.test_value = np.empty((2, self.n_inpt))
            target.tag.test_value = np.empty((2, self.n_output))

        self.mlp = neural.FastDropoutMlp(
            inpt,
            self.n_inpt, self.n_hiddens, self.n_output,
            self.hidden_transfers, self.out_transfer,
            self.p_dropout_inpt, self.p_dropout_hiddens,
            declare=parameters.declare)

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
            if theano.config.compute_test_value:
                imp_weight.tag.test_value = np.empty((2, self.n_output))
        else:
            imp_weight = None

        output = T.concatenate(self.mlp.outputs, 1)

        self.loss_layer = SupervisedLoss(
            target, output, loss=lookup(self.loss_ident, vp_loss),
            imp_weight=imp_weight,
            declare=parameters.declare,
        )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=output,
                                 loss=self.loss_layer.total,
                                 parameters=parameters)
        self.exprs['imp_weight'] = imp_weight

    def transformedData(self, traindata):
        X = self.expand(self.realize(traindata))
        return X

    def realize(self, X):
        def _realize_(x):
            inds = np.argsort(-(x**2).sum(axis=0)**.5+np.random.normal(0, self.noise, x[0].shape))
            x = x[inds,:][:,inds] * 1
            x = x.flatten()[self.triuind]
            return x
        return np.array([_realize_(z) for z in X])

    def expand(self, X):
        Xexp = []
        for i in range(X.shape[1]):
            for k in np.arange(0, self.max[i] + self.step, self.step):
                Xexp += [np.tanh((X[:, i] - k) / self.step)]
            return np.array(Xexp).T


