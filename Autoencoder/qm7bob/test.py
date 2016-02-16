import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
step = 25.0
noise = 1.0
maximum = 0
seed = 3500
np.random.seed(seed)


dataset = pickle.load(open('autoencoder25.pkl','rb'))
X = dataset['X']
print 'X shape',X.shape


def transformedData(traindata):
    X = normalize(expand(realize(traindata)))
    return X

def realize(X):
    def _realize_(x):
        #inds = np.argsort(-(x**2).sum(axis=0)**.5+np.random.normal(0, noise, x[0].shape))
        #x = x[inds,:][:,inds]*1
        a = np.random.permutation(x)
        x = np.random.permutation(a.T).T
        return x
    return np.array([_realize_(z) for z in X])

def expand(X):
    Xexp = []
    for i in range(X.shape[1]):
        for k in np.arange(0, maximum[i] + step, step):
            Xexp += [np.tanh((X[:, i] - k) / step)]
    return np.array(Xexp).T

def normalize(X):
    return (X - mean) / std

for _ in range(10):
    maximum = np.maximum(maximum, realize(X).max(axis=0))


dim_max = expand(realize(X))
mean = dim_max.mean(axis=0)
std = (dim_max - mean).std()
print 'dim_max',dim_max.shape
print 'mean',mean
print 'std',std

traindata = transformedData(X)
print traindata.shape



