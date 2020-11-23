import pandas as pd
import numpy as np
#from funcs import *
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as seabornInstance
import operator as op
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import cvxpy as cp


# ----------------------------#
#        LOSS FUNCTIONS
# ----------------------------#

def mape(y_test, y_pred):
    return np.mean(np.abs((y_pred-y_test)/y_test))

def mspe(y_test, y_pred):
    return np.mean(np.square((y_pred-y_test)/y_test))

def maape(y_test, y_pred):
    return np.mean(np.arctan(np.abs((y_pred-y_test)/y_test)))

def mae(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))



# ----------------------------------#
#        QUADRATIC OPTIMIZATION
# ----------------------------------#
import cvxopt

def create_M(N):
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i==0:
                if j == 0:
                    M[i,j]=1
                else:
                    M[i,j]=0
            elif (i==j):
                M[i,j]=1
            elif (j == (i-1)):
                M[i,j] = -1
            else:
                M[i,j]=0
    return M

def generate_params(X_train, y_train,k,N,lambda_=10e-15):
    M = create_M(N)
    M_tilde = M.T @ M
    X_tilde = X_train.T @ X_train
    P = X_tilde + lambda_*(M_tilde)
    q = -X_train.T@y_train
    G = -np.identity(N)
    h = np.zeros((N,1))
    for i in range(len(h)):
        h[i] = -0.0000001
    return P, q, G, h

def find_best_index(X_train, X_test, y_train, y_test, loss,N):
    """Returns index of maximum gamma that minimizes the MAPE loss"""
    loss = {}
    for k in range(N):
        P, q, G, h = generate_params(X_train, y_train, k, N, lambda_=10e-15)
        #print('P', P)
        #print('q', q)
        #print('G', G)
        #print('h', h)
        gammas = cvxopt_solve_qp(P,q, G, h)
        #print(X_test.shape)
        #print(gammas.shape)
        if not (gammas is None):
            y_pred = X_test@gammas
            y_pred[y_pred < 1] = 1
            y_pred = np.floor(y_pred)
            loss[k] = mape(y_test,y_pred)
        else:
            loss[k] = 999999999
    return min(loss, key=loss.get)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    cvxopt.solvers.options['show_progress'] = False
    P = .5 * (P + P.T)  # make sure P is symmetric
    #print('new P:', P)
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    #print('is optimal in solution status', 'optimal' not in sol['status'])
    #print('solution', sol['x'])
    #if 'optimal' not in sol['status']:
     #   return None
    #print(np.array(sol['x']).reshape((P.shape[1],)))
    return np.array(sol['x']).reshape((P.shape[1],))



def gauss_filter(x, K, parity='even'):

    # constant
    A = 2*K
    #upper bound of sum
    B = K
    if parity == 'odd':
        A += 1
        B += 1

    const = 1/(2**A)


    # range of k
    r_k = np.arange(-K,B+1)

    # x elements that will see their value change
    r_x = np.arange(K, len(x))

    len_x = len(x)

    # add K last elements mirrored to the x array
    x = np.append(x, x[-K:][::-1])
    x_filtered = []

    for i in range(len_x):
        if i not in r_x:
            x_filtered.append(x[i])
        else:
            # list on which we will save values to be summed to yield new x_tilde_t
            ls = []
            for k in r_k:
                #x_{t-k}
                comb = ncr(A, K+k)
                #print('i: ',i,'k: ',k)
                x_tk = x[i-k]
                #print(comb, x_tk, comb*x_tk)
                #print(ls)
                ls.append(int(comb*x_tk*const))
                #print(ls)
            x_filtered.append(np.sum(ls))
    return x_filtered

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def find_best_K(X, y, parity, with_validation=True, training_size = 7, model='quadratic'):
    """Returns optimal K such that MAPE error is minimized for the gaussian smoothing"""
    X_new = np.zeros(X.shape)
    N = X.shape[1]
    # try windows from 4 to 22 as well as no smoothing
    Ks = np.arange(8,22)
    mapes = []
    dic0 = {}
    dic1 = {}


    for K in Ks:
        #print(K)
        for j in range(X.shape[1]):
            #print(j)
            X_new[:,j]= gauss_filter(X[:,j], K, parity)

        # train on first `tr` days, then  predict 7 next / shiftt of seven until end on dataset

        splits = int(np.floor((X.shape[0] - training_size)/7))

        # list of the mape for a given split, this list is reinitialized for every K
        temp_mapes = []
        y_vals = []
        y_preds = []

        for i in range(splits):

            begin = 7*i
            end = training_size + 7*i

            X_tr = X_new[begin:end,:]
            y_tr = y[begin:end]

            X_te = X_new[end:end+7,:]
            y_te = y[end:end+7]


            index = find_best_index(X_tr, X_te, y_tr, y_te, 'mape', N)
            P, q, G, h = generate_params(X_tr, y_tr, index, N, 10e-15)
            gamma = cvxopt_solve_qp(P, q, G, h)
            y_pred = X_te@gamma
            y_pred[y_pred < 1] = 1
            y_pred = np.floor(y_pred)

            temp_mapes.append(mape(y_te, y_pred))
            #print(mapes)
        #mean = np.mean(temp_mapes)
        #temp_mapes.append(mean)
        #print("fo training size ", training_size, "and K=", K, "we have mean loss of", np.mean(temp_mapes))
        # for K, associate the list of mapes for each split
        dic0[K] = temp_mapes

        #append the mean mape for the given K
        mapes.append(np.mean(temp_mapes))

        # for K, associate the mean mape between all splits
        dic1[K] = np.mean(temp_mapes)

    return Ks[np.argmin(mapes)], np.min(mapes)


def apply_smoothing(X, K, parity):

    new_X = np.zeros(X.shape)

    for j in range(X.shape[1]):
        new_X[:,j] = gauss_filter(X[:,j], K, parity=parity)


    return new_X


def gauss_filter(x, K, parity='even'):

    # constant
    A = 2*K
    #upper bound of sum
    B = K
    if parity == 'odd':
        A += 1
        B += 1

    const = 1/(2**A)

    x_filtered = []

    # x elements that will see their value change
    r_x = np.arange(K, len(x)-K)

    # range of k
    r_k = np.arange(-K,B+1)

    for i in range(len(x)):
        if i not in r_x:
            x_filtered.append(x[i])
        else:
            # list on which we will save values to be summed to yield new x_tilde_t
            ls = []
            for k in r_k:
                #x_{t-k}
                comb = ncr(A, K+k)
                #print('i: ',i,'k: ',k)
                x_tk = x[i-k]
                #print(comb, x_tk, comb*x_tk)
                #print(ls)
                ls.append(int(comb*x_tk*const))
                #print(ls)
            x_filtered.append(np.sum(ls))
    return x_filtered

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom
