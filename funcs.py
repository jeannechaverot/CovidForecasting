import numpy as np
from functools import reduce
import operator as op


#SMOOTHING#

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

def find_best_K(X, y_rec_smoothed, parity):
    """Returns optimal K such that MAPE error is minimized for the gaussian smoothing"""
    X_new = np.zeros(X.shape)
    N = X.shape[1]
    Ks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mapes = []
    pct_80 = int(np.ceil(80*len(X)/100))


    for K in Ks:
        for j in range(X.shape[1]):
            #print(j)
            X_new[:,j]= gauss_filter(X[:,j], K, parity)


        X_train, X_test = X_new[:pct_80], X_new[pct_80:]
        y_train, y_test =y_rec_smoothed[:pct_80], y_rec_smoothed[pct_80:]


        #index = find_best_index(X_train, y_train, X_test, y_test, 'mape',X.shape[1])
        #P, q, G, h = generate_params(X_train, y_train, index,N)
        #gamma = cvxopt_solve_qp(P, q, G, h)
        #y_pred = X_test@gamma

        #mapes.append(mape(y_test, y_pred))
        mapes.append(advancement_val(X_new, y_rec_smoothed)[1])

    return Ks[np.argmin(mapes)], min(mapes)


def simple_exponential_smoothing(x, alpha):
    result = [x[0]] # first value is same as series
    for n in range(1, len(x)):
        result.append(alpha * x[n] + (1 - alpha) * x[n-1])
    return result

def exponential_smoothing(x, rho, K):
    const = (1-rho)/(1-rho**(K+1))
    new_x = []
    
    # range of x
    r_x = np.arange(K, len(x)-K)

    # range of k
    r_k = np.arange(0,K)
    
    for i in range(len(x)):
        if i not in r_x:
            new_x.append(x[i])
        else:
            ls = []
            for k in r_k:
                ls.append(int(const*rho**k*x[i-k]))
            new_x.append(np.sum(ls))
                
    return new_x

def find_best_alpha(X, y, N, model='simple',K=0):
    """Returns optimal alpha such that MAPE error is minimized,along with the MAPE index error in question, and its value"""
    X_new = np.zeros(X.shape)
    alphas = [round(0.05*i, 2) for i in range(20)]
    mapes = []
    pct_80 = int(np.ceil(80*len(X)/100))

    if model=='simple':
        for alpha in alphas:
            for j in range(X.shape[1]):
                X_new[:,j]= simple_exponential_smoothing(X[:,j], alpha)
            
            #X_train, X_test = X_new[:pct_80], X_new[pct_80:]
            #y_train, y_test =y_rec_smoothed[:pct_80], y_rec_smoothed[pct_80:]


            #index = find_best_index(X_train, y_train, X_test, y_test, 'mape', N)
            #P, q, G, h = generate_params(X_train, y_train, index,N)
            #gamma = cvxopt_solve_qp(P, q, G, h)
            #y_pred = X_test@gamma
            
            mapes.append(advancement_val(X_new, y)[1])

    else:
        for alpha in alphas:
            for j in range(X.shape[1]):
                X_new[:,j]= exponential_smoothing(X[:,j], alpha,K)


            #X_train, X_test = X_new[:pct_80], X_new[pct_80:]
            #y_train, y_test =y_rec_smoothed[:pct_80], y_rec_smoothed[pct_80:]


            #index = find_best_index(X_train, y_train, X_test, y_test, 'mape', N)
            #P, q, G, h = generate_params(X_train, y_train, index,N)
            #gamma = cvxopt_solve_qp(P, q, G, h)
            #y_pred = X_test@gamma
            
            #mapes.append(mape(y_test, y_pred))
            mapes.append(advancement_val(X_new, y)[1])
            
    return alphas[np.argmin(mapes)], min(mapes)


def advancement_val(X, y):    
    # We want our train set to be of size 40, and then we shift of 10 data points at each new iteration.
    # the size of our test set is the rest of the dataset points
    splits = int(np.floor((X.shape[0] - 40)/10))
    
    N = X.shape[1]

    mapes = []
    y_vals = []
    y_preds = []

    for i in range(splits):

        begin = 10*i
        end = 40 + 10*i

        X_tr = X[begin:end,:]
        y_tr = y[begin:end]

        X_te = X[end:,:]
        y_te = y[end:]

        # Run the model

        index = find_best_index(X_tr, y_tr, X_te, y_te, 'mape', N)
        P, q, G, h = generate_params(X_tr, y_tr, index, N, 10e-5)
        gamma = cvxopt_solve_qp(P, q, G, h)
        y_pred = X_te@gamma

        y_vals.append(y_te)
        y_preds.append(y_pred)

        mapes.append(mape(y_te, y_pred))

    y_vals = [item for sublist in y_vals for item in sublist]
    y_preds =[item for sublist in y_preds for item in sublist]

    return mapes, np.mean(mapes)

def apply_smoothing(X, K, parity):

    new_X = np.zeros(X.shape)

    for j in range(X.shape[1]):
        new_X[:,j] = gauss_filter(X[:,j], K, parity=parity)
        
    
    return new_X


# ----------------------------#
#        LOSS FUNCTIONS 
# ----------------------------#

def mape(y_test, y_pred):
    return np.mean(np.abs((y_pred-y_test)/y_test))

def mspe(y_test, y_pred):
    return np.mean(np.square((y_pred-y_test)/y_test))

def maape(y_test, y_pred):
    return np.mean(np.arctan(np.abs((y_pred-y_test)/y_test)))



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

def generate_G(index,N):
    """index: represents k^*, gamma_{k^*} is such that gamma_0 <= gamma_1 <= ...<= gamma_{k^*} >= ... >= gamma_N
    This function generates a matrix G such that either gamma_index or gamma_{index+1} is the maximum
    """ 
    #this constraint verifies the gaussian-like distribution of the gamma
    G = np.zeros((N,N))
    for i in range(0, index):
        for j in range(N):
            if (i==j):
                G[i,j] = 1
            elif (j == i+1):
                G[i,j] = -1
                
    for i in range(index, N):
        for j in range(N):
            if (i==j):
                G[i,j] = -1
            elif (j == i+1):
                G[i,j] = 1
    
    # we do not put any condition on idx_th element, and use this line to verify that all gammas are superior or
    # equal to zero
    #G[index,:] = 0
    #G[index, 0] = -1

    
    #this constraint verifies that -gamma_i <= 0 <=> gamma_i >= 0 forall i
   # for i in range(N, 2*N):
    #    for j in range(N):
     #       if (i==N+j):
      #          G[i,j]=-1
    return G

def generate_params(X_train, y_train,k,N,lambda_=10e-15):
    M = create_M(N)
    M_tilde = M.T @ M
    X_tilde = X_train.T @ X_train
    P = X_tilde + lambda_*(M_tilde)
    q = -X_train.T@y_train
    G = generate_G(k,N)
    h = np.zeros((N,1))
    for i in range(len(h)):
        h[i] = -0.0000001
    return P, q, G, h

def find_best_index(X_train, y_train, X_test, y_test, loss,N):
    """Returns index of maximum gamma that minimizes the mae loss"""
    loss = {}
    for k in range(N):
        P, q, G, h = generate_params(X_train, y_train, k,N)
        gammas = cvxopt_solve_qp(P,q, G, h)
        y_pred = X_test@gammas
        loss[k] = mape(y_test,y_pred)
    return min(loss, key=loss.get)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    cvxopt.solvers.options['show_progress'] = False
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    
    return np.array(sol['x']).reshape((P.shape[1],))



