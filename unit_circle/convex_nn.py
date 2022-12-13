import numpy as np
import scipy.special
import cvxpy as cp

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return x >= 0

def P(n, d):
    return 2 * int(np.round(scipy.special.comb(n-1, np.arange(d)).sum()))

def train(X, y, m=50, beta=1e-4, n_rand=int(1e5), oversample=True):

    n, d = X.shape

    print("P theoretical:", P(n, d))

    # randomly generated vectors to sample activation patterns

    u = np.random.randn(d,int(n_rand))
    dmat=drelu(X@u)

    dmat, counts = np.unique(dmat, axis=1, return_counts=True)

    if oversample:
        idx = np.argpartition(counts, -m//2)[-m//2:]
        dmat = dmat[:, idx]
    else:
        dmat = dmat[:, :m//2]

    print("P actual: ", dmat.shape[1])

    # build and solve the convex program
    # https://github.com/pilancilab/convex_nn/blob/main/convex_nn.py

    m1 = dmat.shape[1]
    u = cp.Variable((d,m1))
    w = cp.Variable((d,m1))

    yopt1 = cp.sum(cp.multiply(dmat, X @ u), axis=1)
    yopt2 = cp.sum(cp.multiply(dmat, X @ w), axis=1)
    cost = cp.sum(cp.pos(1 - cp.multiply(y, yopt1-yopt2))) / n + beta * (cp.mixed_norm(u.T,2,1) + cp.mixed_norm(w.T,2,1))
    constraints = [
        cp.multiply((2*dmat-1),(X@u))>=0,
        cp.multiply((2*dmat-1),(X@w))>=0,
    ]
    prob = cp.Problem(cp.Minimize(cost),constraints)
    prob.solve(verbose=True)

    print("optimal value: ", prob.value)

    # find the active neurons and generate weights for relu network
    u, w = u.value, w.value
    
    threshold = 1e-8
    u = u[:, np.linalg.norm(u, axis=0) > threshold]
    w = w[:, np.linalg.norm(w, axis=0) > threshold]

    u_norm_sq = np.sqrt(np.linalg.norm(u, axis=0, keepdims=True))
    w_norm_sq = np.sqrt(np.linalg.norm(w, axis=0, keepdims=True))

    w1 = np.hstack((u/u_norm_sq, w/w_norm_sq))
    w2 = np.hstack((u_norm_sq, -w_norm_sq)).T

    # number of active neurons
    print("m*: ", len(w2))

    return w1, w2