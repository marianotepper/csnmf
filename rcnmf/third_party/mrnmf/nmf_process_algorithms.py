"""
   Copyright (c) 2014, Austin R. Benson, David F. Gleich,
   Purdue University, and Stanford University.
   All rights reserved.

   This file is part of MRNMF and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause

   Copyright (c) 2015, Mariano Tepper,
   Duke University.
   All rights reserved.

    Mariano Tepper made the following changes to this file:
    - modify names and line lengths to adhere more closely to PEP8
    - small edits, refactoring, and cleanups
    - removed some code
"""

import numpy as np
from scipy import optimize


def col2norm(X):
    """ Compute all column 2-norms of a matrix. """
    return np.sum(X**2, axis=0)


def spa(X, r):
    """ Successive projection algorithm (SPA) for NMF.  This algorithm
    computes the column indices.
    Args:
        X: The data matrix.
        r: The target separation rank.
    Returns:
        A list of r columns chosen by SPA.
    """
    cols = []
    m, n = X.shape
    for _ in xrange(r):
        col_norms = col2norm(X)
        col_ind = np.argmax(col_norms)
        cols.append(col_ind)
        col = np.atleast_2d(X[:, col_ind]) #col is a row vector
        X = np.dot((np.eye(m) - np.dot(col.T, col) / col_norms[col_ind]), X)

    return cols


def xray(X, r):
    """ X-ray algorithm for NMF.  This algorithm computes the column
    indices.
    Args:
        X: The data matrix.
        r: The target separation rank.
    Returns:
        A list of r columns chosen by X-ray.
    """
    cols = []
    R = np.copy(X)
    while len(cols) < r:
        # Loop until we choose a column that has not been selected.
        while True:
            scores = col2norm(np.dot(R.T, X)) / col2norm(X)
            scores[cols] = -1   # IMPORTANT
            best_col = np.argmax(scores)
            if best_col in cols:
                # Re-try
                continue
            else:
                cols.append(best_col)
                H, rel_res = nnls_frob(X, cols)
                R = X - np.dot(X[:, cols] , H)
                break
    return cols


def gp_cols(data, r):
    """ X-ray algorithm for NMF.  This algorithm computes the column
    indices.
    Args:
        data: The matrix G * X, where X is the nonnegative data matrix
            and G is a matrix with Gaussian i.i.d. random entries.
        r: The target separation rank.
    Returns:
        A list of r columns chosen by Gaussian projection.
    """
    votes = {}
    for row in data:
        min_ind = np.argmin(row)
        max_ind = np.argmax(row)
        for ind in [min_ind, max_ind]:
            if ind not in votes:
                votes[ind] = 1
            else:
                votes[ind] += 1

    votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in votes][0:r]


def nnls_frob(X, cols):
    """ Compute H, the coefficient matrix, by nonnegative least squares
    to minimize the Frobenius norm.  Given the data matrix X and the
    columns cols, H is
             \arg\min_{Y \ge 0} \| X - X(:, cols) H \|_F.
    Args:
        X: The data matrix.
        cols: The column indices.
    Returns:
        The matrix H and the relative resiual.
    """

    ncols = X.shape[1]
    H = np.zeros((len(cols), ncols))
    for i in xrange(ncols):
        sol, res = optimize.nnls(X[:, cols], X[:, i])
        H[:, i] = sol
    rel_res = np.linalg.norm(X - np.dot(X[:, cols], H), 'fro')
    rel_res /= np.linalg.norm(X, 'fro')
    return H, rel_res


def nmf(data, colnorms, alg, r):
    """ Compute an approximate separable NMF of the matrix data.  By
    compute, we mean choose r columns and a best fitting coefficient
    matrix H.  The r columns are selected by the 'alg' option, which
    is one of 'SPA', 'xray', or 'GP'.  The coefficient matrix H is the
    one that produces the smallest Frobenius norm error.  The
    coefficient matrix H and residual only make sense when the
    algorithm is 'SPA' or 'xray'.  However, given the columns selected
    by 'GP', you can call NNLSFrob with the QR data to get H and the
    relative residual.

    Args:
        data: The data matrix.
        colnorms: The column norms.
        alg: Choice of algorithm for computing the columns.  One of 'SPA',
            'xray', or 'GP'.
        r: The target separation rank.
    Returns:
        The selected columns, the matrix H, and the relative residual.
    """

    if alg == 'xray':
        cols = xray(data, r)
    else:
        colinv = np.linalg.pinv(np.diag(colnorms))
        A = np.dot(data, colinv)

        if alg == 'SPA':
            cols = spa(A, r)
        elif alg == 'GP':
            cols = gp_cols(data, r)
        else:
            raise Exception('Unknown algorithm: %s' % str(alg))

    print cols
    H, rel_res = nnls_frob(data, cols)
    return cols, H, rel_res
