"""
   Copyright (c) 2014, Austin R. Benson, David F. Gleich,
   Purdue University, and Stanford University.
   All rights reserved.

   This file is part of MRNMF and is under the BSD 2-Clause License,
   which can be found at http://opensource.org/licenses/BSD-2-Clause

   Copyright (c) 2015, Mariano Tepper,
   Duke University.
   All rights reserved.

    Mariano Tepper made the following changes to this file:
    - modified names and line lengths to adhere more closely to PEP8
    - changed docstrings
    - some numpy operations are more numpy-ish now.
    - small edits, refactoring, and cleanups
    - removed some code
"""

import numpy as np
from scipy.optimize import nnls


def spa(data, r, colnorms):
    """
    Successive projection algorithm (SPA) for NMF.  This algorithm
    computes the column indices.
    :param data: The data matrix.
    :type data: numpy.ndarray
    :param r: The target separation rank.
    :type r: int
    :param colnorms: The column L1 norms.
    :type colnorms: numpy.ndarray
    :return: A list of r columns chosen by SPA.
    :rtype: list of int
    """
    idx = np.nonzero(colnorms)
    x = np.copy(data)
    x[:, idx] /= colnorms[idx]
    cols = []
    m, n = x.shape
    for _ in xrange(r):
        col_norms = np.linalg.norm(x, ord=2, axis=0)
        col_norms[cols] = -1
        col_ind = np.argmax(col_norms)
        cols.append(col_ind)
        col = np.atleast_2d(x[:, col_ind])  # col is a row vector
        x = np.dot(np.eye(m) - np.dot(col.T, col) / col_norms[col_ind], x)

    return cols


def xray(x, r):
    """
    X-ray algorithm for NMF.  This algorithm computes the column
    indices.
    :param x: The data matrix.
    :type x: numpy.ndarray
    :param r: The target separation rank.
    :type r: int
    :return: A list of r columns chosen by X-ray.
    :rtype: list of int
    """
    cols = []
    R = np.copy(x)
    while len(cols) < r:
        # Loop until we choose a column that has not been selected.
        while True:
            p = np.random.random((1, x.shape[0]))
            scores = np.linalg.norm(np.dot(R.T, x), ord=2, axis=0)
            scores /= np.squeeze(np.dot(p, x))
            scores[cols] = -1   # IMPORTANT
            best_col = np.argmax(scores)
            if best_col in cols:
                # Re-try
                continue
            else:
                cols.append(best_col)
                H, rel_res = nnls_frob(x, cols)
                R = x - np.dot(x[:, cols], H)
                break
    return cols


def nnls_frob(x, cols):
    """
    Compute H, the coefficient matrix, by nonnegative least squares
    to minimize the Frobenius norm.  Given the data matrix X and the
    columns cols, H is
    .. math:: \arg\min_{Y \ge 0} \| X - X(:, cols) H \|_F.
    :param X: The data matrix.
    :type X: numpy.ndarray
    :param cols: The column indices.
    :type cols: list of int
    :return: The matrix H and the relative residual.
    """

    ncols = x.shape[1]
    x_sel = x[:, cols]
    H = np.zeros((len(cols), ncols))
    for i in xrange(ncols):
        sol, res = nnls(x_sel, x[:, i])
        H[:, i] = sol
    rel_res = np.linalg.norm(x - np.dot(x_sel, H), 'fro')
    rel_res /= np.linalg.norm(x, 'fro')
    return H, rel_res


def select_columns(data, alg, r, colnorms=None):
    """ Compute an approximate separable NMF of the matrix data.  By
    compute, we mean choose r columns and a best fitting coefficient
    matrix H.  The r columns are selected by the 'alg' option, which
    is one of 'SPA' or 'XRAY'.  The coefficient matrix H is the
    one that produces the smallest Frobenius norm error.

    :param data: The data matrix.
    :type data: numpy.ndarray
    :param alg: Choice of algorithm for computing the columns.  One of
    'SPA' or 'XRAY'.
    :type alg: string
    :param r: The target separation rank.
    :type r: int
    :param colnorms: The column L1 norms, needed only by SPA.
    :type colnorms: numpy.ndarray
    :return The selected columns, the matrix H, and the relative residual.
    """

    if alg == 'XRAY':
        cols = xray(data, r)
    elif alg == 'SPA':
        cols = spa(data, r, colnorms)
    else:
        raise Exception('Unknown algorithm: {0}'.format(alg))

    return cols
