"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
import numpy as np
import dask.array as da
from csnmf.third_party import mrnmf
import csnmf.compression
import csnmf.tsqr


def _compute_colnorms(data):
    """
    Compute the L1 norm for each column
    :param data: numpy.ndarray
        Input matrix
    :return: numpy.ndarray
        L1 norm for each column
    """
    if isinstance(data, np.ndarray):
        colnorms = np.norm(data, ord=1, axis=0)
    elif isinstance(data, da.Array):
        colnorms = data.vnorm(ord=1, axis=0)
    else:
        raise TypeError('Cannot compute columns norms of type ' +
                        type(data).__name__)
    return colnorms


def compute(data, ncols, alg, compress=False, n_power_iter=0):
    """
    Compute separable NMF of the input matrix.
    :param data:  Input matrix
    :type data: numpy.ndarray
    :param ncols:  Number of columns to select
    :type ncols: int
    :param alg: Choice of algorithm for computing the columns.
    One of 'SPA' or 'XRAY'.
    :type alg: basestring
    :param compress: Whether to use compression or not
    :type compress: bool
    :param n_power_iter: Number of power iterations for compression
    :type n_power_iter: int
    :return:
    The selected columns, the right factor of the separable NMF
    decomposition, and the relative error.
    :rtype: tuple of:
     - list of ints
     - right factor matrix
     - relative error of the approximation
    """
    if compress:
        data_comp, _ = csnmf.compression.compress(data, ncols, n_power_iter)
    else:
        if isinstance(data, da.Array):
            _, data_comp = csnmf.tsqr.qr(data)
        elif isinstance(data, np.ndarray):
            _, data_comp = np.linalg.qr(data)
        else:
            raise TypeError('Cannot compute QR decomposition of matrices '
                            'of type ' + type(data).__name__)
    data_comp = np.array(data_comp)

    if alg == 'SPA':
        colnorms = _compute_colnorms(data_comp)
        colnorms = np.array(colnorms)
    else:
        colnorms = None

    cols = mrnmf.select_columns(data_comp, alg, ncols, colnorms=colnorms)
    mat_h, error = mrnmf.nnls_frob(data_comp, cols)

    return cols, mat_h, error


def compute_multiple(data, ncols, alg, compress=False, n_power_iter=0, step=1):
    """
    Compute separable NMF of the input matrix with k columns, for
    k = 1 to ncols.
    :param data:  Input matrix
    :type data: numpy.ndarray
    :param ncols:  Maximum number of columns to select
    :type ncols: int
    :param alg: Choice of algorithm for computing the columns.
    One of 'SPA' or 'XRAY'.
    :type alg: basestring
    :param compress: Whether to use compression or not
    :type compress: bool
    :param n_power_iter: Number of power iterations for compression
    :type n_power_iter: int
    :param step: Step size for k
    :type step: int
    :return:
    The selected columns, the right factors of the separable NMF
    decomposition, and the relative errors.
    :rtype: list of tuples of:
     - list of ints
     - right factor matrix
     - relative error of the approximation
    """
    if compress:
        data_comp, _ = csnmf.compression.compress(data, ncols, n_power_iter)
    else:
        if isinstance(data, da.Array):
            _, data_comp = csnmf.tsqr.qr(data)
        elif isinstance(data, np.ndarray):
            _, data_comp = np.linalg.qr(data)
        else:
            raise TypeError('Cannot compute QR decomposition of matrices '
                            'of type ' + type(data).__name__)

    colnorms = _compute_colnorms(data_comp)

    try:
        data_comp = np.array(data_comp)
        colnorms = np.array(colnorms)
    except:
        raise

    results = []
    for k in range(step, ncols, step):
        cols = mrnmf.select_columns(data_comp, colnorms, alg, k)
        mat_h, error = mrnmf.nnls_frob(data_comp, cols)
        results.append((cols, mat_h, error))

    return results