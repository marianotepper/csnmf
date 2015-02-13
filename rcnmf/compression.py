"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
from into import into
import dask.array as da
import tsqr


def compression_level(q):
    return max(20, q + 10)


def _multiply_power_iterations(data, omega, n_power_iter=0):
    mat_h = data.dot(omega)
    for j in range(n_power_iter):
        mat_h = data.dot(data.T.dot(mat_h))
    return mat_h


def compress_in_memory(data, q, n_power_iter=0):

    n = data.shape[1]
    comp_level = compression_level(q)
    omega = np.random.standard_normal(size=(n, comp_level))

    mat_h = _multiply_power_iterations(data, omega, n_power_iter=n_power_iter)

    comp = np.linalg.qr(mat_h, 'reduced')[0]
    comp = comp.T

    return comp.dot(data), comp


def compress_in_disk(uri, q, n_power_iter=0, blockshape=None):

    data = into(da.Array, uri, blockshape=blockshape)

    n = data.shape[1]
    comp_level = compression_level(q)
    omega = da.random.standard_normal(size=(n, comp_level),
                                      blockshape=(blockshape[1], comp_level))

    mat_h = _multiply_power_iterations(data, omega, n_power_iter=n_power_iter)

    q, r = tsqr.tsqr(mat_h, blockshape=(blockshape[0], mat_h.shape[1]))
    comp = q.T

    return comp.dot(data), comp


def compress(data, q, n_power_iter=0, blockshape=None):

    if isinstance(data, np.ndarray):
        comp_data, comp = compress_in_memory(data, q,
                                             n_power_iter=n_power_iter)
    elif isinstance(data, basestring):
        comp_data, comp = compress_in_disk(data, q, n_power_iter=n_power_iter,
                                           blockshape=blockshape)
    else:
        raise TypeError('Cannot compress data of type ' + type(data).__name__)

    return comp_data, comp
