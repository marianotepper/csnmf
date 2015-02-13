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


def _inner_compress(data, omega, n_power_iter=0, qr=np.linalg.qr):

    mat_h = data.dot(omega)
    for j in range(n_power_iter):
        mat_h = data.dot(data.T.dot(mat_h))
    q, _ = qr(mat_h)
    comp = q.T
    return comp.dot(data), comp


def compress(data, q, n_power_iter=0, blockshape=None):

    n = data.shape[1]
    comp_level = compression_level(q)

    if isinstance(data, np.ndarray):
        omega = np.random.standard_normal(size=(n, comp_level))
        qr = np.linalg.qr
    elif isinstance(data, da.Array):
        omega = da.random.standard_normal(size=(n, comp_level),
                                          blockshape=(blockshape[1],
                                                      comp_level))
        qr = tsqr.tsqr
    else:
        raise TypeError('Cannot compress data of type ' + type(data).__name__)

    return _inner_compress(data, omega, n_power_iter=n_power_iter, qr=qr)
