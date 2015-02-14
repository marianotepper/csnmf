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
from rcnmf.third_party import mrnmf
import rcnmf.compression
import rcnmf.tsqr
import timeit


def _compute_colnorms(data):

    if isinstance(data, np.ndarray):
        colnorms = np.sum(np.fabs(data), axis=0)
    elif isinstance(data, da.Array):
        colnorms = data.vnorm(ord=1, axis=0)
    else:
        raise TypeError('Cannot compute columns norms of type ' +
                        type(data).__name__)
    return colnorms


def compute(data, ncols, alg, compress=False, n_power_iter=0):
    # t = timeit.default_timer()
    if compress:
        data_comp, _ = rcnmf.compression.compress(data, ncols, n_power_iter)
    else:
        if isinstance(data, da.Array):
            _, data_comp = rcnmf.tsqr.qr(data)
        elif isinstance(data, np.ndarray):
            _, data_comp = np.linalg.qr(data)
        else:
            raise TypeError('Cannot compute QR decomposition of matrices '
                            'of type ' + type(data).__name__)
    # time_phase1 = timeit.default_timer() - t

    colnorms = _compute_colnorms(data)

    if isinstance(data, da.Array):
        data_comp = np.array(data_comp)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError('Cannot convert matrices of type ' +
                        type(data).__name__)

    # t = timeit.default_timer()
    res = mrnmf.nmf(data_comp, colnorms, alg, ncols)
    # time_phase2 = timeit.default_timer() - t

    # base_str = 'Time for phase 1: {0:.2f}. Time for phase 2: {1:.2f}'
    # print(base_str.format(time_phase1, time_phase2))
    return res#, time_phase1, time_phase2
