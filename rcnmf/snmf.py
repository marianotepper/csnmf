"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import
import numpy as np
import dask.array as da
from rcnmf.third_party import mrnmf
import rcnmf.compression


def _compute_colnorms(data):

    if isinstance(data, np.ndarray):
        colnorms = np.sum(np.fabs(data), axis=0)
    # elif isinstance(data, da.Array):

    else:
        raise TypeError('Cannot compute columns norms of type ' +
                        type(data).__name__)
    return colnorms


def compute(data, ncols, alg, compress=False, n_power_iter=0):
    if compress:
        data_comp, _ = rcnmf.compression.compress(data, ncols, n_power_iter)
    else:
        _, data_comp = np.linalg.qr(data)

    colnorms = _compute_colnorms(data)

    return mrnmf.nmf(data_comp, colnorms, alg, ncols)
