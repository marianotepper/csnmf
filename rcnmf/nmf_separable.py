"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import

import numpy as np
from rcnmf.third_party import mrnmf
import compression


def compute_compressed(data, q, alg, n_power_iter=0):

    data_comp = compression.compress(data, q, n_power_iter)
    colnorms = np.sum(np.fabs(data), axis=0)

    return mrnmf.nmf(data_comp, colnorms, alg, q)


def compute(data, q, alg, n_power_iter=0):

    data_comp = compression.compress(data, data.shape[1], n_power_iter)
    data_comp = np.linalg.qr(data_comp)[1]
    # TODO will have to re-implement this out-of-core
    colnorms = np.sum(np.fabs(data), axis=0)

    return mrnmf.nmf(data_comp, colnorms, alg, q)


if __name__ == "__main__":

    import time
    np.random.seed(10)

    m = 10000
    n = 500
    q = 5

    x = np.fabs(np.random.standard_normal(size=(m, q)))
    y = np.fabs(np.random.standard_normal(size=(q, n)))
    data = x.dot(y)

    t = time.clock()
    cols, H, rel_res = compute_compressed(data, q, 'xray')
    print cols, rel_res, time.clock() - t

    t = time.clock()
    cols, H, rel_res = compute(data, q, 'xray')
    print cols, rel_res, time.clock() - t