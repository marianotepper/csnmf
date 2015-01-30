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


def compress(data, n_power_iter=0):

    n = data.shape[1]

    omega = np.random.standard_normal(size=(n, n))
    mat_h = data.dot(omega)
    for j in range(n_power_iter):
        mat_h = data.dot(data.T.dot(mat_h))
    comp = np.linalg.qr(mat_h, 'reduced')[0]
    comp = comp.T

    data_comp = comp.dot(data)

    return np.linalg.qr(data_comp, 'reduced')[1]


def compute(data, q, alg, n_power_iter=0):

    left_factor = compress(data, n_power_iter)
    colnorms = np.sum(np.fabs(data), axis=0)

    return mrnmf.nmf(left_factor, colnorms, alg, q)


if __name__ == "__main__":

    m = 100
    n = 50
    q = 5

    x = np.fabs(np.random.standard_normal(size=(m, q)))
    y = np.fabs(np.random.standard_normal(size=(q, n)))
    data = x.dot(y)

    cols, H, rel_res = compute(data, q, 'SPA')
    print rel_res