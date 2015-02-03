"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
from dask.array import Array, random
import blaze
import tempfile


def compress_in_memory(data, q, n_power_iter=0):

    n = data.shape[1]
    l = max(20, q + 10)
    # l = n

    omega = np.random.standard_normal(size=(n, l))
    mat_h = data.dot(omega)
    for j in range(n_power_iter):
        mat_h = data.dot(data.T.dot(mat_h))

    comp = np.linalg.qr(mat_h, 'reduced')[0]
    comp = comp.T

    return comp.dot(data), comp


def compress_in_disk(uri, q, n_power_iter=0, blockshape=None):

    data_array = blaze.into(Array, uri, blockshape=blockshape)
    data = blaze.Data(data_array)

    n = data.shape[1]
    l = max(20, q + 10)

    np.random.seed(0)

    omega_array = random.standard_normal(size=(n, l),
                                         blockshape=(blockshape[1], l))
    omega = blaze.Data(omega_array)

    mat_h = blaze.Data(blaze.into(Array, data.dot(omega)))

    for j in range(n_power_iter):
        temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5')
        blaze.into(temp_file.name + '::/temp', data.T.dot(mat_h))
        temp = blaze.Data(blaze.into(Array, temp_file.name + '::/temp',
                          blockshape=(blockshape[1], l)))
        mat_h = blaze.Data(blaze.into(Array, data.dot(temp)))
        temp_file.close()

    mat_h = blaze.into(np.ndarray, mat_h)

    comp = np.linalg.qr(mat_h, 'reduced')[0]
    comp = comp.T

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
