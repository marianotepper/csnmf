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
import time


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

    return comp.dot(data)


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
        temp = blaze.Data(blaze.into(Array, data.T.dot(mat_h)))
        mat_h = blaze.Data(blaze.into(Array, data.dot(temp)))

    mat_h = blaze.into(np.ndarray, mat_h)

    comp = np.linalg.qr(mat_h, 'reduced')[0]
    comp = comp.T

    return comp.dot(data)


def compress(data, q, n_power_iter=0, blockshape=None):

    if isinstance(data, np.ndarray):
        compress_in_memory(data, q, n_power_iter=0)
    elif isinstance(data, basestring):
        compress_in_disk(data, q, n_power_iter=0, blockshape=blockshape)
    else:
        raise TypeError('Cannot compress data of type ' + type(data).__name__)

if __name__ == '__main__':

    import h5py
    import os

    X_disk = random.standard_normal(size=(10000, 5000), blockshape=(1000, 1000))


    filename = 'data.hdf5'
    if os.path.isfile(filename):
        os.remove(filename)
    f = h5py.File(filename)
    f.close()

    blaze.into(filename + '::/X', X_disk)

    data_array = blaze.into(Array, filename + '::/X', blockshape=(1000, 1000))
    X = blaze.into(np.ndarray, blaze.Data(data_array))

    t = time.clock()
    compress(filename + '::/X', 5, blockshape=(1000, 1000))
    print time.clock() - t
    t = time.clock()
    compress(X, 5)
    print time.clock() - t
    compress(1, 5)