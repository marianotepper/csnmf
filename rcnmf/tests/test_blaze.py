"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
from dask.array import Array, random
import blaze
import os

m = 1000
n = 1000
q = 100

X = random.standard_normal(size=(m, q), blockshape=(200, 100))
Y = random.standard_normal(size=(q, m), blockshape=(100, 200))

filename = 'nmffile.hdf5'
if os.path.isfile(filename):
    os.remove(filename)

f = h5py.File(filename)
f.close()

blaze.into(filename + '::/X', X)
blaze.into(filename + '::/Y', Y)

f = h5py.File(filename, 'a')
# print(f)
# print(f['X'])
# print(f['Y'])
# print(f['Y'][:5, :5])

# print f.get('X')

print f['X'].shape

X2 = np.array(f['X'])
print(X2.shape)
print(X2[0, 0])

X = blaze.into(Array, filename + '::/X', blockshape=(200, 100))
Y = blaze.into(Array, filename + '::/Y', blockshape=(100, 200))
X_data = blaze.Data(X)
Y_data = blaze.Data(Y)

X_data = blaze.abs(X_data)
blaze.into(filename + '::/X', X_data)

print(X_data[0, 0])


X2 = np.array(f['X'])
print(X2.shape)
print(X2[0, 0])

print X_data.shape, Y_data.shape
blaze.into(filename + '::/M', X_data.dot(Y_data))