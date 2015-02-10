"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
from itertools import product
import dask
import dask.array as da
import into
from math import ceil
import matplotlib.pyplot as plt
import tempfile

def findnumblocks(shape, blockshape):

    def div_ceil(t):
        return int(ceil(float(t[0]) / t[1]))
    nb = [div_ceil(t) for t in zip(*[shape, blockshape])]
    return tuple(nb)


def tuple_to_outputs(out0, out1, t):
    # into.into(out0, t[0], inplace=True)
    # into.into(out1, t[1], inplace=True)
    np.copyto(out0, t[0])
    np.copyto(out1, t[1])


def create_keys(name, numblocks):
    block_list = map(tuple, map(range, numblocks))
    keys = [[name]]
    keys.extend(block_list)
    return list(product(*keys))


def tsqr(mat, blockshape=None):

    assert(mat.shape[1] == blockshape[1])

    data = da.from_array(mat, blockshape=blockshape, name='A')

    numblocks = findnumblocks(mat.shape, blockshape)
    dsk_qr = da.core.top(np.linalg.qr, 'QR', 'ij', 'A', 'ij',
                       numblocks={'A': numblocks})

    kn = mat.shape[1] * numblocks[0]
    Q_arr = np.zeros(shape=(mat.shape[0], kn))
    Q = da.from_array(Q_arr, blockshape=blockshape, name='Q')
    R_arr = np.zeros(shape=(kn, mat.shape[1]))
    R = da.from_array(R_arr, blockshape=(blockshape[1], blockshape[1]), name='R')

    numblocks_dict = {'QR': numblocks, 'R': numblocks,
                      'Q': (numblocks[0], numblocks[0])}
    dsk = da.core.top(tuple_to_outputs, 'RES', 'ij', 'Q', 'ii', 'R', 'ij',
                        'QR', 'ij', numblocks=numblocks_dict)

    dsk.update(data.dask)
    dsk.update(dsk_qr)
    dsk.update(Q.dask)
    dsk.update(R.dask)

    keys = create_keys('RES', numblocks)
    dask.get(dsk, keys)

    Q_st2, R_st2 = np.linalg.qr(R_arr)

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(Q_arr, interpolation='nearest')
    plt.subplot(2, 3, 4)
    plt.imshow(R_arr, interpolation='nearest')
    plt.subplot(2, 3, 5)
    plt.imshow(Q_st2, interpolation='nearest')
    plt.subplot(2, 3, 6)
    plt.imshow(R_st2, interpolation='nearest')

    plt.show()


if __name__ == '__main__':

    mat = np.random.randn(1000, 20)
    tsqr(mat, blockshape=(250, 20))