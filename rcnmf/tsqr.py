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
import operator


def findnumblocks(shape, blockshape):

    def div_ceil(t):
        return int(ceil(float(t[0]) / t[1]))
    nb = [div_ceil(t) for t in zip(*[shape, blockshape])]
    return tuple(nb)


def create_keys(name, numblocks):
    block_list = map(tuple, map(range, numblocks))
    keys = [[name]]
    keys.extend(block_list)
    return list(product(*keys))


def tsqr(data, blockshape=None):

    m, n = mat.shape
    assert(n == blockshape[1])

    numblocks = findnumblocks(mat.shape, blockshape)

    dsk_qr_st1 = da.core.top(np.linalg.qr, 'QR_st1', 'ij', 'A', 'ij',
                             numblocks={'A': numblocks})
    # qr[0]
    dsk_q_st1 = {('Q_st1', i, 0): (operator.getitem, ('QR_st1', i, 0), 0)
                 for i in xrange(numblocks[0])}
    # qr[1]
    dsk_r_st1 = {('R_st1', i, 0): (operator.getitem, ('QR_st1', i, 0), 1)
                 for i in xrange(numblocks[0])}

    def _vstack(*args):
        tup = tuple(args)
        # print [t.shape for t in tup]
        return np.vstack(tup)

    to_append = [_vstack] + [('R_st1', i, 0) for i in xrange(numblocks[0])]
    dsk_r_st1_stacked = {('R_st1_stacked', 0, 0): tuple(to_append)}

    dsk_qr_st2 = da.core.top(np.linalg.qr, 'QR_st2', 'ij', 'R_st1_stacked', 'ij',
                             numblocks={'R_st1_stacked': (1, 1)})
    # qr[0]
    dsk_q_st2_aux = {('Q_st2_aux', 0, 0): (operator.getitem, ('QR_st2', 0, 0), 0)}
    dsk_q_st2 = dict((('Q_st2',) + ijk,
                      (operator.getitem, ('Q_st2_aux', 0, 0),
                       tuple(slice(i*d, (i+1)*d) for i, d in zip(ijk, (n, n)))))
                     for ijk in product(*map(range, numblocks)))
    # qr[1]
    dsk_r_st2 = {('R', i, 0): (operator.getitem, ('QR_st2', i, 0), 1)
                 for i in xrange(numblocks[0])}

    dsk_q_st3 = da.core.top(np.dot, 'Q', 'ij', 'Q_st1', 'ij', 'Q_st2', 'ij',
                            numblocks={'Q_st1': numblocks, 'Q_st2': numblocks})

    dsk_q = {}
    dsk_q.update(data.dask)
    dsk_q.update(dsk_qr_st1)
    dsk_q.update(dsk_q_st1)
    dsk_q.update(dsk_r_st1)
    dsk_q.update(dsk_r_st1_stacked)
    dsk_q.update(dsk_qr_st2)
    dsk_q.update(dsk_q_st2_aux)
    dsk_q.update(dsk_q_st2)
    dsk_q.update(dsk_q_st3)
    dsk_r = {}
    dsk_r.update(data.dask)
    dsk_r.update(dsk_qr_st1)
    dsk_r.update(dsk_r_st1)
    dsk_r.update(dsk_r_st1_stacked)
    dsk_r.update(dsk_qr_st2)
    dsk_r.update(dsk_r_st2)

    # print dsk_q
    # print dsk_r

    Q = da.Array(dsk_q, 'Q', shape=mat.shape, blockshape=blockshape)
    R = da.Array(dsk_r, 'R', shape=(n, n), blockshape=(n, n))

    return Q, R

if __name__ == '__main__':

    mat = np.random.rand(100, 20)
    blockshape=(100, 20)
    data = da.from_array(mat, blockshape=blockshape, name='A')
    print data.dask
    Q, R = tsqr(data, blockshape=blockshape)

    print Q.shape
    Q = np.array(Q)

    R = np.array(R)
    print R.shape

    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(mat, interpolation='nearest')
    plt.subplot(2, 4, 2)
    plt.imshow(Q, interpolation='nearest')
    plt.subplot(2, 4, 3)
    plt.imshow(np.dot(Q.T, Q), interpolation='nearest')
    plt.subplot(2, 4, 4)
    plt.imshow(R, interpolation='nearest')

    plt.subplot(2, 4, 5)
    plt.spy(mat)
    plt.subplot(2, 4, 6)
    plt.spy(Q)
    plt.subplot(2, 4, 8)
    plt.spy(R)

    plt.show()
