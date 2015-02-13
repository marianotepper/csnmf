"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from dask.dot import dot_graph
import rcnmf.tsqr


def run():
    mat = np.random.rand(1000, 20)
    blockshape = (200, 20)
    data = da.from_array(mat, blockshape=blockshape, name='A')

    q, r = rcnmf.tsqr.tsqr(data)

    dot_graph(q.dask, filename='q')
    dot_graph(r.dask, filename='r')

    print q.shape
    q = np.array(q)

    r = np.array(r)
    print r.shape

    print np.linalg.norm(mat - np.dot(q, r))

    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(mat, interpolation='nearest')
    plt.title('Original matrix')
    plt.subplot(2, 4, 2)
    plt.imshow(q, interpolation='nearest')
    plt.title('$\mathbf{Q}$')
    plt.subplot(2, 4, 3)
    plt.imshow(np.dot(q.T, q), interpolation='nearest')
    plt.title('$\mathbf{Q}^T \mathbf{Q}$')
    plt.subplot(2, 4, 4)
    plt.imshow(r, interpolation='nearest')
    plt.title('$\mathbf{R}$')

    plt.subplot(2, 4, 8)
    plt.spy(r)
    plt.title('Nonzeros in $\mathbf{R}$')

    plt.show()

if __name__ == '__main__':
    run()
