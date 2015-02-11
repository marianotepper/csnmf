import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from dask.dot import dot_graph
import rcnmf.tsqr


def run():
    mat = np.random.rand(100, 20)
    blockshape = (100, 20)
    data = da.from_array(mat, blockshape=blockshape, name='A')

    q, r = rcnmf.tsqr.tsqr(data, blockshape=blockshape)

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
    plt.subplot(2, 4, 2)
    plt.imshow(q, interpolation='nearest')
    plt.subplot(2, 4, 3)
    plt.imshow(np.dot(q.T, q), interpolation='nearest')
    plt.subplot(2, 4, 4)
    plt.imshow(r, interpolation='nearest')

    plt.subplot(2, 4, 5)
    plt.spy(mat)
    plt.subplot(2, 4, 6)
    plt.spy(q)
    plt.subplot(2, 4, 8)
    plt.spy(r)

    plt.show()

if __name__ == '__main__':
    run()