"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

# import math
# import numpy as np


# def compute(mat, q, n_iter=1e5, n_power_iter=4):
#
#     n_iter = int(n_iter)
#
#     m = mat.shape[0]
#     n = mat.shape[1]
#
#     l = max(20, q + 10)
#
#     omega_left = np.random.randn(n, l)
#     mat_h = mat.dot(omega_left)
#     for j in range(n_power_iter):
#         mat_h = mat.dot(mat.T.dot( mat_h))
#     left_comp = np.linalg.qr(mat_h, 'reduced')[0]
#     left_comp = left_comp.T
#
#     omega_right = np.random.randn(l, m)
#     mat_h = omega_right.dot(mat)
#     for j in range(n_power_iter):
#         mat_h = (mat_h.dot(mat.T)).dot(mat)
#     right_comp = np.linalg.qr(mat_h.T, 'reduced')[0]
#
#     mat_lr = left_comp.dot(mat.dot(right_comp))
#
#     mat_l = left_comp.dot(mat)
#     mat_r = mat.dot(right_comp)
#
#     u = np.fabs(np.random.randn(m, q))
#     v = np.fabs(np.random.randn(q, n))
#
#     u_comp = left_comp.dot(u)
#
#     err = np.zeros((n_iter, 1))
#     for i in xrange(n_iter):
#
#         temp1 = u_comp.T.dot(mat_l)
#         temp2 = u_comp.T.dot(u_comp)
#         num = projection_pos(temp1) + projection_neg(temp2).dot(v)
#         denominator = projection_neg(temp1) + projection_pos(temp2).dot(v)
#         v = np.sqrt(num / denominator) * v
#
#         v_comp = v.dot(right_comp)
#
#         temp1 = mat_r.dot(v_comp.T)
#         temp2 = v_comp.dot(v_comp.T)
#         num = projection_pos(temp1) + u.dot(projection_neg(temp2))
#         denominator = projection_neg(temp1) + u.dot(projection_pos(temp2))
#         u = u * np.sqrt(num / denominator)
#
#         u_comp = left_comp.dot(u)
#
#         err[i] = math.log10(np.linalg.norm(mat_lr - u_comp.dot(v_comp), 'fro'))
#
#         if i >= 20 and (err[i] < -10 or math.fabs(err[i] - err[i-1]) <= 1e-10):
#             err.reshape((i+1, 1))
#             break
#
#     return u, v, err
#
#
# def projection_neg(mat):
#     return (np.fabs(mat) - mat) / 2
#
#
# def projection_pos(mat):
#     return (np.fabs(mat) + mat) / 2


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import h5py
    from dask.array import Array
    from blaze import Data, into
    import random_dask
    import os

    m = 1000
    n = 1000
    q = 100

    X = random_dask.standard_normal(size=(m, q), blockshape=(200, 100))
    Y = random_dask.standard_normal(size=(q, m), blockshape=(100, 200))

    filename = 'nmffile.hdf5'
    if os.path.isfile(filename):
        os.remove(filename)

    f = h5py.File(filename)
    f.close()

    into('nmffile.hdf5::/X', X)
    into('nmffile.hdf5::/Y', Y)

    f = h5py.File(filename)
    # print(f)
    # print(f['X'])
    # print(f['Y'])
    # print(f['Y'][:5, :5])

    # print f.get('X')

    print f['X'].shape

    Y2 = np.array(f['X'][:5, :5])
    print(Y2)

    X3 = into(np.ndarray, X)
    print X3

    # block_m = q#min(f['X'].shape[0], 1000)
    # block_n = q#min(f['X'].shape[1], 1000)
    # X_into = into(Array, 'nmffile.hdf5::/X', blockshape=(block_m, block_n))
    # X_data = Data(X_into)
    #
    # block_m = q#min(f['Y'].shape[0], 1000)
    # block_n = q#min(f['Y'].shape[1], 1000)
    # Y_into = into(Array, 'nmffile.hdf5::/Y', blockshape=(block_m, block_n))
    # Y_data = Data(Y_into)
    #
    # print X_data.shape, Y_data.shape
    # into('nmffile.hdf5::/M', X_data.dot(Y_data))

    # start = time.clock()
    # u, v, err = compute(M, q, 1e3)
    # end = time.clock()
    # print(end-start)
    #
    # plt.figure()
    # plt.plot(err)
    #
    # plt.show()

    f.close()
