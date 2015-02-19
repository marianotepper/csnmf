"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
import numpy as np
import dask.array as da
import timeit
import itertools
import matplotlib.pyplot as plt
import pickle
import rcnmf.snmf


def run(m, n, q, ncols, blockshape):
    """
    Create a low-rank matrix. We then compute its separable NMF
    decomposition, using the SPA and XRAY algorithms, in-core and
    out-of-core computations, and the QR and compression variants.
    :param m: number of rows of the input matrix
    :type m: int
    :param n: number of columns of the input matrix
    :type n: int
    :param q: rank of the input matrix
    :type q: int
    :param ncols: number of columns to use in the decomposition
    :type ncols: int
    :param blockshape: shape of the block to use for out-of-core
    computations.
    :type blockshape: tuple of int
    :return: a list of dictionaries where each dictionary contains the
    following keys:
    - 'alg': specifying the algorithms for finding the extreme columns
    - 'comp': boolean specifying if compression was done; otherwise the
    QR decomposition is used
    - 'data_type': basetring specifying if the computations were done
    in-core or out-of-core
    - 'cols': the delected columns
    - 'error': the relative error of the decomposition residual
    - 'time': execution time
    """

    x = np.fabs(np.random.standard_normal(size=(m, q)))
    y = np.fabs(np.random.standard_normal(size=(q, n)))
    mat = x.dot(y)

    res_list = []

    algorithms = ['SPA', 'XRAY']
    compress = [False, True]
    data = [mat, da.from_array(mat, blockshape=blockshape)]

    for tup in itertools.product(algorithms, compress, data):

        t = timeit.default_timer()
        cols, _, error = rcnmf.snmf.compute(tup[2], ncols, 'SPA',
                                            compress=tup[1])
        t = timeit.default_timer() - t

        if isinstance(tup[2], np.ndarray):
            dtype = 'in-core'
        elif isinstance(tup[2], da.Array):
            dtype = 'out-of-core'

        res_dict = {'alg': tup[0], 'comp': tup[1], 'data_type': dtype,
                    'cols': cols, 'error': error, 'time': t}

        base_str = 'algorithm: {alg:4s}; compressed: {comp:d}; ' \
                   'type: {data_type:11s}; error {error:.4f}; ' \
                   'time {time:.2f}'
        print(base_str.format(**res_dict))
        res_list.append(res_dict)

    return res_list


def test_rank(m, n, only_draw=False):
    """
    Test snmf as the matrix rank changes
    :param m: number of rows
    :param n: number of columns
    :param only_draw: do not run test, only read data from file
    """
    m = int(m)
    n = int(n)
    q_max = n
    blockshape = (max(m/10, int(1e4)), n)

    test_name = 'test_snmf_rank_{0:.0e}_{1:.0e}'.format(m, n)

    q_list = range(q_max/10, q_max+1, q_max/10)
    shape = (len(q_list), 1)

    if not only_draw:
        time_vecs = {}

        for i, q in enumerate(q_list):
            res_list = run(m, n, q, q, blockshape)
            for res in res_list:
                key = (res['alg'], res['data_type'])
                if key not in time_vecs:
                    time_vecs[key] = {}
                    time_vecs[key][True] = np.zeros(shape)
                    time_vecs[key][False] = np.zeros(shape)
                time_vecs[key][res['comp']][i] = res['time']

        with open(test_name, 'w') as f:
            pickle.dump(time_vecs, f)

    with open(test_name, 'r') as f:
        time_vecs = pickle.load(f)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    plt.figure(figsize=(10, 5))
    ax1 = plt.axes()
    ax1.hold(True)
    k = 0
    for (alg, dtype) in time_vecs.keys():
        for comp in time_vecs[(alg, dtype)]:
            if len(alg) < 4:
                label = '{0:4s}'.format(alg.upper()) + ' - '
            else:
                label = '{0:4s}'.format(alg.upper()) + ' - '
            if comp:
                label += '{0:5s} - '.format('comp.')
                linestyle = '-'
            else:
                label += '{0:5s} - '.format('QR')
                linestyle = '--'
            label += dtype

            ax1.plot(q_list, time_vecs[(alg, dtype)][comp], label=label,
                     linestyle=linestyle, linewidth=2, marker='o',
                     markeredgecolor='none', color=colors[k])
        k += 1

    ax1.hold(False)

    ax1.set_xticks(q_list)
    ax1.set_xticklabels(q_list)
    ax1.set_xlabel('Rank of the input matrix')
    ax1.set_ylabel('Time (s)')

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.55, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               prop={'family': 'monospace'})

    plt.savefig(test_name + '.pdf')


def test_ncols(m, func, only_draw=False):
    """
    Test snmf as the number of columns changes
    :param m: number of rows
    :param func: function determining the rank of the input matrix
        as a function of n
    :param only_draw: do not run test, only read data from file
    """
    m = int(m)
    n_max = int(1e3)

    test_name = 'test_snmf_ncols_{0:.0e}'.format(m)

    n_list = range(n_max/10, n_max+1, n_max/10)
    shape = (len(n_list), 1)

    if not only_draw:
        time_vecs = {}

        for i, n in enumerate(n_list):
            blockshape = (max(m/10, int(1e4)), n)
            q = func(n)
            res_list = run(m, n, q, q, blockshape)
            for res in res_list:
                key = (res['alg'], res['data_type'])
                if key not in time_vecs:
                    time_vecs[key] = {}
                    time_vecs[key][True] = np.zeros(shape)
                    time_vecs[key][False] = np.zeros(shape)
                time_vecs[key][res['comp']][i] = res['time']

        with open(test_name, 'w') as f:
            pickle.dump(time_vecs, f)

    with open(test_name, 'r') as f:
        time_vecs = pickle.load(f)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    plt.figure(figsize=(10, 5))
    ax1 = plt.axes()
    ax1.hold(True)
    k = 0
    for (alg, dtype) in time_vecs.keys():
        for comp in time_vecs[(alg, dtype)]:
            if len(alg) < 4:
                label = '{0:4s}'.format(alg.upper()) + ' - '
            else:
                label = '{0}'.format(alg.upper()) + ' - '
            if comp:
                label += '{0:5s} - '.format('comp.')
                linestyle = '-'
            else:
                label += '{0:5s} - '.format('QR')
                linestyle = '--'
            label += dtype

            ax1.plot(n_list, time_vecs[(alg, dtype)][comp], label=label,
                     linestyle=linestyle, linewidth=2, marker='o',
                     markeredgecolor='none', color=colors[k])
        k += 1

    ax1.hold(False)

    ax1.set_xticks(n_list)
    ax1.set_xticklabels(n_list)
    ax1.set_xlabel('Number of columns in the input matrix')
    ax1.set_ylabel('Time (s)')

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.55, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               prop={'family': 'monospace'})

    plt.savefig(test_name + '.pdf')


if __name__ == '__main__':
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac
    test_rank(1e6, 1e2, only_draw=False)
    test_ncols(1e5, lambda x: x/10, only_draw=False)
    plt.show()
