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
import h5py
import pickle
import scipy.io
import rcnmf.snmf


def run(mat, ncols, blockshape):

    algorithms = ['SPA', 'xray']
    compress = [False, True]
    data = [mat, da.from_array(mat, blockshape=blockshape)]

    res_list = []
    for tup in itertools.product(algorithms, compress, data):

        t = timeit.default_timer()
        cols, mat_h, error = rcnmf.snmf.compute(tup[2], ncols, 'SPA',
                                            compress=tup[1])
        t = timeit.default_timer() - t

        if isinstance(tup[2], np.ndarray):
            dtype = 'in-core'
        elif isinstance(tup[2], da.Array):
            dtype = 'out-of-core'

        diff = mat - np.dot(mat[:, cols], mat_h)
        error = np.linalg.norm(diff) / np.linalg.norm(mat)

        res_dict = {'alg': tup[0], 'comp': tup[1], 'data_type': dtype,
                    'cols': sorted(cols), 'error': error, 'time': t}

        base_str = 'algorithm: {alg:4s}; compressed: {comp:d}; ' \
                   'type: {data_type:11s}; error {error:.4f}; ' \
                   'time {time:.2f}'
        print(base_str.format(**res_dict))
        res_list.append(res_dict)

    return res_list


def plot(x, dict_res, ax):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    ax.hold(True)
    k = 0
    for (alg, dtype) in dict_res.keys():
        for comp in dict_res[(alg, dtype)]:
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

            ax.plot(x, dict_res[(alg, dtype)][comp], label=label,
                    linestyle=linestyle, linewidth=2, marker='o',
                    markeredgecolor='none', color=colors[k])
        k += 1

    ax.hold(False)
    ax.set_xticks(x)
    ax.set_xticklabels(x)


def plot_bivalued(x, dict_res1, dict_res2, ax):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    ax.hold(True)
    k = 0
    for (alg, dtype) in dict_res1.keys():
        for comp in dict_res1[(alg, dtype)]:
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

            ax.plot(dict_res1[(alg, dtype)][comp], dict_res2[(alg, dtype)][comp], label=label,
                    linestyle=linestyle, linewidth=2, marker='o',
                    markeredgecolor='none', color=colors[k])
        k += 1

    ax.hold(False)


def test_jester(only_draw=False):

    test_name = 'test_jester'

    with h5py.File('../data/jester-2.h5', 'r') as f:

        idx = f['data']['int0']
        values = f['data']['double1']
        m, n = np.max(idx, axis=1)
        data = np.zeros((m, n))
        data[idx[0, :] - 1, idx[1, :] - 1] = values[:]

        nonzero_rows = np.any(data != 0, axis=1)
        nonzero_cols = np.any(data != 0, axis=0)
        masked_data = data[nonzero_rows, :]
        masked_data = masked_data[:, nonzero_cols]

        n = masked_data.shape[1]
        blockshape = (int(1e4), n)

        q_list = range(n/10, n+1, n/10)
        shape = (len(q_list), 1)

        if not only_draw:
            time_vecs = {}
            err_vecs = {}

            for i, q in enumerate(q_list):
                res_list = run(masked_data, q, blockshape)
                for res in res_list:
                    key = (res['alg'], res['data_type'])
                    if key not in time_vecs:
                        time_vecs[key] = {}
                        time_vecs[key][True] = np.zeros(shape)
                        time_vecs[key][False] = np.zeros(shape)
                        err_vecs[key] = {}
                        err_vecs[key][True] = np.zeros(shape)
                        err_vecs[key][False] = np.zeros(shape)
                    time_vecs[key][res['comp']][i] = res['time']
                    err_vecs[key][res['comp']][i] = res['error']

            with open(test_name, 'w') as f:
                pickle.dump(time_vecs, f)
                pickle.dump(err_vecs, f)

        with open(test_name, 'r') as f:
            time_vecs = pickle.load(f)
            err_vecs = pickle.load(f)

        plt.figure(figsize=(10, 5))
        ax = plt.axes()
        plot(q_list, time_vecs, ax)
        ax.set_xlabel('Rank of the input matrix')
        ax.set_ylabel('Time (s)')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  prop={'family': 'monospace'})

        plt.savefig(test_name + '_time.pdf')

        plt.figure(figsize=(10, 5))
        ax = plt.axes()
        plot(q_list, err_vecs, ax)
        ax.set_xlabel('Rank of the input matrix')
        ax.set_ylabel('Relative error')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  prop={'family': 'monospace'})

        plt.savefig(test_name + '_error.pdf')


def test_climate(only_draw=False):

    test_name = 'test_climate'

    f = scipy.io.loadmat('../data/air_mon.mat')

    data = f['A']
    n = data.shape[1]
    blockshape = (int(1e3), n)

    q_list = range(1, 11)
    shape = (len(q_list), 1)

    if not only_draw:
        time_vecs = {}
        err_vecs = {}

        for i, q in enumerate(q_list):
            res_list = run(data, q, blockshape)
            for res in res_list:
                key = (res['alg'], res['data_type'])
                if key not in time_vecs:
                    time_vecs[key] = {}
                    time_vecs[key][True] = np.zeros(shape)
                    time_vecs[key][False] = np.zeros(shape)
                    err_vecs[key] = {}
                    err_vecs[key][True] = np.zeros(shape)
                    err_vecs[key][False] = np.zeros(shape)
                time_vecs[key][res['comp']][i] = res['time']
                err_vecs[key][res['comp']][i] = res['error']

        with open(test_name, 'w') as f:
            pickle.dump(time_vecs, f)
            pickle.dump(err_vecs, f)

    with open(test_name, 'r') as f:
        time_vecs = pickle.load(f)
        err_vecs = pickle.load(f)

    plt.figure(figsize=(10, 5))
    ax = plt.axes()
    plot(q_list, time_vecs, ax)
    ax.set_xlabel('Rank of the input matrix')
    ax.set_ylabel('Time (s)')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              prop={'family': 'monospace'})

    plt.savefig(test_name + '_time.pdf')

    plt.figure(figsize=(10, 5))
    ax = plt.axes()
    plot(q_list, err_vecs, ax)
    ax.set_xlabel('Rank of the input matrix')
    ax.set_ylabel('Relative error')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              prop={'family': 'monospace'})

    plt.savefig(test_name + '_error.pdf')


if __name__ == '__main__':
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac
    # test_jester(only_draw=True)
    test_climate(only_draw=False)
    plt.show()