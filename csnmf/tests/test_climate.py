"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
from operator import mul
import numpy as np
import dask.array as da
import timeit
import itertools
import matplotlib.pyplot as plt
import pickle
import scipy.io
import csnmf.snmf
import csnmf.third_party


def run(mat, ncols, blockshape, compute_qr_until):

    algorithms = ['SPA', 'XRAY']
    compress = [False, True]
    data_list = [mat, da.from_array(mat, blockshape=blockshape)]

    base_str = 'algorithm: {alg:4s}; compressed: {comp:d}; ' \
               'type: {data_type:11s}; error {error:.4f}; time {time:.2f}'

    res_list = []
    for alg, comp, data in itertools.product(algorithms, compress, data_list):

        if isinstance(data, np.ndarray):
            dtype = 'in-core'
        elif isinstance(data, da.Array):
            dtype = 'out-of-core'

        if not comp and ncols > compute_qr_until:
            res_dict = {'alg': alg, 'comp': comp, 'data_type': dtype,
                        'cols': None, 'error': np.nan, 'time': np.nan}

            print(base_str.format(**res_dict))
            res_list.append(res_dict)
            continue

        t = timeit.default_timer()
        cols, mat_h, error = csnmf.snmf.compute(data, ncols, alg, compress=comp)
        t = timeit.default_timer() - t

        res_dict = {'alg': alg, 'comp': comp, 'data_type': dtype,
                    'cols': sorted(cols), 'error': error, 'time': t}
        print(base_str.format(**res_dict))
        print(cols)
        res_list.append(res_dict)

    return res_list


def plot(x, dict_res, plot_func):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    plt.hold(True)
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

            y = dict_res[(alg, dtype)][comp]
            valid = np.isfinite(y).flatten()
            if not np.any(valid):
                continue
            plot_func(x[valid], y[valid], label=label,
                      linestyle=linestyle, linewidth=2,
                      marker='o', markeredgecolor='none', color=colors[k])
        k += 1

    plt.hold(False)


def test_climate(filename, plot_func, q_max=11, compute_qr_until=11,
                 only_draw=False):

    test_name = 'test_' + filename

    q_list = np.arange(1, q_max, dtype=np.int)
    shape = (len(q_list), 1)

    if not only_draw:

        f = scipy.io.loadmat('../data/' + filename + '.mat')
        data = f['A']
        n = data.shape[1]
        blockshape = (int(1e3), n)

        time_vecs = {}
        err_vecs = {}

        for i, q in enumerate(q_list):
            res_list = run(data, q, blockshape, compute_qr_until)
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
    plot(q_list, time_vecs, plot_func)
    ax = plt.axes()
    ax.set_xticks(q_list)
    ax.set_xticklabels(q_list)
    ax.set_xlabel('Number of extracted columns')
    ax.set_ylabel('Time (s)')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              prop={'family': 'monospace'})

    plt.savefig(test_name + '_time.pdf')

    plt.figure(figsize=(10, 5))
    plot(q_list, err_vecs, plt.plot)
    ax = plt.axes()
    ax.set_xticks(q_list)
    ax.set_xticklabels(q_list)
    ax.set_xlabel('Number of extracted columns')
    ax.set_ylabel('Relative error')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])

    # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #           prop={'family': 'monospace'})

    plt.savefig(test_name + '_error.pdf')


if __name__ == '__main__':
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac
    test_climate('air_mon', plt.plot, only_draw=False)
    test_climate('air_day', plt.semilogy, q_max=11, compute_qr_until=2,
                 only_draw=False)
    plt.show()