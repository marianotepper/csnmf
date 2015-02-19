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
import scipy.io
import h5py
import csnmf.snmf


def run(mat, ncols, blockshape):

    algorithms = ['SPA', 'XRAY']
    compress = [False, True]
    data = [mat, da.from_array(mat, blockshape=blockshape)]

    res_list = []
    for tup in itertools.product(algorithms, compress, data):

        t = timeit.default_timer()
        cols, mat_h, error = csnmf.snmf.compute(tup[2], ncols, 'SPA',
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

            plot_func(x, dict_res[(alg, dtype)][comp], label=label,
                      linestyle=linestyle, linewidth=2, marker='o',
                      markeredgecolor='none', color=colors[k])
        k += 1

    plt.hold(False)


def test_climate(filename, plot_func, only_draw=False):

    test_name = 'test_' + filename
    f = scipy.io.loadmat('../data/' + filename + '.mat')

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


def create_url_dataset():
    """
    Data downloaded from http://sysnet.ucsd.edu/projects/url/
    :return:
    """

    f = scipy.io.loadmat('../data/url.mat')
    shape_data = np.zeros((2, 121))

    shape = f['Day1'][0, 0][0].shape

    h5f = h5py.File('../data/url.hdf5', 'w')
    dset = h5f.create_dataset("mat", (shape[1], shape[0]))

    step = 50
    for start in range(0, shape[0], step):
        end = min(start + step, shape[0])
        print(start, end)
        mat = f['Day1'][0, 0][0][start:end, :]
        dset[:, start:end] = mat.toarray().T

    # for i, name in enumerate(sorted(f.keys())):
    #     if name[:3] != 'Day':
    #         continue
    #     if f[name][0, 0][0].shape[0] < 2e4:
    #         print(name)
    #     shape_data[:, i] = f[name][0, 0][0].shape


def test_url(create_dataset=False, only_draw=False):
    create_url_dataset()
    # data = f['A']
    # n = data.shape[1]
    # blockshape = (int(1e3), n)


if __name__ == '__main__':
    plt.switch_backend('TkAgg')  # otherwise, monospace fonts do not work in mac
    test_climate('air_mon', plt.plot, only_draw=False)
    test_climate('air_day', plt.semilogy, only_draw=False)
    # test_url(create_dataset=True, only_draw=False)
    plt.show()