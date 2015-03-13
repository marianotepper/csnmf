"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
import numpy as np
from dask.array import from_array
import timeit
import matplotlib.pyplot as plt
import csnmf.compression as randcomp


def select_blocksize(k):
    return min(k/10, int(1e4))


def compression_vs_qr(m, n, q):

    x = np.random.standard_normal(size=(m, n))

    t = timeit.default_timer()
    randcomp.compress(x, q, n_power_iter=0)
    tim_comp = timeit.default_timer() - t

    t = timeit.default_timer()
    np.linalg.qr(x)
    tim_qr = timeit.default_timer() - t

    return tim_comp, tim_qr


def test_compression_vs_qr(only_draw=False):

    m = int(1e4)
    sizes_n = map(int, [1e2, 3e2, 5e2, 7.5e2,
                        1e3, 3e3, 5e3, 7.5e3])

    repetitions = 1

    if not only_draw:
        shape = (len(sizes_n), repetitions)
        times_in_memory_comp = np.zeros(shape)
        times_in_memory_qr = np.zeros(shape)

        for i, n in enumerate(sizes_n):
            print('iteration: {0}, size: {1}, {2}'.format(i, m, n))
            for k in range(repetitions):
                q = n/10
                res = compression_vs_qr(m, n, q)
                times_in_memory_comp[i, k] = res[0]
                times_in_memory_qr[i, k] = res[1]

        if repetitions > 0:
            times_in_memory_comp = np.mean(times_in_memory_comp, axis=1)
            times_in_memory_qr = np.mean(times_in_memory_qr, axis=1)

        with open('test_compression_vs_qr', 'w') as f:
            np.save(f, times_in_memory_comp)
            np.save(f, times_in_memory_qr)

    with open('test_compression_vs_qr', 'r') as f:
        times_in_memory_comp = np.load(f)
        times_in_memory_qr = np.load(f)

    colors = ['#e41a1c', '#377eb8']

    plt.figure()
    ax1 = plt.axes()

    sizes_n_crop = [s for s in sizes_n if s < 1e4]
    idx = [i for i, s in enumerate(sizes_n) if s < 1e4]
    ax1.hold(True)
    ax1.loglog(sizes_n_crop, times_in_memory_comp[idx],
               label='In-core compression', marker='o',
               markeredgecolor=colors[0], linewidth=2,
               linestyle='-', color=colors[0])
    ax1.loglog(sizes_n_crop, times_in_memory_qr[idx], label='In-core QR',
               marker='o', markeredgecolor=colors[1], linewidth=2,
               linestyle='--', color=colors[1])
    ax1.hold(False)

    ax1.set_xticks(sizes_n_crop)
    ax1.set_xticklabels(['{:.1e}'.format(z) for z in sizes_n_crop], rotation=45)
    ax1.set_xlabel(r'Number $m$ of columns')
    ax1.set_ylabel('Time (s)')
    ax1.legend(loc='upper left')
    plt.subplots_adjust(bottom=0.15)

    plt.savefig('test_compression_vs_qr_ic.pdf')


def size_timing(m, n, q):

    x = np.random.standard_normal(size=(m, n))

    blockshape = (select_blocksize(m), select_blocksize(n))
    data = from_array(x, blockshape=blockshape)

    t = timeit.default_timer()
    data_comp, _ = randcomp.compress(data, q)
    np.array(data_comp)
    tid = timeit.default_timer() - t

    t = timeit.default_timer()
    randcomp.compress(x, q, n_power_iter=0)
    tim = timeit.default_timer() - t

    return tid, tim


def test_compression_ic_vs_ooc(only_draw=False):

    sizes_m = map(int, [1e3, 3e3, 5e3, 7.5e3,
                        1e4, 3e4, 5e4, 7.5e4,
                        1e5, 3e5, 5e5, 7.5e5,
                        1e6])
    n = int(5e2)
    q = 10
    repetitions = 1

    if not only_draw:
        shape = (len(sizes_m), repetitions)
        times_in_disk = np.zeros(shape)
        times_in_memory = np.zeros(shape)

        for i, s in enumerate(sizes_m):
            print('iteration: {0}, size: {1}, {2}'.format(i, s, n))
            for k in range(repetitions):
                res = size_timing(s, n, q)
                times_in_disk[i, k] = res[0]
                times_in_memory[i, k] = res[1]

        if repetitions > 0:
            times_in_disk = np.mean(times_in_disk, axis=1)
            times_in_memory = np.mean(times_in_memory, axis=1)

        with open('test_compression_ic_vs_ooc', 'w') as f:
            np.save(f, times_in_disk)
            np.save(f, times_in_memory)

    with open('test_compression_ic_vs_ooc', 'r') as f:
        times_in_disk = np.load(f)
        times_in_memory = np.load(f)

    colors = ['#e41a1c', '#377eb8']

    plt.figure()
    ax1 = plt.axes()

    ax1.hold(True)
    ax1.loglog(sizes_m, times_in_memory, label='In-core compression',
               marker='o', markeredgecolor=colors[0], linewidth=2,
               linestyle='-', color=colors[0])
    ax1.loglog(sizes_m, times_in_disk, label='Out-of-core compression',
               marker='o', markeredgecolor=colors[1], linewidth=2,
               linestyle='--', color=colors[1])
    ax1.hold(False)

    ax1.set_xticks(sizes_m)
    ax1.set_xticklabels(['{:.1e}'.format(z) for z in sizes_m], rotation=45)
    ax1.set_xlabel(r'Number $m$ of rows')
    ax1.set_ylabel('Time (s)')
    ax1.legend(loc='upper left')
    plt.subplots_adjust(bottom=0.15)

    plt.savefig('test_compression_ic_vs_ooc.pdf')

if __name__ == '__main__':
    test_compression_ic_vs_ooc(only_draw=False)
    test_compression_vs_qr(only_draw=False)
    plt.show()