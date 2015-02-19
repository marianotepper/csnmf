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
from into import into
from dask.array.into import discover # required by dask.array
import timeit
import os
import matplotlib.pyplot as plt
import tempfile
import rcnmf.compression as randcomp
import rcnmf.tsqr


def select_blocksize(k):
    return min(k/10, int(1e4))

def compression_vs_qr(m, n, q):

    x = np.random.standard_normal(size=(m, n))

    blockshape = (select_blocksize(m), n)
    data = from_array(x, blockshape=blockshape)

    t = timeit.default_timer()
    data_comp, _ = randcomp.compress(data, q, n_power_iter=0)
    np.array(data_comp)
    tid_comp = timeit.default_timer() - t

    t = timeit.default_timer()
    rcnmf.tsqr.qr(data)
    np.array(data_comp)
    tid_qr = timeit.default_timer() - t

    t = timeit.default_timer()
    randcomp.compress(x, q, n_power_iter=0)
    tim_comp = timeit.default_timer() - t

    t = timeit.default_timer()
    np.linalg.qr(x)
    tim_qr = timeit.default_timer() - t

    # temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5')
    # filename = temp_file.name
    # into(filename + '::/X', x)
    # hdf5size = float(os.path.getsize(filename))
    # temp_file.close()
    hdf5size = 0

    return hdf5size, tid_comp, tid_qr, tim_comp, tim_qr


def test_compression_vs_qr(only_draw=False):

    m = int(1e4)
    sizes_n = map(int, [1e2, 3e2, 5e2, 7.5e2,
                        1e3, 3e3, 5e3, 7.5e3])

    q = 10
    repetitions = 1

    if not only_draw:
        shape = (len(sizes_n), repetitions)
        times_in_disk_comp = np.zeros(shape)
        times_in_disk_qr = np.zeros(shape)
        times_in_memory_comp = np.zeros(shape)
        times_in_memory_qr = np.zeros(shape)
        # hdf5sizes = np.zeros((1, len(sizes_n)))

        for i, n in enumerate(sizes_n):
            print('iteration: {0}, size: {1}, {2}'.format(i, m, n))
            for k in range(repetitions):
                res = compression_vs_qr(m, n, q)
                # hdf5sizes[0, i] = res[0]
                times_in_disk_comp[i, k] = res[1]
                times_in_disk_qr[i, k] = res[2]
                times_in_memory_comp[i, k] = res[3]
                times_in_memory_qr[i, k] = res[4]

        if repetitions > 0:
            times_in_disk_comp = np.mean(times_in_disk_comp, axis=1)
            times_in_disk_qr = np.mean(times_in_disk_qr, axis=1)
            times_in_memory_comp = np.mean(times_in_memory_comp, axis=1)
            times_in_memory_qr = np.mean(times_in_memory_qr, axis=1)
        # hdf5sizes /= 2**30

        with open('test_compression_ooc_result', 'w') as f:
            np.save(f, times_in_disk_comp)
            np.save(f, times_in_disk_qr)
            np.save(f, times_in_memory_comp)
            np.save(f, times_in_memory_qr)
            # np.save(f, hdf5sizes)

    with open('test_compression_ooc_result', 'r') as f:
        times_in_disk_comp = np.load(f)
        times_in_disk_qr = np.load(f)
        times_in_memory_comp = np.load(f)
        times_in_memory_qr = np.load(f)
        # hdf5sizes = np.load(f)

    colors = ['#e41a1c', '#377eb8']

    plt.figure()
    ax1 = plt.axes()

    ax1.hold(True)
    ax1.loglog(sizes_n, times_in_memory_comp, label='In-core compression',
               marker='o', markeredgecolor=colors[0], linewidth=2,
               linestyle='-', color=colors[0])
    ax1.loglog(sizes_n, times_in_memory_qr, label='In-core QR', marker='o',
               markeredgecolor=colors[0], linewidth=2, linestyle='--',
               color=colors[0])
    ax1.loglog(sizes_n, times_in_disk_comp, label='Out-of-core compression',
               marker='o', markeredgecolor=colors[1], linewidth=2,
               linestyle='-', color=colors[1])
    ax1.loglog(sizes_n, times_in_disk_qr, label='Out-of-core QR', marker='o',
               markeredgecolor=colors[1], linewidth=2, linestyle='--',
               color=colors[1])
    ax1.hold(False)
    ax1.set_xlim(4e3, max(sizes_n) + 4e3)

    ax1.set_xticks(sizes_n)
    ax1.set_xticklabels(['{:.1e}'.format(z) for z in sizes_n], rotation=45)
    ax1.set_xlabel(r'Number $m$ of columns')
    ax1.set_ylabel('Time (s)')

    ax1.legend(loc='upper left')

    # ax2 = ax1.twiny()
    # ax2.loglog(sizes_m, times_in_memory_comp, linewidth=0)
    #
    # ax2.set_xlim(4e3, max(sizes_m) + 4e3)
    #
    # ax2.set_xticks(sizes_m)
    # ax2.set_xticklabels(['{:.2f}'.format(z) for z in hdf5sizes.tolist()[0]])
    # ax2.set_xlabel('Size of the hdf5 file (GB)')

    plt.subplots_adjust(bottom=0.15)

    plt.savefig('test_compression_vs_qr.pdf')


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

    # temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5')
    # filename = temp_file.name
    # into(filename + '::/X', x)
    # hdf5size = float(os.path.getsize(filename))
    # temp_file.close()
    hdf5size = 0

    return hdf5size, tid, tim


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
        # hdf5sizes = np.zeros((1, len(sizes_m)))

        for i, s in enumerate(sizes_m):
            print('iteration: {0}, size: {1}, {2}'.format(i, s, n))
            for k in range(repetitions):
                res = size_timing(s, n, q)
                # hdf5sizes[0, i] = res[0]
                times_in_disk[i, k] = res[1]
                times_in_memory[i, k] = res[2]

        if repetitions > 0:
            times_in_disk = np.mean(times_in_disk, axis=1)
            times_in_memory = np.mean(times_in_memory, axis=1)
        # hdf5sizes /= 2**30

        with open('test_compression_ooc_result', 'w') as f:
            np.save(f, times_in_disk)
            np.save(f, times_in_memory)
            # np.save(f, hdf5sizes)

    with open('test_compression_ooc_result', 'r') as f:
        times_in_disk = np.load(f)
        times_in_memory = np.load(f)
        # hdf5sizes = np.load(f)

    plt.figure()
    ax1 = plt.axes()

    ax1.hold(True)
    ax1.loglog(sizes_m, times_in_memory, label='In-core', marker='o',
               markeredgecolor='b', linewidth=2, linestyle='-', color='b')
    ax1.loglog(sizes_m, times_in_disk, label='Out-of-core', marker='o',
               markeredgecolor='r', linewidth=2, linestyle='-', color='r')
    ax1.hold(False)
    ax1.set_xlim(4e3, max(sizes_m) + 4e3)

    ax1.set_xticks(sizes_m)
    ax1.set_xticklabels(['{:.1e}'.format(z) for z in sizes_m], rotation=45)
    ax1.set_xlabel(r'Number $m$ of rows')
    ax1.set_ylabel('Time (s)')

    ax1.legend(loc='upper left')

    # ax2 = ax1.twiny()
    # ax2.loglog(sizes_m, times_in_memory, linewidth=0)
    #
    # ax2.set_xlim(4e3, max(sizes_m) + 4e3)
    #
    # ax2.set_xticks(sizes_m)
    # ax2.set_xticklabels(['{:.2f}'.format(z) for z in hdf5sizes.tolist()[0]])
    # ax2.set_xlabel('Size of the hdf5 file (GB)')

    plt.subplots_adjust(bottom=0.15)

    plt.savefig('test_compression_ic_vs_ooc.pdf')

if __name__ == '__main__':
    # test_compression_ic_vs_ooc(only_draw=False)
    test_compression_vs_qr(only_draw=False)
    plt.show()