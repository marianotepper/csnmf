"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
import numpy as np
from dask.array import Array, random
from into import into
from dask.array.into import discover # required by dask.array
import timeit
import os
import matplotlib.pyplot as plt
import tempfile
import rcnmf.compression as randcomp


def size_timing(m, n, q):

    def select_blocksize(k):
        blocksize = 1
        while k / (10 * blocksize) >= 1:
            blocksize *= 10
        return min(blocksize, int(1e4))

    blockshape = (select_blocksize(m), select_blocksize(n))
    print('block shape {0}'.format(blockshape))

    x = random.standard_normal(size=(m, n), blockshape=blockshape)

    temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5')
    filename = temp_file.name

    into(filename + '::/X', x)
    hdf5size = float(os.path.getsize(filename))

    data = into(Array, filename + '::/X', blockshape=blockshape)

    t = timeit.default_timer()
    data_comp, comp = randcomp.compress(data, q)
    np.array(data_comp)
    tid = timeit.default_timer() - t

    if hdf5size / (2**30) < 20:
        t = timeit.default_timer()
        data = np.array(data)
        randcomp.compress(data, q, n_power_iter=0)
        tim = timeit.default_timer() - t
    else:
        tim = np.nan

    temp_file.close()

    return hdf5size, tid, tim


def run(only_draw=False):

    sizes_m = map(int, [5e3, 7.5e3, 1e4, 3e4, 5e4, 7.5e4, 1e5])
    n = int(5e3)
    q = 10
    repetitions = 1

    if not only_draw:
        shape = (len(sizes_m), repetitions)
        times_in_disk = np.zeros(shape)
        times_in_memory = np.zeros(shape)
        hdf5sizes = np.zeros((1, len(sizes_m)))

        for i, s in enumerate(sizes_m):
            print('iteration: {0}, size: {1}, {2}'.format(i, s, n))
            for k in range(repetitions):
                res = size_timing(s, n, q)
                hdf5sizes[0, i] = res[0]
                times_in_disk[i, k] = res[1]
                times_in_memory[i, k] = res[2]

        if repetitions > 0:
            times_in_disk = np.mean(times_in_disk, axis=1)
            times_in_memory = np.mean(times_in_memory, axis=1)
        hdf5sizes /= 2**30

        with open('test_compression_ooc_result', 'w') as f:
            np.save(f, times_in_disk)
            np.save(f, times_in_memory)
            np.save(f, hdf5sizes)

    with open('test_compression_ooc_result', 'r') as f:
        times_in_disk = np.load(f)
        times_in_memory = np.load(f)
        hdf5sizes = np.load(f)

    print(sizes_m)
    print(hdf5sizes)
    print(times_in_memory)
    print(times_in_disk)
    print(times_in_disk / times_in_memory)

    fig = plt.figure()
    ax1 = plt.axes()

    ax1.hold(True)
    line1 = ax1.loglog(sizes_m, times_in_memory,
                       label='In-core',
                       marker='o', markeredgecolor='b',
                       linewidth=2, linestyle='-', color='b')
    line3 = ax1.loglog(sizes_m, times_in_disk,
                       label='Out-of-core',
                       marker='o', markeredgecolor='r',
                       linewidth=2, linestyle='-', color='r')
    ax1.hold(False)
    ax1.set_xlim(4e3, max(sizes_m) + 4e3)

    ax1.set_xticks(sizes_m)
    ax1.set_xticklabels(sizes_m)
    ax1.set_xlabel(r'Number $m$ of rows')
    ax1.set_ylabel('Time (s)')

    ax1.legend(loc='upper left')

    ax2 = ax1.twiny()
    ax2.loglog(sizes_m, times_in_memory, linewidth=0)

    ax2.set_xlim(4e3, max(sizes_m) + 4e3)

    ax2.set_xticks(sizes_m)
    ax2.set_xticklabels(['{:.2f}'.format(z) for z in hdf5sizes.tolist()[0]])
    ax2.set_xlabel('Size of the hdf5 file (GB)')


if __name__ == '__main__':
    run(only_draw=True)
    plt.show()