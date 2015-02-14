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
import rcnmf.snmf

if __name__ == '__main__':

    m = 100000
    n = 100
    q = 10
    ncols = 5

    print(m, n)

    x = np.fabs(np.random.standard_normal(size=(m, q)))
    y = np.fabs(np.random.standard_normal(size=(q, n)))
    data = x.dot(y)

    algorithms = ['SPA', 'xray']
    compress = [False, True]
    data_type = [data, da.from_array(data, blockshape=(10000, 100))]

    for tup in itertools.product(algorithms, compress, data_type):

        t = timeit.default_timer()
        res = rcnmf.snmf.compute(tup[2], ncols, 'SPA', compress=tup[1])
        t = timeit.default_timer() - t

        if isinstance(tup[2], np.ndarray):
            data_type = 'np'
        elif isinstance(tup[2], da.Array):
            data_type = 'da'

        base_str = 'algorithm: {alg:4s}; compressed: {comp:d}; ' \
                   'type: {data_type}; columns {cols}; error {error:.4f};' \
                   'time {time:.2f}'
        print(base_str.format(alg=tup[0], comp=tup[1], data_type=data_type,
                              cols=res[0], error=res[2], time=t))