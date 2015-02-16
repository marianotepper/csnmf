"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
import numpy as np
import rcnmf.snmf
import timeit

if __name__ == '__main__':

    m = 20000
    n = 1000
    q = 8
    ncols = 5

    print(m, n)

    x = np.fabs(np.random.standard_normal(size=(m, q)))
    y = np.fabs(np.random.standard_normal(size=(q, n)))
    data = x.dot(y)

    algorithms = ['SPA', 'xray']
    for alg in algorithms:
        print(alg + ' algorithm:')

        t = timeit.default_timer()
        res1 = rcnmf.snmf.compute(data, ncols, 'SPA')
        t1 = timeit.default_timer() - t

        t = timeit.default_timer()
        res2 = rcnmf.snmf.compute(data, ncols, 'SPA', compress=True)
        t2 = timeit.default_timer() - t

        base_str = 'columns {0}; error {1:.4f}; time {2:.2f}'
        print('original  : ' + base_str.format(res1[0], res1[2], t1))
        print('compressed: ' + base_str.format(res2[0], res2[2], t2))
