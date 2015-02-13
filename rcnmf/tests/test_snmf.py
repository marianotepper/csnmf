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


if __name__ == '__main__':

    import time
    np.random.seed(10)

    m = 10000
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

        t = time.clock()
        cols, H, rel_res = rcnmf.snmf.compute(data, ncols, 'SPA')
        t = time.clock() - t
        print('original:', cols, rel_res, t)

        t = time.clock()
        cols, H, rel_res = rcnmf.snmf.compute(data, ncols, 'SPA', compress=True)
        t = time.clock() - t
        print('compressed:', cols, rel_res, t)