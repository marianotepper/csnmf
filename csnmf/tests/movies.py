"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
from operator import mul, sub
from scipy import misc
import h5py
import timeit
import numpy as np


def png_movie_to_hdf5_matrix(dirname, interval, hdf_filename):
    img_name = '{0:05d}.png'
    temporal_length = -reduce(sub, interval)
    t = timeit.default_timer()
    for i, img_idx in enumerate(range(*interval)):
        if i % 100 == 0:
            print(i, temporal_length)
        img = misc.imread(dirname + img_name.format(img_idx))
        if i == 0:
            f = h5py.File(hdf_filename, 'w')
            img_shape = f.create_dataset('img_shape', (len(img.shape),))
            img_shape[:] = img.shape
            data_shape = (reduce(mul, img.shape), temporal_length)
            data = f.create_dataset('data', data_shape, maxshape=data_shape)
        data[:, i] = img.flatten().astype(np.float) / 255
        if i % 1000 == 0:
            f.flush()
    t = timeit.default_timer() - t
    print(t)
