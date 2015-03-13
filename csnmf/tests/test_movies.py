"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import print_function
from operator import mul, sub
import h5py
import timeit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from into import into
from dask.array.into import discover
from dask.array import Array
import math
import csnmf.snmf
from csnmf.third_party import mrnmf
from movies import png_movie_to_hdf5_matrix


def test_movie(hdf_filename, base_output_name, ncols=None, interval=None,
               max_blockshape=(1e5, 100)):

    f = h5py.File(hdf_filename, 'r')
    img_shape = np.array(f['img_shape'], dtype=np.int)
    f.close()

    m = min(max_blockshape[0], reduce(mul, img_shape))
    if interval is not None:
        n = min(max_blockshape[1], -reduce(sub, interval))
    else:
        n = max_blockshape[1]
    m = int(m)
    n = int(n)
    data = into(Array, hdf_filename + '::/data', blockshape=(m, n))
    if interval is not None:
        data = data[:, interval[0]:interval[1]]
        data = np.array(data)

    if ncols is None:
        ncols = data.shape[1] / 120

    print(data.shape, ncols, m, n)

    t = timeit.default_timer()
    cols, mat_h, error = csnmf.snmf.compute(data, ncols, 'SPA', compress=True)
    t = timeit.default_timer() - t
    print(error)

    data = np.array(data)
    error = mrnmf.nnls_frob(data, cols)[1]

    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    cols_order = argsort(cols)
    cols = sorted(cols)
    mat_h = mat_h[cols_order, :]

    res_dict = {'cols': cols, 'error': error, 'time': t}
    base_str = 'error {error:.4f}; time {time:.2f}; cols {cols}'
    print(base_str.format(**res_dict))

    if interval is not None and ncols <= 10:

        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                 '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
        cmap = ListedColormap(colors)

        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        out = cv2.VideoWriter(base_output_name + '.avi',
                              fourcc, 8.0, (img_shape[1], img_shape[0]), True)
        max_val = np.argmax(mat_h, axis=0)
        for i in range(data.shape[1]):
            img = np.reshape(data[:, i], img_shape) * 255
            img = img.astype(np.uint8)
            norm_idx = float(max_val[i]) / ncols
            c = map(lambda x: int(x*255), cmap(norm_idx))[::-1]
            cv2.rectangle(img, (img_shape[1]-50, img_shape[0]-50),
                          (img_shape[1], img_shape[0]), c, cv2.cv.CV_FILLED)
            out.write(img)
        out.release()

        border_width = 40
        arrangement = int(math.ceil(math.sqrt(ncols)))
        plt.figure()
        for i, c in enumerate(cols):
            img = np.reshape(data[:, c], img_shape)
            norm_idx = float(i) / ncols
            ax = plt.subplot(arrangement, arrangement, i+1,
                             axisbg=cmap(norm_idx))
            ax.imshow(img, aspect='equal', origin='lower',
                      extent=(border_width, img_shape[1] - border_width,
                              border_width, img_shape[0] - border_width))
            ax.imshow(img, alpha=0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.savefig(base_output_name + '_representatives.pdf', dpi=300)

        mat_h_norm = mat_h / np.sum(mat_h, axis=0)
        plt.figure()
        ax = plt.axes()
        for i in range(ncols):
            bottom = np.sum(mat_h_norm[:i, :], axis=0)
            norm_idx = float(i) / ncols
            ax.bar(range(data.shape[1]), mat_h_norm[i, :],  1,
                   color=cmap(norm_idx),
                   linewidth=0, bottom=bottom)
        ax.set_ylim(0, 1)

        plt.savefig(base_output_name + '_activation.pdf', dpi=300)

    for i, c in enumerate(cols):
        img = np.reshape(data[:, c], img_shape)
        plt.figure()
        ax = plt.axes()
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(base_output_name + '_representative_{0}.png'.format(i))
        plt.close()

    plt.close('all')


def process_elephant_dreams(resolution, interval=None, ncols=None,
                            create_matrix=False):
    if create_matrix:
        prefix = '/Volumes/MyBookDuo/movies/ED-{0:d}'.format(resolution)
        png_movie_to_hdf5_matrix(prefix + '-png/', (1, 15692), prefix + '.hdf5')

    prefix = '/Volumes/MyBookDuo/movies/ED-{0:d}'.format(resolution)
    base_output_name = 'elephantDreams_{0:d}p'.format(resolution)
    if interval is not None:
        base_output_name += '_{0:05d}_{1:05d}'.format(*interval)

    test_movie(prefix + '.hdf5', base_output_name,
               ncols=ncols, interval=interval)


if __name__ == '__main__':
    process_elephant_dreams(360, interval=(600, 720), ncols=6)
    process_elephant_dreams(360, interval=(600, 720), ncols=9)
    process_elephant_dreams(360, interval=(600, 720), ncols=15)

    # These two tests take a LONG time to run
    process_elephant_dreams(360)
    process_elephant_dreams(1080)

    plt.show()