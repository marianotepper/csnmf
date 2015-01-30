"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from itertools import count, product
import numpy as np
from functools import partial
from dask.array import Array


def _dims_from_size(size, blocksize):
    """

    >>> list(_dims_from_size(30, 8))
    [8, 8, 8, 6]
    """
    while size - blocksize > 0:
        yield blocksize
        size -= blocksize
    yield size


def _blockdims_from_blockshape(shape, blockshape):
    """
    Convert blockshape to dimensions along each axis

    >>> _blockdims_from_blockshape((30, 30), (10, 10))
    ((10, 10, 10), (10, 10, 10))
    >>> _blockdims_from_blockshape((30, 30), (12, 12))
    ((12, 12, 6), (12, 12, 6))
    """
    return tuple(map(tuple, map(_dims_from_size, shape, blockshape)))


_names = {}


def fromfunction(func, size=None, blockshape=None, blockdims=None, name=None):
    """
    Create a dask Array object using a generator function. It is intended to be
    used with numpy functions, but supports custom functions
    :param func: the function to use
        It must have a single parameter 'size', corresponding to the full size
        of the output matrix
    :param size: tuple of ints
        size of the output matrix
    :param blockshape: tuple of ints
        size of block size
    :param blockdims: iterable of tuples
        block sizes along each dimension
    :param name: string
        dask variable name that will be used as input
    :return: dask.core.Array
    """

    if hasattr(func, '__name__'):
        funcname = func.__name__
    else:
        funcname = 'nameless_function'

    if funcname not in _names:
        _names[funcname] = (funcname + '_%d' % i for i in count(1))

    if not size:
        size = (1, 1)
        blockshape = (1, 1)

    if blockshape:
        blockdims = _blockdims_from_blockshape(size, blockshape)

    name = name or next(_names[funcname])
    keys = product([name], *[range(len(bd)) for bd in blockdims])
    vals = product([func], product(*blockdims))
    dsk = dict(zip(keys, vals))

    return Array(dsk, name, shape=size, blockshape=blockshape)


_random_functions = [np.random.exponential, np.random.gumbel, np.random.laplace,
                    np.random.logistic, np.random.normal, np.random.poisson,
                    np.random.rayleigh, np.random.standard_cauchy,
                    np.random.standard_exponential, np.random.standard_gamma,
                    np.random.standard_normal]

_current_module = __import__(__name__)
for _func in _random_functions:
    setattr(_current_module, _func.__name__, partial(fromfunction, _func))


if __name__ == "__main__":

    fromfunction(np.random.standard_normal, size=(100, 100), blockshape=(50, 50))
    fromfunction(np.random.standard_normal, size=(100, 100), blockshape=(51, 51))
    fromfunction(np.random.standard_gamma, size=(100, 100), blockshape=(50, 50))
    fromfunction(np.random.standard_gamma, size=(100, 100), blockshape=(51, 51))

    fromfunction(np.random.standard_normal)
    fromfunction(np.random.standard_normal)
    fromfunction(np.random.standard_gamma)
    fromfunction(np.random.standard_gamma)

    standard_gamma()
    standard_gamma(size=(100, 100), blockshape=(50, 50), blockdims=None, name=None)

    fromfunction(np.zeros, size=(100, 100), blockshape=(51, 51))
    fromfunction(partial(np.random.beta, a=1, b=1), size=(100, 100), blockshape=(51, 51))