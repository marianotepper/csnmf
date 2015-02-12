import dask.array as da
from into import into
from dask.array.into import discover
from dask.dot import dot_graph
import tempfile
import rcnmf.tsqr

x = da.random.standard_normal(size=(100, 100), blockshape=(100, 50))

temp_file = tempfile.NamedTemporaryFile(suffix='.hdf5')

uri = temp_file.name + '::/X'
into(uri, x)

data = into(da.Array, uri, blockshape=(100, 100))

omega = da.random.standard_normal(size=(100, 20), blockshape=(100, 20))
mat_h = data.dot(omega)

q, r = rcnmf. tsqr.tsqr(mat_h, blockshape=(100, 20))

print data.shape
print q.shape

mul = data.dot(q)

dot_graph(data.dask, filename='data')
dot_graph(omega.dask, filename='omega')
dot_graph(q.dask, filename='q')
dot_graph(mul.dask, filename='mul')

uri = temp_file.name + '::/mul'
into(uri, mul)

temp_file.close()
