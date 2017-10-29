% HDF5 Write
with h5py.File('test.h5', 'w') as f:
    f['data'] = data

% HDF5 read
fileNameh5 = 'test.h5'
data = hdf5read(fileNameh5,'data');

