import h5py
with h5py.File('node.hdf5', 'w') as f:
    dataset = f.create_dataset("node", (node.shape[0],node.shape[1]), data=node)
