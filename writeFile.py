import h5py
with h5py.File('node.hdf5', 'w') as f:
    dataset = f.create_dataset("node", (node.shape[0],node.shape[1]), data=node)


#%% read h5df file
import h5py
with h5py.File('node.hdf5', 'r') as f1:
    print(list(f1.keys()))  # print list of root level objects
    node = f1['node']  
    node = node[:]
