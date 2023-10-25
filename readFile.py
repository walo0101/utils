
import pydicom
import glob

#%% read folder
patient_path = 'data/train/PATIENT/'
# patient_list = sorted(glob.glob(patient_path + '*.dcm')) 
x = []
for path in sorted(glob.glob(f'{patient_path}/*/**/***/', recursive=True)):
    patient_list = sorted(glob.glob(path + '*.dcm'))  
    flen = len(patient_list)
    for i in range(flen): 
        dp = pydicom.read_file(patient_list[i])
        x.append(dp.pixel_array)
    
mask_path =  'data/train/MASK/'
# mask_list = sorted(glob.glob(mask_path + '*.dcm'))
y = []
for mpath in sorted(glob.glob(f'{mask_path}/*/**/***/', recursive=True)):
    mask_list = sorted(glob.glob(mpath + '*.dcm'))  
    flen = len(mask_list)
    for j in range(flen):
        dm = pydicom.read_file(mask_list[j])
        y.append(dm.pixel_array)

import pandas as pd
# read csv file
dataframe = pd.read_csv("file.csv", header=None, engine='python') #option header=None
data = dataframe.values

#%% read h5df file
import h5py
with h5py.File('node.hdf5', 'r') as f1:
    print(list(f1.keys()))  # print list of root level objects
    node = f1['node']  
    node = node[:]
