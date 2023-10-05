
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
