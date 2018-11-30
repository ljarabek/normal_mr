import nibabel as nib
import gzip
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from multi_slice_viewer import multi_slice_viewer
import matplotlib.pyplot as plt

# with gzip.open(, 'rb') as file:
#    nii = file.read()
#
# p#rint(nii)
lol = nib.load("C:\MR slike\clinical-ms-axis2-slice100/"
               "00c3b6bfcaaab48405b3cb0f610ff761a899b757a330129c03d367f8/brainmask.nii.gz")
print(lol.get_fdata().shape)


# plt.imshow(lol.get_fdata()[:,:,0])
# plt.show()

class DataStream:

    def __init__(self, batch_size, root_dir="C:/MR slike/healthy-axis2-slice100/"):  # TODO: to choose masked, proces ..
        """
        :type batch_size: int
        :type root_dir: str
        """
        self.batchSize = batch_size
        self.root = root_dir

        self.dct = self.read()  # reads data, returns dct

        self.moments = pd.DataFrame(index=sorted(list(self.dct.keys())), columns=['mean', 'std'])
        self.normalize()  # fill self.moments dataFrame

        self.currentBatch = self.get_batch()


    def read(self):  # Reads float data, makes self.dct =  {id: data}
        ids = []
        data = []
        dct = {}
        # TODO: brainmask, segmentation
        for folder in tqdm(sorted(os.listdir(self.root))):
            ids.append(folder)
            for file in sorted(os.listdir(self.root + folder)):
                if 't1w' in file:
                    data.append(np.array(nib.load(self.root + folder + "/" + file).get_fdata()))
        data = np.array(data)
        ids = np.array(ids)
        for dt, i_ in zip(data, ids):
            dct[i_] = dt
        return dct

    def normalize(self):  # makes dct data 0-mean 1-std

        for key in self.dct.keys():
            im = self.dct[key][:, :, 0]
            im = np.ma.masked_less(np.array(im, np.float32), 0.3 * np.mean(im))
            self.moments['mean'][key] = im.mean()
            self.moments['std'][key] = im.std()

    # weights argument in DataFrame.sample()!!!
    def get_batch(self):
        batch = []
        ids = []
        sample = self.moments.sample(n=self.batchSize)
        for id, mom in sample.iterrows():
            im = np.array((self.dct[id] - mom['mean']) / mom['std'])
            batch.append(im)
            ids.append(id)
        batch = np.array(batch)

        return ids, batch

        # for file in os.listdir()


print(DataStream(8).get_batch())  # returns none!
# for lol in DataStream(5).dct:
#    print(lol)


"""
lol = nib.load("C:\MR slike\clinical-ms-axis2-slice100/"
               "00c3b6bfcaaab48405b3cb0f610ff761a899b757a330129c03d367f8/t1w.nii.gz")
print(lol)
PRINTS THIS:
data shape (193, 229, 1)
affine: 
[[   1.    0.    0.  -96.]
 [   0.    1.    0. -132.]
 [   0.    0.    1.   22.]
 [   0.    0.    0.    1.]]
metadata:
<class 'nibabel.nifti1.Nifti1Header'> object, endian='<'
sizeof_hdr      : 348
data_type       : b''
db_name         : b''
extents         : 0
session_error   : 0
regular         : b'r'
dim_info        : 0
dim             : [  3 193 229   1   1   1   1   1]
intent_p1       : 0.0
intent_p2       : 0.0
intent_p3       : 0.0
intent_code     : none
datatype        : float32
bitpix          : 32
slice_start     : 0
pixdim          : [ 1.  1.  1.  1.  0.  0.  0.  0.]
vox_offset      : 0.0
scl_slope       : nan
scl_inter       : nan
slice_end       : 0
slice_code      : unknown
xyzt_units      : 2
cal_max         : 0.0
cal_min         : 0.0
slice_duration  : 0.0
toffset         : 0.0
glmax           : 0
glmin           : 0
descrip         : b''
aux_file        : b''
qform_code      : scanner
sform_code      : unknown
quatern_b       : 0.0
quatern_c       : 0.0
quatern_d       : 0.0
qoffset_x       : -96.0
qoffset_y       : -132.0
qoffset_z       : 22.0
srow_x          : [ 0.  0.  0.  0.]
srow_y          : [ 0.  0.  0.  0.]
srow_z          : [ 0.  0.  0.  0.]
intent_name     : b''
magic           : b'n+1'
"""
