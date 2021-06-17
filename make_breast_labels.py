import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

breastmask = np.asarray(nib.load('base_images_phantom4/breastmask.nii').dataobj)
imshape = breastmask.shape

label_map = np.zeros(imshape).astype(np.int)

label_map[breastmask==1] = 1

chestmask = np.asarray(nib.load('base_images_phantom4/chestmask.nii').dataobj)

label_map[chestmask==1] = 2

skinmask = np.asarray(nib.load('base_images_phantom4/skinmask.nii').dataobj)

label_map[skinmask==1] = 3

fgtmask = np.asarray(nib.load('base_images_phantom4/fgtmask.nii').dataobj)

label_map[fgtmask==1] = 4

fatmask = np.multiply(breastmask, np.subtract(1,fgtmask))

label_map[fatmask==1] = 5

# downsample data
label_map = label_map[::2,::2,:]
print(label_map.shape)

nifti_map = nib.Nifti1Image(label_map,np.eye(4))

nib.save(nifti_map, 'base_images_phantom4/breast_label_map.nii.gz')