import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np


in_file = r"data/mprage.nii.gz"

def read_nifty_image(in_file):
    return  sitk.ReadImage(in_file)

#read the nifty image
t1_img = sitk.ReadImage(in_file)

#read the array from the image
t1_data = sitk.GetArrayFromImage(t1_img)

#plot the random slices
size = t1_img.GetSize()
random_slices = np.random.randint(10, size[2], 4)

f, axarr = plt.subplots(2,2, figsize = (30,30))
axarr[0,0].imshow(t1_data[random_slices[0]])
axarr[0,1].imshow(t1_data[random_slices[1]])
axarr[1,0].imshow(t1_data[random_slices[2]])
axarr[1,1].imshow(t1_data[random_slices[3]])
plt.show()



