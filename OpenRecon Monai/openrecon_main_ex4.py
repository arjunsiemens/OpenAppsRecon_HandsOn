import numpy as np
import matplotlib.pyplot as plt
# Using MONAI inf. modules for inferencing
from monaiseg_firedock import monai_segresnet_inf

#GIVE DATA HERE...
inp_path = 'OpenRecon Monai/raw.npy'
data = np.load(inp_path)

#Using MONAI 3D-SegResNet Model inferencing
"""
INFO.: INPUT Imgs shape:(S,H,W) and OUTPUT Labels shape:(B,C,H,W,S)
"""
data = np.expand_dims(data, axis=0) 
_, seg_labels = monai_segresnet_inf(data) #OUTPUT Labels shape:(S,H,W)

#Making seg_labels shape same as input
seg_labels = np.transpose(seg_labels, (4,0,1,2,3))[:,0,0,...] #Making it (S,H,W) for plotting
for label in seg_labels:
    plt.imshow(label, cmap='gray')
    plt.show()
