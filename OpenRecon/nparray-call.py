import numpy as np
import matplotlib.pyplot as plt
import recon_nparray
my_ReconFromNPArray = recon_nparray.ReconFromNPArray(np.load(r"raw.npy")[0])
data = my_ReconFromNPArray.perform_recon()
plt.imshow(data, cmap='gray')
plt.show()