import numpy as np
import matplotlib.pyplot as plt

class ReconFromNPArray:
    def __init__(self, rawdata_array):
        self.rawdata_array = rawdata_array
    
    def perform_recon(self):
        data = np.abs(np.fft.ifftshift(np.fft.ifft2(self.rawdata_array)))
        
        # Normalize and convert to int16
        data *= 32767/data.max()
        data = np.around(data)
        data = data.astype(np.int16)
        #plt.imshow(data)
        return data
