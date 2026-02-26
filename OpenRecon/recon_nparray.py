import numpy as np
import matplotlib.pyplot as plt

class ReconFromNPArray:
    def __init__(self, rawdata_array):
        self.rawdata_array = rawdata_array
    
    def perform_recon(self):
        data = np.abs(np.fft.ifftshift(np.fft.ifft2(self.rawdata_array)))
        # data = np.abs(np.fft.ifftshift(np.fft.ifft2(self.rawdata_array, axes=(1 ,2)), axes=(1 ,2)))
        
        if ((np.ndim(data)>2)) : 
            # if the number of dimensions are greater than 2 then assumption is that zeroth dimension is channel
            # Sum of squares coil combination
            data = np.abs(data)
            data = np.square(data)
            data = np.sum(data, axis=0)
            data = np.sqrt(data)
        
        # Normalize and convert to int16
        mymaxval = 4095
        data *= mymaxval/data.max()
        data = np.around(data)

        #data = mymaxval/2-data
        data = np.abs(data)
        data = data.astype(np.int16)
        #plt.imshow(data)
        return data
