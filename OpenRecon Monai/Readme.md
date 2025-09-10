
# OpenRecon Hands On - Follow below steps:

1. Make sure you have setup Python development environment. If not, refer to the instructions from https://github.com/OpenAppsRecon/demo for guidance on setting up the Python development environment.

2. Download the Siemens raw data converted to Python NumPy format from https://github.com/OpenAppsRecon/demo/tree/main/OpenRecon/raw.npy.
   (If already downloaded as part of the entire GitHub repo, then this step is not needed)

3. Download the sample Python file for performing a reconstruction from https://github.com/OpenAppsRecon/demo/tree/main/OpenRecon/recon_nparray.py.
   (If already downloaded as part of the entire GitHub repo, then this step is not needed)

4. Execute the sample program nparray-call.py from https://github.com/OpenAppsRecon/demo/tree/main/OpenRecon/nparray-call.py.
<BR>The expected output:
   
![image](https://github.com/OpenAppsRecon/demo/assets/142770538/a38d85fa-d3ed-489d-9025-0b3a24114583)

5. Modify the recon_nparray.py sample program according to your needs. For example, you can try inverting the data by modifying the below code.
   
        data = np.abs(np.fft.ifftshift(np.fft.ifft2(self.rawdata_array)))
        data *= 32767/data.max()
        data = np.around(data)
        data = data.astype(np.int16)

6. Open the nparray-call.py file in your Python IDE and execute it. For instructions on how to execute, refer to the instructions provided at https://github.com/OpenAppsRecon/demo.

7. Once you have a working Python code (i.e., the modified recon_nparray.py), you can hand it over for integration.
