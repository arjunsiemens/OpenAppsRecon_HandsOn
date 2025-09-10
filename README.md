# open research platform demo and hands on session
## Table of Contents
**[How to setup the python environment](#SetupEnvironment)**<br>
**[Exercise 1: Integrating of post-processing algo using Open Apps platform](#Exercise1)**<br>
**[Exercise 2: Integrating of reconstruction algo using Open Recon platform](#Exercise2)**<br>
**[Exercise 3: Using MONAI 3DSegResNet Model for segmentation on either given or user data.](#Exercise3)**<br>
**[Exercise 4: Integrate the MONAI 3DSegResNet Model in the reconstruction pipeline.](#Exercise4)**<br>

## <a name='SetupEnvironment'></a>How to setup the Python environment

<br>If you want to setup in Google Colab!<br>

1. Go to following link: https://drive.google.com/drive/folders/1OI0DyGY1caFL1WzLbdIVoohhtAEFDRzh?usp=sharing

2. Add this line: !git clone https://github.com/arjunsiemens/OpenAppsRecon_HandsOn.git

3. And pip install [if required!]: !pip install -r requirements.txt

### 👉USE THIS BINDER LINK: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arjunsiemens/OpenAppsRecon_HandsOn.git/main?urlpath=%2Fdoc%2Ftree%2FSiemens_Workshop_Demo.ipynb)


### <a name='Exercise1'></a>Exercise 1: Integrating of post-processing algo using Open Apps platform
1. Browse the folder 'OpenApps' in Project pane present to the left of your workspace.
2. Double click on the file 'main.py'
   ( *This is a sample program. 
     It reads 2 dicom series – T1 images and Flair images which are available in the 'data' folder, finds the difference between T1 and Flair images, displays the result in the pane right side of your window, 
     and converts the result which is in numpy array to DICOM images. Finally it exports the results to syngo via.* )
3. Run the application by clicking the button ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/57f76a87-b6fe-41e7-9b06-cb5cdbec9122) in the top right corner of the window.
   You will see the result as below. This shows the difference between two input images.
   
   ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/8c6724f9-21e5-41ea-8de0-d2785e333f93)
5. Replace the logic of finding the difference between two images to your own logic. For instance thresholding, inverting, masking the images etc.
   It can be done in the function ["def computeResult(t1_data, flair_data)"](https://github.com/OpenAppsRecon/demo/blob/main/OpenApps/main.py#L52) in line number 52
  
   ```
     def computeResult(t1_data, flair_data):  
         Logger.info("Computation of result started")
         t1_data /= np.max(t1_data)
         flair_data /= np.max(flair_data)
         difference = np.abs(t1_data - flair_data) * 4095
         difference = difference.astype(np.uint16)
         Logger.info("Computation of result finished")
         return difference
   ```
6. Select the current file in ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/b6fc629e-a4c2-4d9d-b0c2-c08e64a72e06) in the top right corner of the window
7. Run the application by clicking the button ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/57f76a87-b6fe-41e7-9b06-cb5cdbec9122) in the top right corner of the window
8. You can also debug the applicaiton by clicking the button ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/a2e80267-a69f-43d2-91e0-a8340d24ca6d) in the top right corner of the window.
9. Once you have implemented your modifications and tested by successfully running in your environment, please share the modified code with any member from Siemens team to integrate it in the open apps platform and test its performance on a clinic-like environment. 


### <a name='Exercise2'></a>Exercise 2: Integrating of reconstruction algo using Open Recon platform

1. Make sure you have setup Python development environment. If not, refer to the instructions [above](#SetupEnvironment) for guidance on setting up the Python development environment.

2. Download the Siemens raw data converted to Python NumPy format from [raw.npy](https://github.com/OpenAppsRecon/demo/tree/main/OpenRecon/raw.npy).
   (If already downloaded as part of the entire GitHub repo, then this step is not needed)

3. Download the sample Python file for performing a reconstruction from [recon_nparray.py](https://github.com/OpenAppsRecon/demo/tree/main/OpenRecon/recon_nparray.py).
   (If already downloaded as part of the entire GitHub repo, then this step is not needed)

4. Execute the sample program [nparray-call.py](https://github.com/OpenAppsRecon/demo/tree/main/OpenRecon/nparray-call.py).
<BR>The expected output:
   
![image](https://github.com/OpenAppsRecon/demo/assets/142770538/a38d85fa-d3ed-489d-9025-0b3a24114583)

5. Modify the recon_nparray.py sample program according to your needs. For example, you can try inverting the data by modifying the below code in [recon_nparray.py](https://github.com/OpenAppsRecon/demo/tree/main/OpenRecon/recon_nparray.py#L13).
   
        data = np.abs(np.fft.ifftshift(np.fft.ifft2(self.rawdata_array)))
        data *= 32767/data.max()
        data = np.around(data)
        data = data.astype(np.int16)

6. Open the nparray-call.py file in your Python IDE and execute it. For instructions on how to execute, refer to the instructions provided at https://github.com/OpenAppsRecon/demo.

7. Once you have a working Python code (i.e., the modified recon_nparray.py), you can hand it over to anyone from Siemens team for integration and performance testing on open recon platform.

### <a name='Exercise3'></a>Exercise 3: [OpenApps] Using MONAI 3DSegResNet Model for segmentation on either given or user data.

1. Run this command ["python.exe main.py -i data -o outputFolder -c configFolder"] to generate outputs via main.py.
2. Update the main.py file a/c to your requirements.
3. Give the data in proper DICOM format.

### <a name='Exercise4'></a>Exercise 4: [OpenRecon] Integrate the MONAI 3DSegResNet Model in the reconstruction pipeline.

1. Modify the code only in [monaiseg_firedock.py](https://github.com/arjunsiemens/OpenAppsRecon_HandsOn/blob/main/OpenRecon%20Monai/monaiseg_firedock.py) and [monaiseg_support.py](https://github.com/arjunsiemens/OpenAppsRecon_HandsOn/blob/main/OpenRecon%20Monai/monaiseg_support.py) a/c to your requirements.

2. Once it works in your local system, send to the volunteers to deploy on the MRI scanner.