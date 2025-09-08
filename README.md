# open research platform demo and hands on session
## Table of Contents
**[How to setup the python environment](#SetupEnvironment)**<br>
**[Exercise 1: Integrating of post-processing algo using Open Apps platform](#Exercise1)**<br>
**[Exercise 2: Integrating of reconstruction algo using Open Recon platform](#Exercise2)**<br>

## <a name='SetupEnvironment'></a>How to setup the Python environment
BINDER LINK: [Use this interactive binder](https://mybinder.org/v2/gh/arjunsiemens/OpenAppsRecon_HandsOn/e5f14047d4a9b2de743ae00ae9191495df62a4a5?urlpath=lab%2Ftree%2FSiemens_Workshop_Demo.ipynb)

1.	Download the PyCharm community edition and install it.
https://www.jetbrains.com/pycharm/download/?section=windows
     * While installing PyCharm, please check the checkbox “Create Desktop Shortcut”.      
![image](https://github.com/OpenAppsRecon/demo/assets/142770538/58ae3a7e-2892-4405-a9e2-879fb0427db4)



2.	Download Python and install it. Remember the path where Python is installed.
https://www.python.org/downloads/

3. Open PyCharm by double-clicking on the icon (as shown below) on your desktop
   ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/3047ddad-19fc-4c68-8cce-b7dfef8de40e)

4. When the PyCharm is started the first time after installation, the following popup will appear.
        Please click OK and continue
   ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/58254e72-e9aa-4267-a1d2-27e23e97179d)

5. PyCharm opens and it appears as shown below
   
     ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/57e82f8b-f37e-4706-9df7-0e2511b26473)

6. Click on the button  ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/48018137-a52f-43cc-abd0-ed28e018b1b0) to open the project, below window appears

7. Browse to the folder where you have downloaded the project from the Githut repository.
    ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/821497c8-7a83-454f-818d-7483e1f2ecdc)
    ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/937bafb2-a588-4a1c-8aba-cc1c27686ab6)
    ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/6c3cb30d-6d0e-442a-9919-b63b8aea60ec)

8. Open File->Setting->Python Interpreter
   ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/abc0192f-0733-4ffb-b0c9-581f247b0755)

   Point Python Interpreter toth directory where you have installed python.exe Step 2.
9. Click on + symbole to add the following packages.
    SimpleITK ,pydicom, numpy  ,matplotlib
   ![image](https://github.com/OpenAppsRecon/demo/assets/142770538/68e95f54-4715-4a03-9de2-325ad1c9c76f)

   
   
11.	Open helloworld/welcome.py and run file.

12.	Now you can start with either of the two below exercises ([reconstruction](#Exercise2) or [postprocessing](#Exercise1)) based on your preference. 
	

Enjoy the session

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
