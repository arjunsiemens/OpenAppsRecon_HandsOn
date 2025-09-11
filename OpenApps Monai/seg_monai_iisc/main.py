import os
import SimpleITK as sitk
import numpy as np
from pydicom.uid import generate_uid
import argparse
import shutil
import sys
import logging
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import inf_monaiseg_onnx as inf_monai

isDebug = False

Logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter( logging.Formatter('%(asctime)s %(levelname)s: %(filename)s:%(funcName)s(%(lineno)d): %(message)s'))
Logger.addHandler(handler)

Logger.setLevel(20)

def readDicomImages(in_dir):
    print("Reading dicom images started")
    t1_img = None
    flair_img = None
    flair_series_reader = None
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(in_dir)
    if not series_IDs:
        raise ValueError(f"No DICOM series found in: {in_dir}")
 
    for series_ID in series_IDs :
        # Collect file names for this series
        series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(in_dir, series_ID)
 
        # Setup reader
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_files)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        # Read the series as image
        img = series_reader.Execute()
        series_description = series_reader.GetMetaData(0, "0008|103e") if series_reader.HasMetaDataKey(0, "0008|103e") else "Unknown"
       
        # Save image + metadata + geometry
        if "T1" in series_description:
            t1_img = img
        elif "FLAIR" in series_description :
            flair_img = img
            flair_series_reader = series_reader
        elif "gre" in series_description :
            flair_img = img
            flair_series_reader = series_reader
    Logger.info("Reading dicom images finished")
    #Logger.info(flair_img.GetSpacing())
    #Logger.info(flair_img.GetOrigin())
    #Logger.info(flair_img.GetDirection())
    return flair_img, flair_series_reader

def run_monai_segmentation(data,model_path = "OpenApps Monai/seg_monai_iisc/model/monai_3dsegresnet_best.onnx"):        
    seg_monai = inf_monai.MONAISegResNetONNX(model_path)
    result_img = seg_monai.run_inf(data)  
    
    return result_img   

def readImagesIntoNPArray(flair_img):
    #t1_data = sitk.GetArrayFromImage(t1_img).astype(np.float32)    
    flair_data = sitk.GetArrayFromImage(flair_img).astype(np.float32)    
    return flair_data

# Modify this function for your needs
def computeResult(t1_data, flair_data):
    Logger.info("Computation of result started")
    t1_data /= np.max(t1_data)
    flair_data /= np.max(flair_data)
    difference = np.abs(t1_data + flair_data) / 4095
    difference = difference.astype(np.uint16)
    Logger.info("Computation of result finished")
    return difference

def showResult(t1_data, result):
    global isDebug
    if isDebug:
        #plot the random slices
        sliceCount = t1_data.shape[0]
        random_slices = np.random.randint(0, sliceCount, 4)
        f, axarr = plt.subplots(2,2, figsize = (30,30))
        axarr[0,0].imshow(result[random_slices[0]],cmap="gray")
        axarr[0,1].imshow(result[random_slices[1]],cmap="gray")
        axarr[1,0].imshow(result[random_slices[2]],cmap="gray")
        axarr[1,1].imshow(result[random_slices[3]],cmap="gray")
        plt.show()
    return
    
def convertResultToImage(result, flair_img):
    result_img = sitk.GetImageFromArray(result)
    result_img.SetDirection(flair_img.GetDirection())
    result_img.SetOrigin(flair_img.GetOrigin())
    result_img.SetSpacing(flair_img.GetSpacing())
    return result_img

def saveResultImageAsDicom(result_img, outputFolder,flair_series_reader):
    Logger.info("Saving result as dicom started")
    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()
    voxels = sitk.GetArrayFromImage(result_img).astype(np.uint16)
    max_voxel = np.max(voxels)
    uid_series = generate_uid()
    for i in range(result_img.GetDepth()):
        image_slice = result_img[:, :, i]
        # Tags shared by the series.
        for key in flair_series_reader.GetMetaDataKeys(i):
            image_slice.SetMetaData(key, flair_series_reader.GetMetaData(i, key))
            if key == "0020|000e" : # series UID
                image_slice.SetMetaData(key, uid_series)
            if key == "0028|0107": # largest pixel value
                image_slice.SetMetaData(key, str(max_voxel))
            if key == "0028|1050": # center
                image_slice.SetMetaData(key, str(max_voxel/2))
            if key == "0028|1051": # width
                image_slice.SetMetaData(key, str(max_voxel))
            if key == "0008|0018": # Sop instance UID
                image_slice.SetMetaData(key, generate_uid())
        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.
        writer.SetFileName(os.path.join(outputFolder, str(i) + ".dcm"))
        writer.Execute(image_slice)
    Logger.info("Saving the result as dicom finished")
    return

def sendResultsToSyngo(tempFolder, resultFolder):
    Logger.info("Sending the result to syngo via server is started")
    file_names = os.listdir(tempFolder)
    for aFile in os.listdir(resultFolder):
        filePath = os.path.join(resultFolder, aFile)
        if os.path.isfile(filePath):
            os.remove(filePath)
    Logger.info("Result files are:")
    for file_name in file_names:
        Logger.info(file_name)
        shutil.move(os.path.join(tempFolder, file_name), resultFolder)
    Logger.info("Sending the result to syngo via server is finished")
    return

def main(argv):
    Logger.warning('Main function invoked')
    Logger.info("Arguments passed : " + str(argv))
    parser = argparse.ArgumentParser("")
    parser.add_argument('--inputFolder', "-i", help='Input folder which contains the source dicom images')
    parser.add_argument('--outputFolder', "-o", help='Output folder to store the result dicom data')
    parser.add_argument('--configFolder', "-c", help='Folder containing the configuration.json File')
    parser.add_argument('--tempFolder', "-t", help='Folder to store temporary data')
    parser.add_argument('--dbFolder', "-db", help="optional argument to specify the folder where to store the dicom database")
    parser.add_argument('--logFolder', "-l", help="optional argument to specify the folder where to store teh logfile")
    args = parser.parse_args(argv)
    flair_img, flair_series_reader = readDicomImages(args.inputFolder)
    flair_data = readImagesIntoNPArray(flair_img)    
    computed_data = run_monai_segmentation(flair_data) 
        
    result_img = convertResultToImage(computed_data, flair_img)
    saveResultImageAsDicom(result_img, args.configFolder, flair_series_reader)
    sendResultsToSyngo(args.configFolder, args.outputFolder)
    Logger.info("Main function Finished")
    return
    

if __name__ == "__main__":
    if isDebug:
        dirname = os.path.dirname(__file__)
        args = []
        args.append(os.path.join("-i"))
        args.append(os.path.join(dirname, "data"))
        args.append(os.path.join("-o"))
        args.append(os.path.join(dirname, "outputFolder"))
        args.append(os.path.join("-c"))
        args.append(os.path.join(dirname, "configFolder"))
        main(args)
    else:
        main(sys.argv[1:])
    
    