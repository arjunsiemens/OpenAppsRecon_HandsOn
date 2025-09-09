import numpy as np
import torch
import onnxruntime as ort
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
import SimpleITK as sitk

class MONAISegResNetONNX:
    def __init__(self, model_path, providers=['CUDAExecutionProvider',"CPUExecutionProvider"]):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name #Symbolic name of inps in model graph
        # self.output_name = self.session.get_outputs()[0].name #Symbolic name of ops in model graph
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu" #Setting GPU device
        self.val_amp = True #Default
    
    def onnx_infer(self, inputs):
        ort_inputs = {self.input_name: inputs.cpu().numpy()}
        ort_outs = self.session.run(None, ort_inputs)
        return torch.Tensor(ort_outs[0]).to(inputs.device)

    def predict(self, input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=self.onnx_infer,
                overlap=0.5,
            )
        if torch.cuda.is_available() and self.val_amp:
            with torch.autocast(self.torch_device):
                return _compute(input)
        else:
            return _compute(input)
    
    def preprocess(self, dataset):
        """Preprocesses dataset: Normalize, apply divisible padding, and reshape for ONNX input."""
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset) + 1e-8)  # Normalize to [0,1]
        return dataset.astype(np.float32)

    def run_inf(self, np_input: np.ndarray) -> np.ndarray:
        """
        Run inference on any numpy input with shape (B:1, C:1, H:240, W:240, S:160) for 3D SegResNet.
        Returns numpy output as labels.
        Expected Input: [C S x y], [S x y]
        """
        #print(f'The inp shape: {np_input.shape}')            
        if len(np_input.shape) == 4:
            #print(f'The inp shape: {np_input.shape}') 
            np_input = np_input[0]

        np_input = np.expand_dims(np.expand_dims(np_input,axis=0),axis=0) # Adding B and C
        np_input = self.preprocess(np.transpose(np_input,(0,1,3,4,2))) # [B,C,H,W,S]
        np_input = np.repeat(np_input,4,axis=1)
        #print(f'The inp shape: {np_input.shape}')
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        with torch.no_grad():
            ort_ops = self.predict(torch.Tensor(np_input).to(self.torch_device))
            ort_op_labels = post_trans(torch.Tensor(ort_ops[0]).to(self.torch_device)).unsqueeze(0)
        ort_op_labels = np.clip(ort_op_labels, 0, 1) # Clip [0,1] (if required!)
        ort_op_labels = np.transpose(np_input,(0,1,4,2,3))[:,0,...] if len(np_input.shape) == 4 else np.transpose(np_input,(0,1,4,2,3))[0,0,...]# [B,S,H,W] restored!
        #print(f'The opt shape:{ort_op_labels.shape}')
        return ort_op_labels

#def readDicomImages(in_dir):
 #   print("Reading dicom images started")
 #   t1_img = None
 #   flair_img = None
 #   flair_series_reader = None
 #   series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(in_dir)

 #   for series_ID in series_IDs :
 #       series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(in_dir, series_ID)

  #      series_reader = sitk.ImageSeriesReader()
   #     series_reader.SetFileNames(series_files)
    #    series_reader.MetaDataDictionaryArrayUpdateOn()
   #     series_reader.LoadPrivateTagsOn()
    #    img = series_reader.Execute()
    #    series_description = series_reader.GetMetaData(0, "0008|103e")

    #    if "T1" in series_description:
    #        t1_img = img
    #    elif "FLAIR" in series_description :
     #       flair_img = img
     #       flair_series_reader = series_reader
     #   elif "gre" in series_description :
      #      flair_img = img
      #      flair_series_reader = series_reader
    #print("Reading dicom images finished")
    #return flair_img, flair_series_reader

#def readImagesIntoNPArray(flair_img):
    #t1_data = sitk.GetArrayFromImage(t1_img).astype(np.float32)    
    #flair_data = sitk.GetArrayFromImage(flair_img).astype(np.float32)    
    #return flair_data

