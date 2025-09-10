# Import
# Essentials
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.multiprocessing as mp
import yaml
# support files
from monaiseg_support import datastack_prep_gpu
# MONAI API
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose
)
import torch

'''
################################# INFERENCE GOES HERE #################################
'''
#Evaluate the performance of the  ???

# Viz the outputs
def segresnet_eval(inf_datatstack, cfg, segresnet_model, torch_device):
  VAL_AMP = False
  post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]) #sigmoid → threshold → binary mask for segmentation (separate no need of retrain) MONAI
  #Inference
  def inf_method(ip):
    def _compute(ip):
      return sliding_window_inference(inputs=ip, roi_size=(240, 240, 160), sw_batch_size=1, predictor=segresnet_model, overlap=0.5) #Splits data in chunks roi; batch one at a time
    if VAL_AMP:
      with torch.autocast('cuda'):
        return _compute(ip)
    else:
      return _compute(ip)
    
  # Getting checkpoint file for inferencing
  print(cfg['checkpoint'])
  segresnet_model.load_state_dict(torch.load(cfg['checkpoint'], weights_only=True, map_location=torch_device))
  segresnet_model.eval() # Setting to evaluate
  
  # # Only works with [B,{'image': None, 'label': None}] typr data
  # val_ops_stack = [{'image': None, 'label': None} for _ in range(inf_datatstack.shape[0])]
  # with torch.no_grad():
  #   for b_id in tqdm(range(inf_datatstack.shape[0])):
  #     val_ops_stack[b_id]['image'] = inf_datatstack[b_id].float().cpu()[:1]
  #     val_ops = inf_method(inf_datatstack[b_id].unsqueeze(0).to(torch_device))
  #     val_ops_stack[b_id]['label'] = post_trans(val_ops[0]).detach().cpu()[:1]
  # print(val_ops_stack[0]['image'].shape[-1])
  #nii_imgs_labels(datastack = val_ops_stack,batches_n=4,slice_n=30, img_save=cfg['out_logs'], vid_save=cfg['out_logs'], vid_T=20, args=parse_args())
  
  # Give Imgs and Labels as output separate
  model_ops_labels = []
  for batch in tqdm(inf_datatstack):
    model_ops = inf_method(batch.float().unsqueeze(0).to(torch_device))
    model_ops_labels.append(post_trans(model_ops[0]).detach().cpu())
  return np.array(model_ops_labels)[:,:1,...]

# Main inference algorithm
def monai_segresnet_inf(inf_datastack:np.ndarray):
  """
  INPUT Imgs shape:(S,H,W) and OUTPUT Labels shape:(B,C,H,W,S)
  """
  mp.freeze_support()
  torch_device = "cpu" #Setting GPU device
  yml_filepath = 'OpenRecon Monai/segresnet_firedock.yaml' #Setting yaml loc
  # Setting up for firedock_pipe_wrapper
  #Input Data Sanity Check
  if not isinstance(inf_datastack, np.ndarray):
    raise ValueError('The input data is not Numpy!')
  print("[Input] numpy array shape:", inf_datastack.shape) # Shape: [S,H,W]
  inf_datastack = datastack_prep_gpu(inf_datastack,torch_device) # Prepared dataset for inference
  print("[Input] numpy array shape:", inf_datastack.shape) # Shape: [batch_size, 4, depth, height, width]
  # Load existing config YAML
  with open(yml_filepath, "r") as f:
    cfg = yaml.safe_load(f)
  
  # Extract dir_list and make dirs
  for path in cfg.get("dir_list", {}).values():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Created (or already exists): {path}")
  
  #Create the 3D SegResNet Model
  print(cfg['segresnet_model'])
  VAL_AMP = True

  #Model Architecture
  segresnet_model = SegResNet(blocks_down=[1, 2, 2, 4],
      blocks_up=[1, 1, 1],
      init_filters=16,
      in_channels=4,
      out_channels=3,
      dropout_prob=0.2,).to(torch_device)
  
  # INFERENCE
  val_ops_labels = segresnet_eval(inf_datastack, cfg, segresnet_model, torch_device)
  return inf_datastack, val_ops_labels

# if  __name__ == '__main__':
#   inf_datastack = np.load('monai_seg_algos_openapps/monai_seg_algos/seg_imgs.npy')
#   print(inf_datastack.shape)
#   inf_datastack = np.transpose(inf_datastack,(4,0,1,2,3)) # Shape [S,B:1,C:1,H,W]
#   inf_datastack =  inf_datastack[:,0,0,...]
#   _, seg_labels = monai_segresnet_inf(inf_datastack)
#   print(inf_datastack.shape)
#   print(seg_labels.shape)