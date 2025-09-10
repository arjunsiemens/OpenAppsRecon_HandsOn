# Import Libraries
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torch

# #Nii label imgs vizualisation
# def nii_imgs_labels(datastack, batches_n:int=2, slice_n:int=10, cmap_list=['Blues','Reds','YlOrBr'], img_save: str = None, vid_save:str = None, vid_T:int = 10, args = None):
#     """
#     Combines all slices of all batches into one large image (tiled grid).
#     - Rows = batch × channel
#     - Columns = slices
#     - Masks are overlaid with transparency
#     Returns:
#         concat_img (np.ndarray): Final tiled RGB array
#     """
#     combined_rows = []
#     all_row_slcs = []
#     batches_n = min(batches_n, len(datastack))
#     slice_n = min(slice_n, datastack[0]['image'].shape[-1])

#     for batch_item in tqdm(datastack[:batches_n]):  # B,Ch,Height,Width,Slice
#         batch_imgs, batch_labels = [batch_item['image'], batch_item['label']]
#         for img_3d in batch_imgs:
#             slice0 = 0
#             row_slices = []
#             for s_id in range(slice_n):
#                 disp_img = img_3d[:, :, slice0+s_id].detach().cpu().numpy()
#                 # Normalize to [0,1]
#                 disp_rgb = np.stack([disp_img]*3, axis=-1)
#                 disp_rgb = (disp_rgb - disp_rgb.min()) / (np.ptp(disp_rgb) + 1e-5)
#                 # Overlay masks
#                 for label_i in range(batch_labels.shape[0]):
#                     mask_val = (batch_labels[label_i, :, :, slice0+s_id] > 0.5).cpu().numpy().astype(np.uint8)
#                     if mask_val.sum() > 0:
#                         cmap = plt.get_cmap(cmap_list[label_i % len(cmap_list)])
#                         color = np.array(cmap(1.0)[:3])
#                         alpha = 0.6
#                         for c in range(3):
#                             disp_rgb[..., c] = np.where(
#                                 mask_val == 1,
#                                 (1 - alpha) * disp_rgb[..., c] + alpha * color[c],
#                                 disp_rgb[..., c]
#                             )
#                 row_slices.append(disp_rgb)
#                 all_row_slcs.append(disp_rgb)
#             # Concatenate horizontally
#             row_concat = np.concatenate(row_slices, axis=1)
#             combined_rows.append(row_concat)

#     # Stack vertically
#     concat_img = np.concatenate(combined_rows, axis=0)

#     #Save as Image
#     if img_save is not None:
#         np_img = Image.fromarray((concat_img * 255).astype(np.uint8))
#         np_img.save(img_save)
#     #Save as Video
#     if vid_save is not None:
#         fps = 10
#         duration = vid_T
#         frames = all_row_slcs
#         if not frames:
#             print("Error: The frames list is empty.")
#             return
#         num_frames = int(fps * duration)
#         indices = np.linspace(0, len(frames)-1, num_frames).astype(int)
#         frames = [frames[i] for i in indices]
#         all_vals = np.concatenate([f.ravel() for f in frames])
#         vmin, vmax = all_vals.min(), all_vals.max()

#         norm_frames = [((f - vmin) / (vmax - vmin) * 255).astype(np.uint8) for f in frames]

#         iio.mimwrite(vid_save, norm_frames,  format='FFMPEG', fps=fps, codec='libx264')
#         print(f"Video saved: {vid_save} ({num_frames} frames at {fps} fps).")

#     # Display
#     plt.figure(figsize=(20, 20))
#     plt.savefig(os.path.join(args.outputloc, 'seg-ouput.png'))
#     plt.imshow(concat_img)
#     # ImageViewer(np.array(all_row_slcs),cmap='plasma')
#     plt.axis("off")
#     plt.tight_layout()
#     if args.visualize:
#         plt.show()
#     return concat_img

# Prepare datastack to GPU
def datastack_prep_gpu(datastack, torch_device):
    print(datastack.shape)
    datastack = np.transpose(np.array((datastack-datastack.min())/np.ptp(datastack)),(1,2,0)) #Normalize [-1,1] -> [H,W,S]
    datastack = np.tile(datastack,(4,1,1,1)) #[C,H,W,S]
    inf_datastack = torch.from_numpy(datastack).unsqueeze(0).to(torch_device).half()  # [B,C,H,W,S]
    print(f'DATASTACK PREP: {inf_datastack.shape}')
    return inf_datastack
# # Uncomment if you wanna use parser
# def parse_args():
#   parser = argparse.ArgumentParser()
#   # Give YAML Config file
#   parser.add_argument("-yamlconfig", type=str, default="./python-ismrmrd-server/monai_seg_algos/segresnet_firedock.yaml", help="YAML config file")
#   # Give your dataset folder or zip
#   parser.add_argument("-infdataset", type=str, default="./python-ismrmrd-server/monai_seg_algos/inf_mri_test.npy", help="Path with numpy.ndarray file")
#   # Give the output location (if required)
#   parser.add_argument("-outputloc", type=str, default="./python-ismrmrd-server/monai_seg_algos/segresnet_logs/", help="Path to save logs/plots")
#   # Give your checkpoint (if required)
#   parser.add_argument("-checkpoint", type=str, default="./python-ismrmrd-server/monai_seg_algos/segmodel_checkpoints/3Dsegresnet_monai_model.pth", help="Model weights or checkpoint")
#   # Show the interactive plots
#   parser.add_argument("-visualize", action="store_true", default=True, help="Show interactive plots and save")
#   return parser.parse_args()