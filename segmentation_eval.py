import os
import cv2
import monai
import torch

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch import nn
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from UNet import UNet
from SwinUNet import SwinUNETR

## Load your trained model and testing data here
## df_test_meta = ...
## model = ...
## img_size = ...
## mask_size = ...
## class_specifier = ...

## Below is an example code to run volume-wise evaluation
model.eval()

## For RGB input
norm_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

average_dice_metric = monai.metrics.DiceMetric(include_background=False, get_not_nans=False, ignore_empty=False)
average_surface_distance_metric = monai.metrics.SurfaceDistanceMetric(include_background=False, get_not_nans=False, symmetric=True)

for patient, df_by_patient in df_test_meta.groupby("Patient"):
    for exam, df_by_exam in df_by_patient.groupby("Exam"):
        for vol, df_by_vol in df_by_exam.groupby("Vol"):
            mask_3d = None
            pred_mask_3d = None
            for i, row in df_by_vol.iterrows():
                img = Image.open() ## Load your image
                mask = Image.open() ## Load your mask

                np_img = np.array(img)
                np_mask = np.array(mask)

                np_img_resize = cv2.resize(np_img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                np_mask_resize = cv2.resize(np_mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)

                np_mask_resize = np.isin(np_mask_resize, class_specifier).astype("uint8")

                img = Image.fromarray(np_img_resize.astype("uint8")) 

                img = transforms.ToTensor()(img)
                img = norm_transform(img)

                np_mask_resize = np.expand_dims(np_mask_resize, axis=0)
                mask = torch.from_numpy(np_mask_resize)

                img = torch.unsqueeze(img, dim=0)
                mask = torch.unsqueeze(mask, dim=0)

                img = img.type(torch.FloatTensor).to(device)
                mask = mask.type(torch.FloatTensor).to(device)

                pred_mask = model(img)

                pred_mask = F.sigmoid(pred_mask)
                pred_mask = pred_mask.detach().cpu().squeeze()[1].numpy()
                pred_mask = np.where(pred_mask > 0.5, 1, 0).astype("float")
                pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(dim=0).unsqueeze(dim=0)
                
                if mask_3d is None:
                    mask_3d = mask.detach().cpu().unsqueeze(dim=-1)
                else:
                    mask_3d = torch.concat([mask_3d, mask.detach().cpu().unsqueeze(dim=-1)], dim=-1)

                if pred_mask_3d is None:
                    pred_mask_3d = pred_mask_tensor.unsqueeze(dim=-1)
                else:
                    pred_mask_3d = torch.concat([pred_mask_3d, pred_mask_tensor.unsqueeze(dim=-1)], dim=-1)

            average_dice_metric(pred_mask_3d, mask_3d)
            average_surface_distance_metric(pred_mask_3d, mask_3d)

average_dice = average_dice_metric.aggregate(reduction="mean").item()
average_asd = average_surface_distance_metric.aggregate(reduction="mean").item()

print(f"Average DICE: {average_dice:.4f}")
print(f"Average ASSD: {average_asd:.4f}")


                

