import os
import monai
import torch
import argparse

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from torch import nn
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from dataset import SegmentationDataset
import torchvision.transforms as transforms
from utils import *

parser = argparse.ArgumentParser(
    prog="Segmentation"
)

parser.add_argument("--arch", type=str)
parser.add_argument("--gpuid", type=int, default=0)

def main():
    args = parser.parse_args()

    ## Define your own label
    ## cate_label = []

    device = f"cuda:{args.gpuid}"
    batch_size = 8
    num_epochs = 100
    weight_decay = 1e-4
    lr = 1e-3
    workers = 1
    img_size = 256

    dataset = ## Define your dataset name
    project_name = ## Define your project name
    model_directory = ## Define the directory to save your model
    model_name = ## Define your segmentation model name

    df_train_meta = ## Read your training Dataframe
    df_val_meta = ## Read your validation Dataframe
        
    if args.arch == "unet":
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        model.to(device)
    elif args.arch == "swinunet":
        model = monai.networks.nets.SwinUNETR(
            img_size=(img_size, img_size),
            in_channels=3,
            out_channels=2,
            spatial_dims=2
        )
        model.to(device)
    
    train_transform_img = transforms.Compose(
        ## Define your training augmentation for images
    )

    train_transform_mask = transforms.Compose(
        ## Define your training augmentation for masks
    )
    
    val_transform_img = transforms.Compose(
        ## Define your validation augmentation for images
    )

    val_transform_mask = transforms.Compose(
        ## Define your validation augmentation for masks
    )
    
    train_dataset = SegmentationDataset(
        df_train_meta, 
        data_folder,
        class_specifier=cate_label,
        transform_img=train_transform_img,
        transform_mask=train_transform_mask,
        generator_seed=42
    )
    
    val_dataset = SegmentationDataset(
        df_val_meta, 
        data_folder,
        class_specifier=cate_label,
        transform_img=val_transform_img,
        transform_mask=val_transform_mask,
        generator_seed=42
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        
    print(f"Length of train dataset: {len(train_dataloader.dataset)}")
    print(f"Length of val dataset: {len(val_dataloader.dataset)}")
        
    criterion_dice = monai.losses.DiceLoss(include_background=True, to_onehot_y=True, sigmoid=True, reduction="mean").to(device)
    criterion_CE = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = np.inf
    
    for epoch in tqdm(range(num_epochs), desc="Training", leave=False):
        print(f"------- Epoch {epoch} -------")
        ## Training
        train_dice_loss = 0.
        train_CE_loss = 0.
        train_loss = 0.
        model.train()
        for i, data_batch in enumerate(train_dataloader):
            img = data_batch["image"].type(torch.FloatTensor).to(device)
            mask = data_batch["mask"].type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                pred_mask = model(img)

                ## Dice + CE loss
                loss_dice = criterion_dice(pred_mask, mask)
                loss_CE = criterion_CE(pred_mask, torch.squeeze(mask.type(torch.LongTensor), 1).to(device))
                loss = loss_dice + loss_CE

                loss.backward()
                optimizer.step()

                loss_dice = loss_dice.detach().cpu().numpy().item()
                loss_CE = loss_CE.detach().cpu().numpy().item()
                loss = loss.detach().cpu().numpy().item()

                train_dice_loss += loss_dice
                train_CE_loss += loss_CE
                train_loss += loss

            if i % 100 == 0:
                print(f"{i}/{len(train_dataloader)}   Dice loss: {loss_dice:.5f}   CE loss: {loss_CE:.5f}   Training loss: {loss:.5f}")

        train_dice_loss /= len(train_dataloader)
        train_CE_loss /= len(train_dataloader)
        train_loss /= len(train_dataloader)

        print(f"Epoch: {epoch}   Epoch dice loss: {train_dice_loss:.5f}   Epoch CE loss: {train_CE_loss:.5f}   Epoch training loss: {train_loss:.5f}")

        ## Eval
        val_dice_loss = 0.
        val_CE_loss = 0.
        val_loss = 0.

        model.eval()

        for i, data_batch in enumerate(val_dataloader):
            img = data_batch["image"].type(torch.FloatTensor).to(device)
            mask = data_batch["mask"].type(torch.FloatTensor).to(device)

            with torch.set_grad_enabled(False):
                pred_mask = model(img)

                ## Dice + CE loss
                loss_dice = criterion_dice(pred_mask, mask)
                loss_CE = criterion_CE(pred_mask, torch.squeeze(mask.type(torch.LongTensor), 1).to(device))
                loss = loss_dice + loss_CE

                loss_dice = loss_dice.detach().cpu().numpy().item()
                loss_CE = loss_CE.detach().cpu().numpy().item()
                loss = loss.detach().cpu().numpy().item()

                val_dice_loss += loss_dice
                val_CE_loss += loss_CE
                val_loss += loss

        val_dice_loss /= len(val_dataloader)
        val_CE_loss /= len(val_dataloader)
        val_loss /= len(val_dataloader)

        print(f"Epoch: {epoch}   Epoch dice loss: {val_dice_loss:.5f}   Epoch CE loss: {val_CE_loss:.5f}   Epoch eval loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "val_loss": best_val_loss
            }, f"./{model_directory}/{model_name}.pth")
                
        lr_scheduler.step()
        
if __name__ == "__main__":
    main()