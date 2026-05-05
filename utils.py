import os
import math
import torch
import numpy as np
from PIL import Image
from loguru import logger
from torchvision.utils import save_image


def add_contours_to_noise(noisy_images, data_batch, config, device, num_copy=1, translation=False):
    """
    Add (concatenate) contours to the noise channel.

    Parameters:
    noisy_images (torch.Tensor): The noised version of images.
    data_batch (torch.Tensor): The data batch containing corresponding contours.
    config (class): The traning or translating configuration.
    device (str): GPU or CPU.
    num_copy (int): The number of samples to generate in each iteration.
    translation (bool): If the translation is called in the training phrase or translation phrase.

    Returns:
    torch.Tensor: noised images with clean contours concatenated.
    """
    if config.contour_channel_mode == "single":
        if translation:
            contour = data_batch
        else:
            contour = data_batch["contours"]

        if num_copy > 1:
            contour = torch.cat([contour] * num_copy, dim=0)
        contour = contour.to(device)
        noisy_images = torch.cat((noisy_images, contour), dim=1)

    elif config.contour_channel_mode == "multi":
        if translation:
            contour = data_batch
        else:
            contour = data_batch["contours"]
        
        if config.near_guided:
            near_img = data_batch["near_images"]

        if num_copy > 1:
            contour = torch.cat([contour] * num_copy, dim=0)
            near_img = torch.cat([near_img] * num_copy, dim=0)

        contour = contour.to(device)
        near_img = near_img.to(device)
        noisy_images = torch.cat((noisy_images, contour, near_img), dim=1)
    else:
        raise NotImplementedError("Options not implemented!")
    
    return noisy_images

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def normalize_percentile_to_255(data, lower_percentile=0, upper_percentile=100):
    """
    Normalize data based on the specified lower and upper percentiles and scale to [0, 255].

    Parameters:
    data (torch.Tensor): The image data to normalize (either 2D or 3D).
    lower_percentile (int): The lower percentile for clipping.
    upper_percentile (int): The upper percentile for clipping.

    Returns:
    torch.Tensor: Normalized image data scaled to [0, 255].
    """
    # Convert MRI data to a NumPy array if it's a torch Tensor
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Calculate the percentile values
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    # Clip the data
    data_clipped = np.clip(data, lower_bound, upper_bound)

    # Normalize the data to [0, 1] then scale to [0, 255]
    if upper_bound - lower_bound > 0:
        data_normalized = (data_clipped - lower_bound) / (upper_bound - lower_bound)
    else:
        data_normalized = data_clipped
    data_scaled = data_normalized * 255

    # Convert to integer type suitable for image data
    data_scaled = np.round(data_scaled).astype(np.uint8)
 
    return data_scaled

def calculate_Distance(i1, i2):
    """
    Calculate the L2 distance between two images.

    Parameters:
    i1 (np.array): Array for image 1.
    i2 (np.array): Array for image 2.

    Returns:
    float: L2 distance between two image arrays.
    """
    return np.sum((i1-i2)**2) / i1.size

def evaluate(config, epoch, pipeline, noise_step=1000, conditional=False, contour=False, data_batch=None):
    """
    Helper function to call pipeline generation and save the translated images.

    Parameters:
    config (class): The traning or translating configuration.
    epoch (int): At which epoch the helper function is called.
    pipeline (diffusers.DiffusionPipeline): The pipeline for translating the images.
    noise_step (int): The number of denoised steps.
    conditional (bool): Specify if extra conditions are needed for translation.
    contour (bool): Specify if the translation is contour-guided.
    data_batch (Dictionary): Data batch containing original images and corresponding contours.
    """
    # Either generate or translate images,
    # possibly mask guided and/or class conditioned.
    # The default pipeline output type is `List[PIL.Image]`
    
    if contour:
        assert data_batch is not None
        images = pipeline(
            batch_size=config.eval_batch_size,
            num_inference_steps=noise_step,
            generator=torch.Generator().manual_seed(config.seed),
            data_batch=data_batch,
            contour_batch=data_batch
        ).images

    cols = 4
    rows = math.ceil(len(images) / cols)
    image_grid = make_grid(images, rows=rows, cols=cols)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    grid_path = f"{test_dir}/{epoch:04d}.png"
    image_grid.save(grid_path)
    logger.info("Images de validation sauvegardées → {}", grid_path)

    if conditional:
        img_ori = data_batch["images"]
        save_image(img_ori, f"{test_dir}/{epoch:04d}_ori.png", normalize=True, nrow=cols, padding=0)

    if contour:
        img_ori = data_batch["images"]
        contour_ori = data_batch["contours"]
        save_image(img_ori, f"{test_dir}/{epoch:04d}_ori.png", normalize=True, nrow=cols, padding=0)
        save_image(contour_ori, f"{test_dir}/{epoch:04d}_contour.png", normalize=True, nrow=cols, padding=0)

        if config.near_guided:
            near_img_ori = data_batch["near_images"]
            save_image(near_img_ori, f"{test_dir}/{epoch:04d}_near_ori.png", normalize=True, nrow=cols, padding=0)
