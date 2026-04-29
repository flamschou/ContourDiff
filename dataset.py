import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset


def _load_mat(path):
    """Load a 3D .mat volume → float32 (Z, H, W).

    MATLAB stores volumes as (H, W, Z); np.moveaxis converts to (Z, H, W).
    Supports both scipy (MATLAB < v7.3) and h5py (MATLAB v7.3+ / HDF5).
    """
    path = str(path)
    arr = None
    try:
        import scipy.io
        data = scipy.io.loadmat(path)
        for k, v in data.items():
            if not k.startswith('_') and hasattr(v, 'ndim') and v.ndim >= 2:
                arr = np.array(v).astype(np.float32)
                break
    except Exception:
        pass
    if arr is None:
        import h5py
        with h5py.File(path, 'r') as f:
            key = next(iter(f.keys()))
            arr = np.array(f[key]).astype(np.float32)
    if arr.ndim == 3:
        arr = np.moveaxis(arr, -1, 0)  # MATLAB (H_S/I, W_L/R, Z_A/P) → (A/P, S/I, L/R) coronal
    return arr


def _resize_hw(vol, th, tw, order):
    """Resize spatial axes H,W to (th, tw); Z unchanged."""
    from scipy.ndimage import zoom as _zoom
    _, h, w = vol.shape
    return _zoom(vol, (1.0, th / h, tw / w), order=order)


class MatIRMDataset(Dataset):
    """MRI dataset backed by .mat files.

    Volumes are loaded and resized to (Z, img_size, img_size) once in __init__.
    Augmentation is 3D-coherent: one set of random parameters per volume,
    identical for all its Z slices.

    Call resample() at the start of each training epoch to get fresh augmentations.
    Requires num_workers=0 in the DataLoader (default) so the main process sees
    the updated aug_volumes immediately.
    """

    def __init__(self, mat_pairs, train_transform=None, val_transform=None,
                 train=True, config=None):
        assert config is not None, "config is required"
        self.config = config
        self.train = train
        self.train_transform = train_transform
        self.val_transform = val_transform
        img_size = config.img_size

        self.volumes = []
        self.vol_bounds = []    # (p1, p99) computed on raw data before any resize/aug
        self.contours_data = []
        self.samples = []       # (vol_idx, z_idx)

        split_label = "train" if train else "val"
        pbar = tqdm(mat_pairs, desc=f"Chargement {split_label}", unit="vol", dynamic_ncols=True)
        for vol_idx, (vol_path, cont_path) in enumerate(pbar):
            pbar.set_postfix({"fichier": Path(vol_path).stem[-30:]})
            vol = _load_mat(vol_path)
            cont = _load_mat(cont_path)
            assert vol.shape == cont.shape, (
                f"Shape mismatch: {vol_path} {vol.shape} vs {cont_path} {cont.shape}"
            )
            # Bounds on raw intensities — stable reference across epochs
            self.vol_bounds.append((float(np.percentile(vol, 1)),
                                    float(np.percentile(vol, 99))))
            # One-time resize to target spatial resolution
            self.volumes.append(_resize_hw(vol, img_size, img_size, order=1))
            self.contours_data.append(_resize_hw(cont, img_size, img_size, order=1))
            for z in range(vol.shape[0]):
                self.samples.append((vol_idx, z))

        self.aug_volumes = None
        self.aug_contours = None
        self.resample()

    # ── augmentation ─────────────────────────────────────────────────────────

    def resample(self):
        """Re-draw 3D augmentations. Call once per training epoch."""
        transform = self.train_transform if self.train else self.val_transform
        aug_vols, aug_conts = [], []
        for vol, cont in zip(self.volumes, self.contours_data):
            if transform is not None:
                vol_aug, cont_aug = transform(vol, cont)
            else:
                vol_aug, cont_aug = vol, cont
            aug_vols.append(vol_aug)
            aug_conts.append(cont_aug)
        self.aug_volumes = aug_vols
        self.aug_contours = aug_conts

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        vol_idx, z_idx = self.samples[index]
        p1, p99 = self.vol_bounds[vol_idx]

        # Image slice → float32 tensor (1, H, W) in [-1, 1]
        raw = self.aug_volumes[vol_idx][z_idx]
        img = np.clip(raw, p1, p99)
        if p99 > p1:
            img = (img - p1) / (p99 - p1) * 2.0 - 1.0
        img_tensor = torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0)

        # Contour slice → float32 tensor (1, H, W) in [-1, 1]
        cont_raw = self.aug_contours[vol_idx][z_idx]
        cont = cont_raw * 2.0 - 1.0
        cont_tensor = torch.tensor(cont.copy(), dtype=torch.float32).unsqueeze(0)

        near_tensor = self._near_tensor(vol_idx, z_idx, p1, p99)

        return {
            "images": img_tensor,
            "contours": cont_tensor,
            "near_images": near_tensor,
            "image_name": f"vol{vol_idx}_z{z_idx}",
            "contour_name": f"cont{vol_idx}_z{z_idx}",
        }

    def _near_tensor(self, vol_idx, z_idx, p1, p99):
        img_size = self.config.img_size
        zero = torch.zeros(1, img_size, img_size, dtype=torch.float32)
        if not getattr(self.config, 'near_guided', False):
            return zero
        if random.random() >= getattr(self.config, 'near_guided_ratio', 0.2):
            return zero
        vol = self.aug_volumes[vol_idx]
        z_near = None
        if random.random() < 0.5:
            if z_idx - 1 >= 0:
                z_near = z_idx - 1
            elif z_idx + 1 < vol.shape[0]:
                z_near = z_idx + 1
        else:
            if z_idx + 1 < vol.shape[0]:
                z_near = z_idx + 1
            elif z_idx - 1 >= 0:
                z_near = z_idx - 1
        if z_near is None:
            return zero
        s = np.clip(vol[z_near], p1, p99)
        if p99 > p1:
            s = (s - p1) / (p99 - p1) * 2.0 - 1.0
        return torch.tensor(s.copy(), dtype=torch.float32).unsqueeze(0)


class SegmentationDataset(Dataset):
    def __init__(self, df_meta, image_directory, mask_directory=None, class_specifier=None,
                 generator_seed=None, transform_img=None, train_pre_transform_img=None,
                 transform_mask=None):
        self.df_meta = df_meta
        self.image_directory = image_directory
        self.mask_directory = mask_directory if mask_directory else image_directory
        self.length = len(self.df_meta)
        self.class_specifier = class_specifier
        self.generator_seed = generator_seed
        if generator_seed is not None:
            self.seed_generator = torch.Generator().manual_seed(generator_seed)
        self.transform_img = transform_img
        self.train_pre_transform_img = train_pre_transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        import os
        from PIL import Image
        img_name = self.df_meta.iloc[index]["image_name"]
        img = Image.open(os.path.join(self.image_directory, img_name)).convert("RGB")
        mask_name = self.df_meta.iloc[index]["mask_name"]
        mask = Image.open(os.path.join(self.mask_directory, mask_name))
        np_mask = np.array(mask)
        if self.class_specifier is not None:
            np_mask = np.isin(np_mask, self.class_specifier).astype("uint8") * 255
        mask = Image.fromarray(np_mask)
        if self.train_pre_transform_img is not None:
            img = self.train_pre_transform_img(img)
        if self.generator_seed is not None:
            seed = self.seed_generator.seed()
        if self.transform_img is not None:
            if self.generator_seed is not None:
                torch.manual_seed(seed)
                torch.random.manual_seed(seed)
                random.seed(seed)
            img = self.transform_img(img)
        if self.transform_mask is not None:
            if self.generator_seed is not None:
                torch.manual_seed(seed)
                torch.random.manual_seed(seed)
                random.seed(seed)
            mask = self.transform_mask(mask)
        return {"image": img, "mask": mask, "image_name": img_name, "mask_name": img_name}
