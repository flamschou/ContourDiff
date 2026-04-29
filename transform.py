import numpy as np
from scipy.ndimage import rotate as _nd_rotate, zoom as _nd_zoom, shift as _nd_shift


def _center_crop_or_pad(vol, th, tw):
    """Crop or zero-pad vol (Z, H, W) back to (Z, th, tw), centred."""
    z, h, w = vol.shape
    if h > th:
        s = (h - th) // 2
        vol = vol[:, s:s + th, :]
    elif h < th:
        p = th - h
        vol = np.pad(vol, ((0, 0), (p // 2, p - p // 2), (0, 0)))
    z, h, w = vol.shape
    if w > tw:
        s = (w - tw) // 2
        vol = vol[:, :, s:s + tw]
    elif w < tw:
        p = tw - w
        vol = np.pad(vol, ((0, 0), (0, 0), (p // 2, p - p // 2)))
    return vol


class TrainTransform:
    """3D-coherent augmentation for (Z, H, W) volumes.

    One set of random parameters is drawn once per volume.
    Every Z slice receives the identical flip / rotation / zoom / translate,
    so the augmented volume remains anatomically self-consistent.

    Interpolation order:
      - image  → order=1 (bilinear) to preserve soft intensities
      - contour → order=1 (bilinear) to preserve continuous values
    """

    def __init__(self, config):
        self.degrees = config.degrees
        self.scale = config.scale          # e.g. (0.9, 1.1)
        self.translate = config.translate  # fraction of img_size, e.g. (0.1, 0.1)
        self.flip_p = config.flip_p
        self.apply_p = config.apply_p
        self.img_size = config.img_size

    def __call__(self, image, contour):
        """
        image, contour : float32 numpy (Z, img_size, img_size)
        Returns augmented copies of the same shape.
        """
        # Left-right flip — anatomically valid for paired lungs
        if np.random.random() < self.flip_p:
            image = image[:, :, ::-1].copy()
            contour = contour[:, :, ::-1].copy()

        if np.random.random() < self.apply_p:
            # In-plane rotation (axes 1,2 = H,W): same angle for every Z slice
            angle = np.random.uniform(-self.degrees, self.degrees)
            image = _nd_rotate(image, angle, axes=(1, 2), reshape=False, order=1, cval=0.0)
            contour = _nd_rotate(contour, angle, axes=(1, 2), reshape=False, order=1, cval=0.0)

            # Isotropic zoom in H,W; crop/pad back to img_size
            scale = np.random.uniform(self.scale[0], self.scale[1])
            if abs(scale - 1.0) > 0.01:
                image = _center_crop_or_pad(
                    _nd_zoom(image, (1.0, scale, scale), order=1),
                    self.img_size, self.img_size,
                )
                contour = _center_crop_or_pad(
                    _nd_zoom(contour, (1.0, scale, scale), order=1),
                    self.img_size, self.img_size,
                )

            # In-plane translation (same shift for every Z slice)
            dy = int(np.random.uniform(-self.translate[0], self.translate[0]) * self.img_size)
            dx = int(np.random.uniform(-self.translate[1], self.translate[1]) * self.img_size)
            if dy != 0 or dx != 0:
                image = _nd_shift(image, (0, dy, dx), order=1, cval=0.0)
                contour = _nd_shift(contour, (0, dy, dx), order=1, cval=0.0)

        return image.copy(), contour.copy()


class ValTransform:
    """Validation: no augmentation (resize already handled in MatIRMDataset.__init__)."""

    def __call__(self, image, contour):
        return image, contour
