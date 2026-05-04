from dataclasses import dataclass
from torchvision import transforms


@dataclass
class TrainingConfig:
    model_type: str = "ddpm"
    dataset: str = None
    input_domain: str = None
    output_domain: str = None
    img_size: int = 256
    in_channels: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 16
    num_epochs: int = 600
    gradient_accumulation_steps: int = 1
    noise_step: int = 1000
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 20
    mixed_precision: str = 'fp16'
    output_dir: str = None

    seed: int = 0
    workers: int = 0
    device: str = 'cuda:0'

    # Augmentation — used by TrainTransform in transform.py
    degrees: float = 5.0              # max in-plane rotation (degrees)
    translate: tuple = (0.1, 0.1)    # max translation as fraction of img_size
    scale: tuple = (0.9, 1.1)        # isotropic zoom range
    flip_p: float = 0.5              # left-right flip probability
    apply_p: float = 0.9             # probability to apply affine augmentation

    # Guidance
    contour_guided: bool = True
    contour_channel_mode: str = "multi"
    conditional: bool = False
    near_guided: bool = True
    near_guided_ratio: float = 0.2


@dataclass
class TranslatingConfig:
    model_type: str = "ddim"
    dataset: str = None
    input_domain: str = None
    output_domain: str = None
    eval_batch_size: int = 1
    img_size: int = 256
    denoise_step: int = 50
    training_noise_step: int = 1000
    selected_epoch: int = 1
    in_channels: int = 1
    output_dir: str = None

    img_interpolation = transforms.InterpolationMode.BICUBIC
    contour_interpolation = transforms.InterpolationMode.NEAREST

    seed: int = 0
    workers: int = 8
    device: str = 'cuda:0'

    by_volume: bool = False
    contour_channel_mode: str = "single"
