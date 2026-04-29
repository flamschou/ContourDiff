import os
import torch
import argparse
from pathlib import Path

from loguru import logger
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from config import TrainingConfig
from transform import TrainTransform, ValTransform
from dataset import MatIRMDataset
from ContourDiffPipeline import ContourDiffDDPMPipeline, ContourDiffDDIMPipeline
from utils import evaluate, add_contours_to_noise
from monitor import TrainingMonitor


def find_mat_pairs(directory):
    """Return sorted list of (zscore_path, contours_zscore_path) pairs."""
    pairs = []
    for vol_path in sorted(Path(directory).rglob("zscore_*.mat")):
        cont_path = vol_path.parent / f"contours_{vol_path.stem}.mat"
        if cont_path.exists():
            pairs.append((str(vol_path), str(cont_path)))
        else:
            logger.warning("no contour for {} — skipping", vol_path.name)
    return pairs


def main(args):
    config = TrainingConfig(
        model_type=args.model_type,
        dataset=args.dataset,
        img_size=args.img_size,
        input_domain=args.input_domain,
        output_domain=args.output_domain,
        in_channels=args.in_channels,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        noise_step=args.noise_step,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        save_image_epochs=args.save_image_epochs,
        save_model_epochs=args.save_model_epochs,
        seed=args.seed,
        workers=args.workers,
        contour_guided=args.contour_guided,
        contour_channel_mode=args.contour_channel_mode,
        conditional=args.conditional,
        near_guided=args.near_guided,
        near_guided_ratio=args.near_guided_ratio,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device : {}", device)

    config.output_dir = args.output_dir or (
        f"ContourDiff-{config.input_domain}-{config.output_domain}"
        f"-{config.model_type}-{config.dataset}"
    )
    os.makedirs(config.output_dir, exist_ok=True)
    logger.add(
        os.path.join(config.output_dir, "training.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
        rotation="50 MB",
        encoding="utf-8",
    )
    logger.info("Output : {}", config.output_dir)

    # ── Discover .mat pairs ────────────────────────────────────────────────
    all_pairs = find_mat_pairs(args.mat_directory)
    if not all_pairs:
        raise RuntimeError(f"No pairs found under {args.mat_directory}")

    if args.val_mat_directory:
        train_pairs = all_pairs
        val_pairs = find_mat_pairs(args.val_mat_directory)
        if not val_pairs:
            raise RuntimeError(f"No pairs found under {args.val_mat_directory}")
    else:
        n_val = max(1, round(len(all_pairs) * args.val_ratio)) if args.val_ratio > 0 else 0
        val_pairs  = all_pairs[-n_val:] if n_val else all_pairs
        train_pairs = all_pairs[:-n_val] if n_val else all_pairs

    logger.info("Volumes  — train: {}  val: {}", len(train_pairs), len(val_pairs))

    # ── Transforms ────────────────────────────────────────────────────────
    train_transform = TrainTransform(config)
    val_transform   = ValTransform()

    # ── Datasets & dataloaders ────────────────────────────────────────────
    logger.info("Chargement des volumes (resize → {0}×{0})…", config.img_size)
    train_dataset = MatIRMDataset(
        train_pairs, train_transform=train_transform, val_transform=val_transform,
        train=True, config=config,
    )
    val_dataset = MatIRMDataset(
        val_pairs, train_transform=train_transform, val_transform=val_transform,
        train=False, config=config,
    )
    logger.info(
        "Slices   — train: {}  val: {}  (chaque slice = 1 coupe coronale 2D d'un volume 3D)",
        len(train_dataset), len(val_dataset),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.eval_batch_size,
        shuffle=True, num_workers=config.workers,
    )

    # ── Noise scheduler & model ────────────────────────────────────────────
    if config.model_type == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=config.noise_step)
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=config.noise_step)

    if config.contour_guided:
        if config.contour_channel_mode == "single":
            model_in_channels = config.in_channels + 1
        elif config.contour_channel_mode == "multi" and config.near_guided:
            model_in_channels = config.in_channels + 2
        else:
            raise NotImplementedError("contour_channel_mode not implemented")
    else:
        model_in_channels = config.in_channels

    logger.info(
        "Modèle UNet2D — in_channels: {}  out_channels: {}  img_size: {}",
        model_in_channels, config.in_channels, config.img_size,
    )
    model = UNet2DModel(
        sample_size=config.img_size,
        in_channels=model_in_channels,
        out_channels=config.in_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "Optimiseur AdamW  lr={:.1e}  warmup={} steps  total={} steps",
        config.learning_rate, config.lr_warmup_steps, total_steps,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    monitor = TrainingMonitor(config.output_dir, config)
    logger.info("Monitor HTML : {}/monitor.html", config.output_dir)

    global_step = 0
    epoch_bar = tqdm(
        range(config.num_epochs),
        desc="Training",
        unit="epoch",
        position=0,
    )

    for epoch in epoch_bar:
        # Re-draw 3D augmentations for this epoch (num_workers=0 required)
        train_dataset.resample()
        model.train()

        epoch_loss_sum = 0.0
        step_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            unit="step",
            position=1,
            leave=False,
        )

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            noise        = torch.randn_like(clean_images)
            bs           = clean_images.shape[0]
            timesteps    = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bs,), device=device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config.contour_guided:
                noisy_images = add_contours_to_noise(noisy_images, batch, config, device)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss       = F.mse_loss(noise_pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            current_loss    = loss.detach().item()
            current_lr      = lr_scheduler.get_last_lr()[0]
            epoch_loss_sum += current_loss
            running_avg     = epoch_loss_sum / (step + 1)

            monitor.step(current_loss, current_lr)

            step_bar.update(1)
            step_bar.set_postfix(
                loss=f"{current_loss:.5f}",
                avg=f"{running_avg:.5f}",
                lr=f"{current_lr:.2e}",
                step=global_step,
            )
            global_step += 1

        step_bar.close()
        epoch_avg = epoch_loss_sum / len(train_dataloader)
        epoch_bar.set_postfix(train=f"{epoch_avg:.5f}", val="…", lr=f"{current_lr:.2e}")

        # ── Validation loss ────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                clean_images = val_batch["images"].to(device)
                noise        = torch.randn_like(clean_images)
                bs           = clean_images.shape[0]
                timesteps    = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bs,), device=device,
                ).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                if config.contour_guided:
                    noisy_images = add_contours_to_noise(noisy_images, val_batch, config, device)
                noise_pred    = model(noisy_images, timesteps, return_dict=False)[0]
                val_loss_sum += F.mse_loss(noise_pred, noise).item()
        val_avg = val_loss_sum / len(val_dataloader)
        logger.info(
            "Epoch {}/{}  train_avg={:.5f}  val_avg={:.5f}",
            epoch + 1, config.num_epochs, epoch_avg, val_avg,
        )
        epoch_bar.set_postfix(train=f"{epoch_avg:.5f}", val=f"{val_avg:.5f}", lr=f"{current_lr:.2e}")
        monitor.end_epoch(epoch + 1, val_avg=val_avg)

        # ── Image generation & checkpointing ──────────────────────────────
        if config.model_type == "ddpm":
            pipeline = ContourDiffDDPMPipeline(
                unet=model.module, scheduler=noise_scheduler,
                data_loader=val_dataloader, external_config=config,
            )
        else:
            pipeline = ContourDiffDDIMPipeline(
                unet=model.module, scheduler=noise_scheduler,
                data_loader=val_dataloader, external_config=config,
            )

        save_img = (epoch == 0) or (epoch + 1) % config.save_image_epochs == 0 \
                   or epoch == config.num_epochs - 1
        if save_img and config.contour_guided:
            logger.info("Génération d'images de validation (epoch {})…", epoch + 1)
            data_batch = next(iter(val_dataloader))
            evaluate(config, epoch + 1, pipeline, noise_step=config.noise_step,
                     contour=True, data_batch=data_batch)

        save_model = (epoch == 0) or (epoch + 1) % config.save_model_epochs == 0 \
                     or epoch == config.num_epochs - 1
        if save_model:
            tag = "model" if args.overwrite else f"model_epoch_{epoch + 1}"
            save_path = os.path.join(config.output_dir, tag)
            pipeline.save_pretrained(save_path)
            logger.info("Modèle sauvegardé → {}", save_path)

        monitor.end_epoch(epoch + 1)

    logger.success("Entraînement terminé — {} epochs, {} steps", config.num_epochs, global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument('--mat_directory', type=str, required=True,
                        help="directory containing zscore_*.mat + contours_zscore_*.mat")
    parser.add_argument('--val_mat_directory', type=str, default=None,
                        help="separate val directory (omit to split from mat_directory)")
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="fraction of volumes for val (0.0 = same volumes as train)")

    # ── Domain labels ──────────────────────────────────────────────────────
    parser.add_argument('--input_domain', type=str, default="IRM")
    parser.add_argument('--output_domain', type=str, default="IRM")

    # ── Training ───────────────────────────────────────────────────────────
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--noise_step', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument('--save_image_epochs', type=int, default=20)
    parser.add_argument('--save_model_epochs', type=int, default=20)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0,
                        help="must be 0 — resample() requires single-process data loading")
    parser.add_argument('--in_channels', type=int, default=1)

    # ── Contour / near guidance ────────────────────────────────────────────
    parser.add_argument('--contour_guided', action='store_true')
    parser.add_argument('--contour_channel_mode', type=str, default="single")
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--near_guided', action='store_true')
    parser.add_argument('--near_guided_ratio', type=float, default=0.2)

    args = parser.parse_args()
    main(args)
