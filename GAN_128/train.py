import argparse
import csv
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from models import Discriminator, Generator, nz, weights_init
from utils import get_dataloader

# Default training hyperparameters (overridable via CLI).
DEFAULT_NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.0002
DEFAULT_NZ = nz
DEFAULT_BETA1 = 0.5
DEFAULT_BETA2 = 0.999

CHECKPOINT_DIR = "checkpoints"
TRAINING_SAMPLES_DIR = "training_samples"
TRAINING_LOG_CSV = "training_log.csv"
FIXED_SAMPLE_COUNT = 16

CSV_FIELDNAMES = [
    "epoch",
    "loss_d",
    "loss_g",
    "d_real",
    "d_fake_before_update",
    "d_fake_after_update",
    "epoch_time_seconds",
    "learning_rate",
    "batch_size",
]


def get_training_device():
    """Select CUDA, then MPS, then CPU once at startup."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DCGAN on custom image data.")
    parser.add_argument("--dataroot", type=str, default="data/train", help="Path to training images.")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=128, help="Spatial size of training images.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for Adam.")
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ, help="Latent vector size for the generator input.")
    parser.add_argument("--beta1", type=float, default=DEFAULT_BETA1, help="Adam beta1 hyperparameter.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a generator checkpoint (.pth) to resume training from.",
    )
    parser.add_argument("--seed", type=int, default=999, help="Random seed.")
    return parser.parse_args()


def build_training_config(args):
    """Hyperparameters stored in every checkpoint for inspection and resume."""
    return {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "betas": (args.beta1, DEFAULT_BETA2),
        "nz": args.nz,
    }


def save_checkpoint(model, optimizer, epoch, config, path):
    """
    Save a checkpoint containing model weights, optimizer state, epoch number,
    and the training config dict so a run can be inspected or resumed later.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def discriminator_checkpoint_path(generator_checkpoint_path):
    """Map a generator checkpoint path to its paired discriminator checkpoint."""
    return generator_checkpoint_path.replace("generator", "discriminator")


def load_checkpoint(model, optimizer, path, device):
    """Restore model and optimizer state from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def save_epoch_samples(netG, fixed_noise, epoch, samples_dir):
    """
    Save a fixed-noise sample grid each epoch. The same fixed_noise tensor is
    reused for every epoch so visual progress is directly comparable.
    """
    os.makedirs(samples_dir, exist_ok=True)
    output_path = os.path.join(samples_dir, f"epoch_{epoch:03d}.png")

    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_noise).detach()
    vutils.save_image(
        fake,
        output_path,
        nrow=4,
        normalize=True,
    )
    netG.train()


def open_training_log(resume):
    """
    Open the CSV log for writing.

    New runs overwrite any existing file and write a fresh header.
    Resume runs append to the existing file without rewriting the header.
    """
    if resume:
        csv_file = open(TRAINING_LOG_CSV, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        return csv_file, writer

    csv_file = open(TRAINING_LOG_CSV, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    csv_file.flush()
    return csv_file, writer


def train():
    args = parse_args()

    if args.nz != nz:
        raise ValueError(
            f"--nz {args.nz} does not match the Generator input size defined in models.py ({nz}). "
            "Change models.py to alter the architecture."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_training_device()
    print(f"Using device: {device}")

    if args.image_size != 128:
        raise ValueError("The model architecture is fixed at 128x128. Set --image-size 128.")

    dataloader = get_dataloader(
        dataroot=args.dataroot,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"No training batches available. Dataset may be smaller than batch size "
            f"({args.batch_size}) with drop_last enabled."
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TRAINING_SAMPLES_DIR, exist_ok=True)

    training_config = build_training_config(args)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, DEFAULT_BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, DEFAULT_BETA2))

    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        discriminator_path = discriminator_checkpoint_path(args.resume)
        if not os.path.isfile(discriminator_path):
            raise FileNotFoundError(
                f"Paired discriminator checkpoint not found: {discriminator_path}"
            )

        completed_epoch = load_checkpoint(netG, optimizerG, args.resume, device)
        load_checkpoint(netD, optimizerD, discriminator_path, device)
        start_epoch = completed_epoch
        print(f"Resumed from epoch {completed_epoch}")
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)

    criterion = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0

    # Seeded once before training and reused unchanged for every epoch's sample grid.
    fixed_noise = torch.randn(FIXED_SAMPLE_COUNT, args.nz, 1, 1, device=device)

    csv_file = None
    print("Starting training...")

    try:
        csv_file, csv_writer = open_training_log(resume=bool(args.resume))

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            epoch_number = epoch + 1

            epoch_loss_d = 0.0
            epoch_loss_g = 0.0
            epoch_d_real = 0.0
            epoch_d_fake_before = 0.0
            epoch_d_fake_after = 0.0
            num_batches = 0

            for real_cpu, _ in dataloader:
                ############################
                # (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                netD.zero_grad()
                real_cpu = real_cpu.to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                d_x = output.mean().item()

                noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                # D(G(z)) before G is updated — discriminator's view of fakes.
                d_fake_before = output.mean().item()

                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                # D(G(z)) after G is updated — same fake batch, generator's loss forward pass.
                d_fake_after = output.mean().item()
                optimizerG.step()

                epoch_loss_d += errD.item()
                epoch_loss_g += errG.item()
                epoch_d_real += d_x
                epoch_d_fake_before += d_fake_before
                epoch_d_fake_after += d_fake_after
                num_batches += 1

            epoch_time = time.time() - epoch_start
            avg_loss_d = epoch_loss_d / num_batches
            avg_loss_g = epoch_loss_g / num_batches
            avg_d_real = epoch_d_real / num_batches
            avg_d_fake_before = epoch_d_fake_before / num_batches
            avg_d_fake_after = epoch_d_fake_after / num_batches

            # Numbered epoch checkpoints are never overwritten; only *_latest.pth is refreshed.
            save_checkpoint(
                netG,
                optimizerG,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch_number:03d}.pth"),
            )
            save_checkpoint(
                netD,
                optimizerD,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch_number:03d}.pth"),
            )
            save_checkpoint(
                netG,
                optimizerG,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, "generator_latest.pth"),
            )
            save_checkpoint(
                netD,
                optimizerD,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, "discriminator_latest.pth"),
            )

            save_epoch_samples(netG, fixed_noise, epoch_number, TRAINING_SAMPLES_DIR)

            csv_writer.writerow(
                {
                    "epoch": epoch_number,
                    "loss_d": avg_loss_d,
                    "loss_g": avg_loss_g,
                    "d_real": avg_d_real,
                    "d_fake_before_update": avg_d_fake_before,
                    "d_fake_after_update": avg_d_fake_after,
                    "epoch_time_seconds": epoch_time,
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                }
            )
            csv_file.flush()

            print(
                f"[{epoch_number}/{args.epochs}] "
                f"Loss_D: {avg_loss_d:.4f} Loss_G: {avg_loss_g:.4f} "
                f"D(x): {avg_d_real:.4f} "
                f"D(G(z)): {avg_d_fake_before:.4f} / {avg_d_fake_after:.4f} "
                f"Time: {epoch_time:.1f}s"
            )

    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"Wrote training log to {TRAINING_LOG_CSV}")


if __name__ == "__main__":
    train()
