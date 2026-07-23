import argparse
import csv
import os
import random
import time
from itertools import cycle

import torch
import torch.optim as optim
import torchvision.utils as vutils

from config import (
    BATCH_SIZE,
    BETA1,
    BETA2,
    CHECKPOINT_DIR,
    DATASET_PATH,
    EPOCHS,
    FALLBACK_BATCH_SIZE,
    FIXED_SAMPLE_COUNT,
    IMAGE_SIZE,
    LAMBDA_GP,
    LATENT_DIM,
    LEARNING_RATE,
    N_CRITIC,
    NUM_WORKERS,
    SAMPLE_DIR,
    TRAINING_LOG_CSV,
)
from models import Critic, Generator, nz, weights_init
from utils import compute_gradient_penalty, get_dataloader

# Defaults sourced from config.py (overridable via CLI).
DEFAULT_NUM_EPOCHS = EPOCHS
DEFAULT_BATCH_SIZE = BATCH_SIZE
DEFAULT_LR = LEARNING_RATE
DEFAULT_NZ = LATENT_DIM
DEFAULT_BETA1 = BETA1
DEFAULT_BETA2 = BETA2
DEFAULT_N_CRITIC = N_CRITIC
DEFAULT_LAMBDA_GP = LAMBDA_GP

TRAINING_SAMPLES_DIR = SAMPLE_DIR

CSV_FIELDNAMES = [
    "epoch",
    "loss_c",
    "loss_g",
    "gradient_penalty",
    "wasserstein_distance",
    "epoch_time_seconds",
    "learning_rate",
    "batch_size",
    "n_critic",
]


def get_training_device():
    """Select CUDA, then MPS, then CPU once at startup."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a WGAN-GP on custom image data.")
    parser.add_argument("--dataroot", type=str, default=DATASET_PATH, help="Path to training images.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Spatial size of training images.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam learning rate for G and Critic.")
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ, help="Latent vector size for the generator input.")
    parser.add_argument("--beta1", type=float, default=DEFAULT_BETA1, help="Adam beta1 hyperparameter.")
    parser.add_argument("--beta2", type=float, default=DEFAULT_BETA2, help="Adam beta2 hyperparameter.")
    parser.add_argument("--n-critic", type=int, default=DEFAULT_N_CRITIC, help="Critic updates per generator update.")
    parser.add_argument("--lambda-gp", type=float, default=DEFAULT_LAMBDA_GP, help="Gradient penalty coefficient λ.")
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
        "betas": (args.beta1, args.beta2),
        "nz": args.nz,
        "n_critic": args.n_critic,
        "lambda_gp": args.lambda_gp,
        "model": "WGAN-GP",
    }


def save_checkpoint(model, optimizer, epoch, config, path, history=None):
    """
    Save a checkpoint containing model weights, optimizer state, epoch number,
    training config, and optional training history for resume/inspection.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "history": history if history is not None else [],
    }
    torch.save(payload, path)


def critic_checkpoint_path(generator_checkpoint_path):
    """Map a generator checkpoint path to its paired critic checkpoint."""
    return generator_checkpoint_path.replace("generator", "critic")


def load_checkpoint(model, optimizer, path, device):
    """Restore model and optimizer state from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", [])
    return checkpoint["epoch"], history


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
    if args.n_critic < 1:
        raise ValueError("--n-critic must be >= 1.")
    if args.lambda_gp < 0:
        raise ValueError("--lambda-gp must be >= 0.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_training_device()
    print(f"Using device: {device}")

    if args.image_size != 256:
        raise ValueError("The model architecture is fixed at 256x256. Set --image-size 256.")

    batch_size = args.batch_size
    dataloader = get_dataloader(
        dataroot=args.dataroot,
        image_size=args.image_size,
        batch_size=batch_size,
        workers=args.workers,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"No training batches available. Dataset may be smaller than batch size "
            f"({batch_size}) with drop_last enabled."
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TRAINING_SAMPLES_DIR, exist_ok=True)

    training_config = build_training_config(args)

    netG = Generator().to(device)
    netC = Critic().to(device)

    def try_training_batch(current_batch_size):
        real = torch.randn(current_batch_size, 3, args.image_size, args.image_size, device=device)
        noise = torch.randn(current_batch_size, args.nz, 1, 1, device=device)
        fake = netG(noise)
        netC(real)
        netC(fake)
        # Smoke-test gradient penalty path as well.
        _ = compute_gradient_penalty(netC, real, fake.detach(), device, lambda_gp=args.lambda_gp)
        return fake.shape

    try:
        fake_shape = try_training_batch(batch_size)
    except RuntimeError as exc:
        if batch_size != FALLBACK_BATCH_SIZE and "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = FALLBACK_BATCH_SIZE
            training_config["batch_size"] = batch_size
            print(f"Batch size {args.batch_size} exceeded memory; falling back to {batch_size}.")
            dataloader = get_dataloader(
                dataroot=args.dataroot,
                image_size=args.image_size,
                batch_size=batch_size,
                workers=args.workers,
            )
            if len(dataloader) == 0:
                raise ValueError(
                    f"No training batches available. Dataset may be smaller than batch size "
                    f"({batch_size}) with drop_last enabled."
                ) from exc
            fake_shape = try_training_batch(batch_size)
        else:
            raise

    if fake_shape[-1] != 256 or fake_shape[-2] != 256:
        raise RuntimeError(f"Generator output must be 256x256, got {tuple(fake_shape)}")

    # WGAN-GP: Adam with betas=(0, 0.9) for both networks.
    optimizerC = optim.Adam(netC.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_epoch = 0
    history = []
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        critic_path = critic_checkpoint_path(args.resume)
        if not os.path.isfile(critic_path):
            raise FileNotFoundError(f"Paired critic checkpoint not found: {critic_path}")

        completed_epoch, history = load_checkpoint(netG, optimizerG, args.resume, device)
        _, critic_history = load_checkpoint(netC, optimizerC, critic_path, device)
        if not history and critic_history:
            history = critic_history
        start_epoch = completed_epoch
        print(f"Resumed from epoch {completed_epoch}")
    else:
        netG.apply(weights_init)
        netC.apply(weights_init)

    # Seeded once before training and reused unchanged for every epoch's sample grid.
    fixed_noise = torch.randn(FIXED_SAMPLE_COUNT, args.nz, 1, 1, device=device)

    # Infinite cycling iterator so each critic step can pull a fresh real batch.
    data_iter = cycle(dataloader)
    generator_steps_per_epoch = max(1, len(dataloader) // args.n_critic)

    csv_file = None
    print(
        f"Starting WGAN-GP training "
        f"(n_critic={args.n_critic}, lambda_gp={args.lambda_gp}, lr={args.lr})..."
    )

    try:
        csv_file, csv_writer = open_training_log(resume=bool(args.resume))

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            epoch_number = epoch + 1

            epoch_loss_c = 0.0
            epoch_loss_g = 0.0
            epoch_gp = 0.0
            epoch_w_dist = 0.0
            num_g_steps = 0

            for _ in range(generator_steps_per_epoch):
                ############################################################
                # (1) Update Critic n_critic times.
                #     Loss_C = E[C(fake)] - E[C(real)] + λ * GP
                ############################################################
                for _critic_step in range(args.n_critic):
                    netC.zero_grad(set_to_none=True)

                    real_cpu, _ = next(data_iter)
                    real_cpu = real_cpu.to(device)
                    current_batch = real_cpu.size(0)

                    noise = torch.randn(current_batch, args.nz, 1, 1, device=device)
                    fake = netG(noise).detach()

                    critic_real = netC(real_cpu).view(-1)
                    critic_fake = netC(fake).view(-1)

                    # Wasserstein critic objective (without GP).
                    loss_c_wasserstein = critic_fake.mean() - critic_real.mean()
                    # Soft constraint that enforces 1-Lipschitz via gradient penalty.
                    gp = compute_gradient_penalty(
                        netC,
                        real_cpu,
                        fake,
                        device,
                        lambda_gp=args.lambda_gp,
                    )
                    loss_c = loss_c_wasserstein + gp
                    loss_c.backward()
                    optimizerC.step()

                    # Estimated Wasserstein distance ≈ E[C(real)] - E[C(fake)].
                    w_distance = (critic_real.mean() - critic_fake.mean()).detach()

                    epoch_loss_c += loss_c.item()
                    epoch_gp += gp.item()
                    epoch_w_dist += w_distance.item()

                ############################################################
                # (2) Update Generator once.
                #     Loss_G = -E[C(fake)]  (maximize critic score on fakes)
                ############################################################
                netG.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                fake = netG(noise)
                loss_g = -netC(fake).view(-1).mean()
                loss_g.backward()
                optimizerG.step()

                epoch_loss_g += loss_g.item()
                num_g_steps += 1

            epoch_time = time.time() - epoch_start
            num_critic_steps = num_g_steps * args.n_critic
            avg_loss_c = epoch_loss_c / num_critic_steps
            avg_loss_g = epoch_loss_g / num_g_steps
            avg_gp = epoch_gp / num_critic_steps
            avg_w_dist = epoch_w_dist / num_critic_steps

            epoch_record = {
                "epoch": epoch_number,
                "loss_c": avg_loss_c,
                "loss_g": avg_loss_g,
                "gradient_penalty": avg_gp,
                "wasserstein_distance": avg_w_dist,
                "epoch_time_seconds": epoch_time,
                "learning_rate": args.lr,
                "batch_size": batch_size,
                "n_critic": args.n_critic,
            }
            history.append(epoch_record)

            # Numbered epoch checkpoints are never overwritten; only *_latest.pth is refreshed.
            save_checkpoint(
                netG,
                optimizerG,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch_number:03d}.pth"),
                history=history,
            )
            save_checkpoint(
                netC,
                optimizerC,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, f"critic_epoch_{epoch_number:03d}.pth"),
                history=history,
            )
            save_checkpoint(
                netG,
                optimizerG,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, "generator_latest.pth"),
                history=history,
            )
            save_checkpoint(
                netC,
                optimizerC,
                epoch_number,
                training_config,
                os.path.join(CHECKPOINT_DIR, "critic_latest.pth"),
                history=history,
            )

            save_epoch_samples(netG, fixed_noise, epoch_number, TRAINING_SAMPLES_DIR)

            csv_writer.writerow(epoch_record)
            csv_file.flush()

            print(
                f"[{epoch_number}/{args.epochs}] "
                f"Loss_C: {avg_loss_c:.4f} Loss_G: {avg_loss_g:.4f} "
                f"GP: {avg_gp:.4f} W_dist: {avg_w_dist:.4f} "
                f"Time: {epoch_time:.1f}s"
            )

    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"Wrote training log to {TRAINING_LOG_CSV}")


if __name__ == "__main__":
    train()
