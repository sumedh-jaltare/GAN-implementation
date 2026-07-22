import argparse
import os

import torch

from models import Generator, nz
from utils import get_device, save_generated_images


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic images with a trained DCGAN generator.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="generator.pth",
        help="Path to the trained generator checkpoint.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where generated PNG files will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used during generation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser.parse_args()


def load_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    else:
        raise KeyError(
            f"Invalid checkpoint: missing 'model_state_dict' or 'generator_state_dict' in {checkpoint_path}"
        )
    netG = Generator().to(device)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG


def generate():
    args = parse_args()

    if args.num_images <= 0:
        raise ValueError("--num-images must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    netG = load_generator(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    generated = 0
    with torch.no_grad():
        while generated < args.num_images:
            current_batch = min(args.batch_size, args.num_images - generated)
            noise = torch.randn(current_batch, nz, 1, 1, device=device)
            fake = netG(noise)
            save_generated_images(fake, args.output_dir, start_index=generated)
            generated += current_batch

    print(f"Saved {args.num_images} images to {args.output_dir}/")


if __name__ == "__main__":
    generate()
