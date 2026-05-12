"""
Sample from a trained DDPM U-Net.

Example:
    python sample.py --checkpoint weights/ddpm_mnist_weights.pt --n-samples 16 --output generated.png
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src import UNet, make_schedule, sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a trained DDPM U-Net.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="weights/ddpm_mnist_weights.pt",
        help="Path to the trained .pt weights file.",
    )
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--output", type=str, default="generated_samples.png")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def save_grid(samples: torch.Tensor, n_samples: int, output_path: Path) -> None:
    """Save samples as a square (or near-square) grid image."""
    grid_size = int(math.ceil(math.sqrt(n_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")

    plt.suptitle(f"DDPM samples (n={n_samples})", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device
    print(f"Device: {device}")

    # Model and schedule
    model = UNet(
        in_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    schedule = make_schedule(T=args.timesteps).to(device)

    # Load weights
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {checkpoint_path}")

    # Sample
    print(f"Generating {args.n_samples} samples...")
    samples = sample(
        model=model,
        schedule=schedule,
        n_samples=args.n_samples,
        image_size=args.image_size,
        channels=1,
        device=device,
    )
    samples = ((samples + 1) / 2).clamp(0, 1).cpu()

    # Save
    save_grid(samples, args.n_samples, Path(args.output))
    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    main()
