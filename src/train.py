"""
Train a DDPM U-Net on MNIST.

Example:
    python train.py --epochs 30 --batch-size 128 --output weights/ddpm_mnist_weights.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src import UNet, make_schedule, q_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDPM U-Net on MNIST.")
    p.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=200, help="Number of diffusion timesteps T.")
    p.add_argument("--image-size", type=int, default=32, help="Spatial resolution to train on.")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--output", type=str, default="weights/ddpm_mnist_weights.pt")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} images | Batch size: {args.batch_size}")

    # Model and schedule
    model = UNet(
        in_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    schedule = make_schedule(T=args.timesteps).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for i, (x0, _) in enumerate(loader):
            x0 = x0.to(device)
            B = x0.shape[0]

            t = torch.randint(0, schedule.T, (B,), device=device).long()
            noise = torch.randn_like(x0)
            x_t = q_sample(schedule, x0, t, noise)

            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if i % args.log_every == 0:
                print(
                    f"Epoch {epoch + 1:2d}/{args.epochs} | "
                    f"Step {i:4d}/{len(loader)} | Loss {loss.item():.4f}"
                )

        avg = epoch_loss / n_batches
        elapsed = time.time() - epoch_start
        print(
            f"=== Epoch {epoch + 1:2d}/{args.epochs} done | "
            f"avg loss {avg:.4f} | time {elapsed:.1f}s ==="
        )

    # Save weights
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved weights to {output_path}")


if __name__ == "__main__":
    main()
