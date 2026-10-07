"""Minimal 1D CPAB example using the transform_grid path."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from cpab import Cpab


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    # Defaults only.
    tess = 5
    n_points = 8192
    iters = 2000
    lr = 0.4
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    cpab = Cpab([tess])

    x = torch.linspace(0.0, 1.0, n_points, device="cuda", dtype=torch.float32).view(1, -1)
    x_np = x[0].cpu().numpy()

    # Smooth monotone target.
    y_np = x_np + 0.18 * np.sin(2.0 * np.pi * x_np) + 0.03 * np.sin(4.0 * np.pi * x_np)
    y_np[0], y_np[-1] = x_np[0], x_np[-1]
    if np.any(np.diff(y_np) <= 0.0):
        raise ValueError("Target must be strictly increasing.")
    y = torch.tensor(y_np, device="cuda", dtype=torch.float32).view(1, 1, -1)

    theta = (torch.randn((1, cpab.get_theta_dim()), device="cuda") * 1e-3).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        pred = cpab.transform_grid(x, theta)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()

    theta = theta.detach()
    warped = cpab.transform_grid(x, theta)
    x_recovered = cpab.transform_grid(warped, -theta)

    print(f"Fit mean abs (warp vs target):      {(warped - y).abs().mean().item():.2e}")
    print(f"Cycle mean abs (recover vs source): {(x_recovered - x.view(1, 1, -1)).abs().mean().item():.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(x_np, y_np, color="#0284c7")
    axes[0].set_title("Target")
    axes[1].plot(x_np, warped[0, 0].detach().cpu().numpy(), color="#0f766e")
    axes[1].set_title("Learned warp")
    axes[2].plot(x_np, x_np, color="#9ca3af", label="source")
    axes[2].plot(x_np, x_recovered[0, 0].detach().cpu().numpy(), color="#db2777", label="recovered")
    axes[2].set_title("Warp + inverse")
    axes[2].legend(loc="upper left")

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("x'")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "rectangular_grid_diffeomorphism_1d.png")
    plt.savefig(out, dpi=160)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
