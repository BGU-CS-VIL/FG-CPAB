"""Minimal 2D CPAB example using the transform_grid path."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from cpab import Cpab


def build_line_grid(nx_lines, ny_lines, points_per_line, lo=0.05, hi=0.95):
    xs = np.linspace(lo, hi, nx_lines, dtype=np.float32)
    ys = np.linspace(lo, hi, ny_lines, dtype=np.float32)
    line = np.linspace(lo, hi, points_per_line, dtype=np.float32)

    lines = []
    slices = []
    start = 0

    for y in ys:
        pts = np.stack([line, np.full_like(line, y)], axis=1)
        lines.append(pts)
        end = start + points_per_line
        slices.append((start, end))
        start = end

    for x in xs:
        pts = np.stack([np.full_like(line, x), line], axis=1)
        lines.append(pts)
        end = start + points_per_line
        slices.append((start, end))
        start = end

    return np.concatenate(lines, axis=0), slices


def make_target(points_xy):
    x = points_xy[:, 0] - 0.5
    y = points_xy[:, 1] - 0.5
    r2 = x * x + y * y
    angle = 1.1 * np.exp(-8.0 * r2)
    c = np.cos(angle)
    s = np.sin(angle)

    out = points_xy.copy()
    out[:, 0] = np.clip(0.5 + c * x - s * y, 0.0, 1.0)
    out[:, 1] = np.clip(0.5 + s * x + c * y, 0.0, 1.0)
    return out.astype(np.float32)


def plot_grid(ax, points_xy, line_slices, color, label=None, alpha=1.0):
    for i, (start, end) in enumerate(line_slices):
        kwargs = {"color": color, "linewidth": 1.0, "alpha": alpha}
        if label is not None and i == 0:
            kwargs["label"] = label
        line = points_xy[start:end]
        ax.plot(line[:, 0], line[:, 1], **kwargs)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    # Defaults only.
    tess = [20, 20]
    grid_lines = 17
    points_per_line = 100
    iters = 2000
    lr = 0.2
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    cpab = Cpab(tess)

    x_np, line_slices = build_line_grid(grid_lines, grid_lines, points_per_line)
    y_np = make_target(x_np)

    x = torch.tensor(x_np.T, device="cuda", dtype=torch.float32)
    y = torch.tensor(y_np.T, device="cuda", dtype=torch.float32).unsqueeze(0)

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
    print(f"Cycle mean abs (recover vs source): {(x_recovered - x.unsqueeze(0)).abs().mean().item():.2e}")

    warped_np = warped[0].detach().cpu().numpy().T
    recovered_np = x_recovered[0].detach().cpu().numpy().T

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_grid(axes[0], y_np, line_slices, color="#0284c7")
    axes[0].set_title("Target")
    plot_grid(axes[1], warped_np, line_slices, color="#0f766e")
    axes[1].set_title("Learned warp")
    plot_grid(axes[2], x_np, line_slices, color="#9ca3af", label="source", alpha=0.65)
    plot_grid(axes[2], recovered_np, line_slices, color="#db2777", label="recovered", alpha=0.8)
    axes[2].set_title("Warp + inverse")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "rectangular_grid_diffeomorphism_2d.png")
    plt.savefig(out, dpi=160)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
