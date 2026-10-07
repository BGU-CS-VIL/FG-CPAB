"""1D STN demo: learn to undo random CPAB warps of a signal.

Run from the repo root:  python stn/demos/demo_1d.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from stn import CpabWarp, SpatialTransformer

torch.manual_seed(0)
dev = "cuda"
LENGTH, TESS, STEPS, BATCH = 256, (8,), 2000, 64

# Template: three bumps of different heights and widths.
t = torch.linspace(-1, 1, LENGTH, device=dev)
template = sum(a * torch.exp(-(((t - c) / s) ** 2)) for a, c, s in [(1.0, -0.5, 0.08), (0.6, 0, 0.15), (0.9, 0.45, 0.06)])
template = template.view(1, 1, LENGTH)

# Data: the template under random CPAB warps that keep the end points fixed, plus noise.
distort = CpabWarp(TESS, zero_boundary=True)


def sample(n):
    theta = 1.0 * torch.randn(n, distort.theta_dim, device=dev)
    return distort(template.expand(n, -1, -1), theta) + 0.02 * torch.randn(n, 1, LENGTH, device=dev)


# STN: two composed CPAB layers, each with a small conv localization net.
def locnet():
    return nn.Sequential(
        nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        nn.Flatten(), nn.Linear(32 * 8, 64), nn.ReLU(),
    )


stn = SpatialTransformer([locnet(), locnet()], 64, CpabWarp(TESS, zero_boundary=True)).to(dev)
opt = torch.optim.Adam(stn.parameters(), lr=1e-3)
for step in range(STEPS):
    loss = F.mse_loss(stn(sample(BATCH)), template.expand(BATCH, -1, -1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step:4d}  loss {loss.item():.5f}")

# Evaluate on fresh samples.
with torch.no_grad():
    x = sample(32)
    y = stn(x)
target = template.expand_as(x)
print(f"test MSE to template: input {F.mse_loss(x, target):.5f}, aligned {F.mse_loss(y, target):.5f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, data, title in zip(axes, (x, y), ("Input: random warps", "Aligned by the STN")):
    ax.plot(t.cpu(), data[:, 0].T.cpu(), color="#94a3b8", lw=0.6, alpha=0.7)
    ax.plot(t.cpu(), data[:, 0].mean(0).cpu(), color="#0f766e", lw=2, label="mean")
    ax.plot(t.cpu(), template[0, 0].cpu(), "--", color="#db2777", lw=1.5, label="template")
    ax.set_title(title)
    ax.grid(alpha=0.25)
axes[0].legend(loc="upper left")
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=160)
print(f"Saved: {out}")
