"""3D STN demo: learn to undo random CPAB warps of a volume.

Run from the repo root:  python stn/demos/demo_3d.py
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
SIZE, TESS, STEPS, BATCH = 32, (4, 4, 4), 3000, 32

# Template: an asymmetric object made of Gaussian blobs, as a [1, 1, D, H, W] volume.
grid = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, SIZE, device=dev)] * 3, indexing="ij"))
blobs = [((0, 0, 0), 0.4), ((0, 0, 0.55), 0.28), ((0, 0.55, 0), 0.25), ((0.5, -0.35, -0.35), 0.25)]
template = sum(torch.exp(-((grid - torch.tensor(c, device=dev).view(3, 1, 1, 1)) ** 2).sum(0) / r**2) for c, r in blobs)
template = (template / template.max())[None, None]

# Data: the template under random CPAB warps.
distort = CpabWarp(TESS)


def sample(n):
    theta = 1.0 * torch.randn(n, distort.theta_dim, device=dev)
    return distort(template.expand(n, -1, -1, -1, -1), theta)


# STN: two composed CPAB layers, each with a small conv localization net.
def locnet():
    return nn.Sequential(
        nn.Conv3d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool3d(4), nn.Flatten(), nn.Linear(16 * 64, 256), nn.ReLU(),
    )


stn = SpatialTransformer([locnet(), locnet()], 256, CpabWarp(TESS)).to(dev)
opt = torch.optim.Adam(stn.parameters(), lr=1e-3)
for step in range(STEPS):
    loss = F.mse_loss(stn(sample(BATCH)), template.expand(BATCH, -1, -1, -1, -1))
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

# Maximum-intensity projections along depth. Columns: template, six samples, mean; rows: input, aligned.
mip = lambda v: v[0].amax(0).cpu()
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for row, (data, label) in enumerate([(x, "input"), (y, "aligned")]):
    for col, vol in enumerate([template[0], *data[:6], data.mean(0)]):
        axes[row, col].imshow(mip(vol), cmap="magma", vmin=0, vmax=1)
        axes[row, col].set(xticks=[], yticks=[])
    axes[row, 0].set_ylabel(label, fontsize=13)
for col, title in [(0, "template"), (7, "mean")]:
    axes[0, col].set_title(title)
fig.suptitle("Maximum-intensity projections along depth")
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
