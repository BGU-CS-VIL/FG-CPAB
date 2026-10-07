"""2D STN demo: learn to undo random CPAB warps of an image.

Run from the repo root:  python stn/demos/demo_2d.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from stn import CpabWarp, SpatialTransformer

torch.manual_seed(0)
dev = "cuda"
SIZE, TESS, STEPS, BATCH = 128, (3, 3), 5000, 32

# Template: the project's test card, as a [1, 3, SIZE, SIZE] image in [0, 1].
image = Image.open(ROOT / "docs/data/test-card.jpg").convert("RGB").resize((SIZE, SIZE))
template = torch.tensor(np.asarray(image), device=dev).permute(2, 0, 1)[None].float() / 255

distort = CpabWarp(TESS, padding_mode="border")


def sample(n):
    theta = 0.7 * torch.randn(n, distort.theta_dim, device=dev)
    return distort(template.expand(n, -1, -1, -1), theta)


# STN: two composed CPAB layers, each with a small conv localization net.
def locnet():
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(16), nn.Flatten(), nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
    )


stn = SpatialTransformer([locnet(), locnet()], 64, CpabWarp(TESS, padding_mode="border")).to(dev)
opt = torch.optim.Adam(stn.parameters(), lr=3e-4)
for step in range(STEPS):
    loss = F.mse_loss(stn(sample(BATCH)), template.expand(BATCH, -1, -1, -1))
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

# Columns: template, six samples, mean over the test batch; rows: input, aligned.
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for row, (data, label) in enumerate([(x, "input"), (y, "aligned")]):
    for col, im in enumerate([template[0], *data[:6], data.mean(0)]):
        axes[row, col].imshow(im.permute(1, 2, 0).clamp(0, 1).cpu())
        axes[row, col].set(xticks=[], yticks=[])
    axes[row, 0].set_ylabel(label, fontsize=13)
for col, title in [(0, "template"), (7, "mean")]:
    axes[0, col].set_title(title)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
