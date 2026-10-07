"""2D STN demo: end-to-end MNIST classification with an STN in front of a CNN.

Digits are placed on a 48x48 canvas and distorted by a random rotation, scale and shift
plus a smooth elastic field (a distortion outside the CPAB family). The same CNN is
trained from labels only, without and with a CPAB STN in front of it.

Repeatable: the distorted train/test sets are generated once from fixed seeds (bit-identical
across runs), and every model starts from the same classifier weights and sees the same
batches in the same order. The no-STN run is reproducible exactly; the STN runs can differ
by a few tenths of a percent, since the grid_sample and CPAB backward passes use GPU atomics.

Run from the repo root:  python stn/demos/demo_classify_2d.py   (downloads MNIST, ~11 MB)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from stn import CpabWarp, SpatialTransformer

dev = "cuda"
torch.backends.cudnn.deterministic = True  # deterministic convolutions
SEED, SIZE, EPOCHS, BATCH = 7, 48, 10, 128
DATA = Path.home() / ".cache" / "torchvision"


def distort(x, gen):
    """Random rotation (+-30 deg), scale (+-20%) and shift, then a smooth elastic field."""
    n = len(x)
    rand = lambda *shape: torch.rand(*shape, generator=gen, device=dev) * 2 - 1  # U(-1, 1)
    angle, scale, shift = rand(n) * math.radians(30), 1 + 0.2 * rand(n), 0.3 * rand(n, 2)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    A = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    field = F.interpolate(0.12 * torch.randn(n, 2, 4, 4, generator=gen, device=dev), size=x.shape[-2:],
                          mode="bicubic", align_corners=True)
    grid = F.affine_grid(A, x.shape, align_corners=True) + field.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, align_corners=True)


def load(train, seed):
    """MNIST digits on a SIZE x SIZE canvas, distorted with a generator seeded by `seed`."""
    ds = datasets.MNIST(DATA, train=train, download=True)
    x = F.pad(ds.data.to(dev, torch.float32)[:, None] / 255, [(SIZE - 28) // 2] * 4)
    gen = torch.Generator(device=dev).manual_seed(seed)
    return torch.cat([distort(chunk, gen) for chunk in x.split(10_000)]), ds.targets.to(dev)


def locnet():
    return nn.Sequential(
        nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(),
    )


def classifier():
    return nn.Sequential(
        nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * (SIZE // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, 10),
    )


def build(warp):
    torch.manual_seed(SEED)  # the same classifier weights in every run
    cnn = classifier()
    return (cnn if warp is None else nn.Sequential(SpatialTransformer(locnet(), 64, warp), cnn)).to(dev)


@torch.no_grad()
def accuracy(model, x, y):
    return torch.cat([model(chunk).argmax(1) for chunk in x.split(1000)]).eq(y).float().mean().item()


def train(model, x, y):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(SEED)  # the same batch order in every run
    for epoch in range(EPOCHS):
        for idx in torch.randperm(len(x), generator=gen, device=dev).split(BATCH):
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  epoch {epoch + 1}: train loss {loss.item():.3f}, test accuracy {accuracy(model, *test):.2%}")


train_set, test = load(train=True, seed=SEED), load(train=False, seed=SEED + 1)
warps = {"no STN": None, "CPAB STN": CpabWarp((3, 3), zero_boundary=False)}
models, results = {}, {}
for name, warp in warps.items():
    print(f"{name}:")
    models[name] = build(warp)
    train(models[name], *train_set)
    results[name] = accuracy(models[name], *test)
summary = "   ".join(f"{name}: {acc:.2%}" for name, acc in results.items())
print(f"test accuracy   {summary}")

# Rows: distorted test digits, and what the CPAB STN feeds the classifier.
x = test[0][:10]
with torch.no_grad():
    rows = [("input", x), ("CPAB STN", models["CPAB STN"][0](x))]
fig, axes = plt.subplots(2, 10, figsize=(13, 3.2))
for r, (label, images) in enumerate(rows):
    for c, im in enumerate(images):
        axes[r, c].imshow(im[0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].set(xticks=[], yticks=[])
    axes[r, 0].set_ylabel(label, fontsize=12)
fig.suptitle(f"test accuracy   {summary}")
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
