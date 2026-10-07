# Spatial transformer networks with FG-CPAB

A [spatial transformer](https://arxiv.org/abs/1506.02025) (STN) is a layer that looks at its input, predicts a transformation, and warps the input with it before passing it on. It is trained end to end from the task loss alone. This folder implements STNs whose warp is a **CPAB diffeomorphism**, integrated and differentiated with FG-CPAB, following Deep Diffeomorphic Transformer Networks (Detlefsen et al., 2018).

- Works on 1D signals `[B, C, W]`, 2D images `[B, C, H, W]` and 3D volumes `[B, C, D, H, W]`.
- Supports several transformer layers, stacked or looped. Their warps are **composed**, so the input is sampled once, never re-interpolated from an already warped image.
- Includes affine and homography warps as baselines. Both are built with the matrix exponential, so they are always invertible.

Everything is in [`stn.py`](stn.py). Like FG-CPAB itself, it needs a CUDA GPU. Run your code from the repository root.

## Quick start

```python
import torch.nn as nn
from stn import CpabWarp, SpatialTransformer

# Any network that maps the input to [B, features] can be the localization net.
locnet = nn.Sequential(
    nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
    nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
    nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(),
)
stn = SpatialTransformer(locnet, 64, CpabWarp((3, 3), zero_boundary=False))
model = nn.Sequential(stn, classifier).cuda()   # train with your usual loss

y, theta = stn(x, return_theta=True)            # y has the same shape as x
```

The warps can also be used on their own:

```python
warp = CpabWarp((4, 4))                 # 4x4 tessellation, 2D
y = warp(x, theta)                      # theta: [B, warp.theta_dim]; theta = 0 is the identity
y = warp(x, [theta_1, theta_2])         # composition T_1(T_2(.)); x is sampled once
```

## Demos

Run from the repository root. Each script saves a figure next to itself (`stn/demos/<name>.png`).

```bash
python stn/demos/demo_1d.py
python stn/demos/demo_2d.py
python stn/demos/demo_3d.py
python stn/demos/demo_classify_2d.py
```

| Script | What it shows |
|---|---|
| [`demo_1d.py`](demos/demo_1d.py) | A 2-layer STN learns to undo random CPAB warps of a 1D signal, trained with MSE to the clean template. |
| [`demo_2d.py`](demos/demo_2d.py) | The same for a 2D image, the project's test card. |
| [`demo_3d.py`](demos/demo_3d.py) | The same for a 3D volume, shown as maximum-intensity projections. |
| [`demo_classify_2d.py`](demos/demo_classify_2d.py) | End-to-end MNIST classification on distorted digits (random rotation, scale, shift and elastic bending). The same CNN is trained with and without a CPAB STN, from labels only. The data is generated from fixed seeds. Downloads MNIST (~11 MB) to `~/.cache/torchvision`. |