"""Spatial transformer networks (Jaderberg et al., 2015) with CPAB diffeomorphisms
(Detlefsen et al., 2018) computed by FG-CPAB, plus affine and homography baselines.

Inputs can be 1D [B, C, W], 2D [B, C, H, W] or 3D [B, C, D, H, W]. Stacked
transformer layers compose their warps, so every image is sampled once from the
original input instead of re-interpolating an already warped one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cpab import Cpab
from cpab.pytorch.transformer import CPAB_transformer


class _Warp(nn.Module):
    """Resample x [B, C, *spatial] on its own pixel grid deformed by theta; the output
    has the same spatial size and dtype as x, and theta = 0 is the identity.

    Subclasses implement transform(points, thetas) -> [B, ndim, P], mapping the pixel grid
    points [ndim, P] (in [-1, 1], ordered (x, y, z) like F.grid_sample) through T_1∘...∘T_K
    for thetas = [theta_1, ..., theta_K], where T_K acts first.
    """

    def __init__(self, ndim, padding_mode="zeros"):
        super().__init__()
        assert ndim in (1, 2, 3), "only 1D, 2D and 3D inputs are supported"
        self.ndim, self.padding_mode = ndim, padding_mode

    def transform(self, points, thetas):
        raise NotImplementedError

    def forward(self, x, theta):
        """theta: [B, theta_dim] for a single warp T, or a list (or [K, B, theta_dim] tensor)
        [theta_1, ..., theta_K] for the composition T_1∘...∘T_K, where T_K acts on the grid first.
        """
        size = x.shape[2:]
        assert len(size) == self.ndim, f"expected a {self.ndim}D input [B, C, *spatial], got {tuple(x.shape)}"
        thetas = [theta] if torch.is_tensor(theta) and theta.dim() == 2 else theta
        with torch.autocast(x.device.type, enabled=False):  # keep sampling points in float32
            # Pixel centres in [-1, 1]. A size-1 axis sits at 0, as in F.affine_grid (linspace would put it at -1).
            # Explicit dtype: linspace otherwise follows torch's default dtype, while theta is float32 here.
            f32 = dict(device=x.device, dtype=torch.float32)
            axes = [torch.linspace(-1, 1, s, **f32) if s > 1 else torch.zeros(1, **f32) for s in size]
            points = torch.stack(torch.meshgrid(*axes, indexing="ij")[::-1]).flatten(1)  # [ndim, P] pixel grid
            points = self.transform(points, [th.float() for th in thetas])
        # Sample in float32: casting the grid to a float16/bfloat16 x would round the sampling points.
        grid = points.transpose(1, 2).reshape(len(points), *size, -1)
        src = x.float()
        if self.ndim == 1:  # grid_sample has no 1D mode: sample a [B, C, 1, W] image at y = 0
            grid = F.pad(grid, (0, 1)).unsqueeze(1)
            src = src.unsqueeze(2)
        out = F.grid_sample(src, grid, align_corners=True, padding_mode=self.padding_mode)
        return (out.squeeze(2) if self.ndim == 1 else out).to(x.dtype)


class CpabWarp(_Warp):
    """CPAB diffeomorphism, integrated and differentiated with FG-CPAB.

    tess_size and theta mean the same as in cpab.Cpab: CpabWarp(tess)(x, theta) gives the
    same result as Cpab(tess).transform_data(x, theta, x.shape[2:]), which repeats the edge
    pixels outside the image like padding_mode="border".

    tess_size: cells per spatial dim; tess_size[i] counts the cells along x.shape[2 + i].
        Its length sets whether the warp is 1D, 2D or 3D.
    zero_boundary: True (default) keeps the image boundary fixed; False lets the warp
        move the whole image, e.g. translate, zoom or rotate it.
    cpab_kwargs: passed to cpab.Cpab (e.g. volume_perservation=True).
    """

    def __init__(self, tess_size, zero_boundary=True, nstepsolver=50, padding_mode="zeros", **cpab_kwargs):
        tess_size = [int(t) for t in tess_size]
        super().__init__(len(tess_size), padding_mode)
        self.cpab = Cpab(tess_size, zero_boundary=zero_boundary, **cpab_kwargs)
        self.cpab.set_solver_params(nstepsolver)
        self.theta_dim = self.cpab.get_theta_dim()

    def transform(self, points, thetas):
        # Cpab's coordinate i runs along x.shape[2 + i], the reverse of grid_sample's (x, y, z)
        # order, so flip the coordinates on the way in and out. The CPAB domain is [0, 1]^n.
        points = (points.flip(-2) + 1) / 2
        for theta in reversed(thetas):  # T_K acts first
            # Point gradients are needed only when the points come from another warp, not the constant grid.
            points = CPAB_transformer(points, theta, self.cpab.params, points.requires_grad)
        return (points * 2 - 1).flip(-2)


class _MatrixWarp(_Warp):
    """Warp given by an (n+1)x(n+1) matrix acting on homogeneous points. A composition
    T_1∘...∘T_K multiplies the matrices first and then maps the grid once.

    Subclasses implement matrix(theta) -> [B, n+1, n+1] and apply(M, points) -> [B, n, P].
    """

    def transform(self, points, thetas):
        M = self.matrix(thetas[0])
        for theta in thetas[1:]:
            M = M @ self.matrix(theta)  # M_1 @ ... @ M_K, so T_K acts first
        return self.apply(M, points)


class AffineWarp(_MatrixWarp):
    """Diffeomorphic affine warp (the classic STN baseline). theta [B, n(n+1)] is a velocity
    matrix [A | b]; the transformation is the matrix exponential of [[A, b], [0, 0]], so it is
    always invertible (its inverse is -theta) and theta = 0 is the identity.
    """

    def __init__(self, ndim, padding_mode="zeros"):
        super().__init__(ndim, padding_mode)
        self.theta_dim = ndim * (ndim + 1)

    def matrix(self, theta):
        n = self.ndim
        return torch.linalg.matrix_exp(F.pad(theta.view(-1, n, n + 1), (0, 0, 0, 1)))  # append the zero row

    def apply(self, M, points):
        return M[:, :-1, :-1] @ points + M[:, :-1, -1:]


class HomographyWarp(_MatrixWarp):
    """Homography (projective) warp, also via the matrix exponential. theta [B, n(n+2)] is a
    velocity matrix [[A, b], [c, 0]]: the first n(n+1) entries mean the same as in AffineWarp,
    and the last n (c) add perspective, so c = 0 gives exactly the affine warp. theta = 0 is the
    identity and -theta the inverse.
    """

    def __init__(self, ndim, padding_mode="zeros"):
        super().__init__(ndim, padding_mode)
        self.theta_dim = ndim * (ndim + 2)

    def matrix(self, theta):
        n = self.ndim
        return torch.linalg.matrix_exp(F.pad(theta, (0, 1)).view(-1, n + 1, n + 1))  # [[A, b], [c, 0]]

    def apply(self, H, points):
        p = H[:, :, :-1] @ points + H[:, :, -1:]  # homogeneous points [B, n+1, P]
        w = p[:, -1:]
        w = torch.where(w < 0, w.clamp(max=-1e-18), w.clamp(min=1e-18))
        return (p[:, :-1] / w).clamp(-1e6, 1e6)


class SpatialTransformer(nn.Module):
    """One or more stacked spatial transformer layers that compose their warps.

    Layer k sees x warped by T_1∘...∘T_{k-1} and predicts theta_k = head_k(locnet_k(.)).
    Every image, including the output, is sampled once from the original x, so nothing
    is lost to repeated interpolation (blur, content pushed out of frame). K CPAB layers
    cost K(K+1)/2 integrations, since each composition is integrated exactly.

    locnets: a module, or a list with one module per layer, mapping an image to [B, features].
    warp: a CpabWarp, AffineWarp or HomographyWarp, shared by all layers.
    The linear heads are zero-initialised, so training starts at the identity.
    """

    def __init__(self, locnets, features, warp):
        super().__init__()
        self.locnets = nn.ModuleList(locnets if isinstance(locnets, (list, tuple, nn.ModuleList)) else [locnets])
        if not self.locnets:
            raise ValueError("locnets must contain at least one module")
        self.heads = nn.ModuleList(nn.Linear(features, warp.theta_dim) for _ in self.locnets)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.warp = warp

    def forward(self, x, return_theta=False):
        """Returns the warped x, plus theta [K, B, theta_dim] if return_theta."""
        return self._compose(x, zip(self.locnets, self.heads), return_theta)

    def _compose(self, x, layers, return_theta):
        thetas, out = [], x
        for locnet, head in layers:
            thetas.append(head(locnet(out)))
            out = self.warp(x, thetas)  # always sample the original x
        return (out, torch.stack(thetas)) if return_theta else out


class LoopedSpatialTransformer(SpatialTransformer):
    """One localization net (and head) applied n_iters times for iterative refinement,
    with the warps composed as in SpatialTransformer.

    Weights are shared across iterations, so the parameter count does not depend on
    n_iters, and forward can run more or fewer iterations than training used (at least 1).
    n_iters CPAB iterations cost n_iters(n_iters+1)/2 integrations.
    """

    def __init__(self, locnet, features, warp, n_iters=3):
        super().__init__([locnet], features, warp)
        self.n_iters = self._check_iters(n_iters)

    def forward(self, x, return_theta=False, n_iters=None):
        """Returns the warped x, plus theta [n_iters, B, theta_dim] if return_theta.
        n_iters defaults to the value given at construction."""
        n_iters = self.n_iters if n_iters is None else self._check_iters(n_iters)
        layer = (self.locnets[0], self.heads[0])
        return self._compose(x, [layer] * n_iters, return_theta)

    @staticmethod
    def _check_iters(n_iters):
        if n_iters < 1:
            raise ValueError(f"n_iters must be at least 1, got {n_iters}")
        return n_iters
