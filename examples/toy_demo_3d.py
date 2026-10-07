"""
3D toy demo: recover a known 3D CPAB transformation from transformed points.

This script:
  1. Creates a known ground-truth 3D transformation theta*.
  2. Transforms a sphere mesh with theta* using the public CPAB API.
  3. Starts from theta=0 and optimizes theta to recover theta* by minimizing
     MSE on transformed 3D vertices.
  4. Saves an interactive HTML viewer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from cpab import Cpab
from cpab.core.utility import get_dir
from helpers.toy_demo_3d_plotting import save_mesh_viewer


def create_sphere_mesh(n_lat=30, n_lon=45, radius=0.35, center=(0.5, 0.5, 0.5)):
    """Create a UV-sphere mesh centered in [0,1]^3."""
    phi = np.linspace(0, np.pi, n_lat)
    theta = np.linspace(0, 2 * np.pi, n_lon)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
    uv = np.stack([(theta / (2 * np.pi)).ravel(), (phi / np.pi).ravel()], axis=1)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            v0 = i * n_lon + j
            v1 = v0 + 1
            v2 = (i + 1) * n_lon + j
            v3 = v2 + 1
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])

    return verts, faces, uv


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires CUDA.")

    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    tess = [5, 5, 5]
    nstepsolver = 50
    n_iters = 1000
    lr = 0.05

    print("=" * 70)
    print("3D Toy Demo")
    print("=" * 70)
    print(f"Config: tess={tess}, nstepsolver={nstepsolver}, n_iters={n_iters}, lr={lr}")

    T = Cpab(tess, zero_boundary=True, volume_perservation=False, override=False)
    T.set_solver_params(nstepsolver=nstepsolver)
    print(f"Theta dim: {T.get_theta_dim()}, Cells: {T.get_params().nC}")

    verts, faces, uv = create_sphere_mesh(n_lat=36, n_lon=54, radius=0.35, center=(0.5, 0.5, 0.5))
    points = torch.tensor(verts, dtype=torch.float32, device="cuda")
    print(f"Sphere mesh: {verts.shape[1]} vertices, {len(faces)} faces")

    theta_star = T.sample_transformation(1) * 2
    print(f"theta* shape: {tuple(theta_star.shape)}")

    points_target = T.transform_grid(points, theta_star).detach()

    theta_est = torch.zeros_like(theta_star, requires_grad=True)
    optimizer = torch.optim.Adam([theta_est], lr=lr)

    point_errors = []

    for i in range(n_iters):
        optimizer.zero_grad()
        points_pred = T.transform_grid(points, theta_est)
        loss = torch.mean((points_pred - points_target) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            point_errors.append((points_pred - points_target).norm(dim=1).mean().item())

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  iter {i + 1:4d} | loss={loss.item():.6e} | "
                f"mean point err={point_errors[-1]:.6e}"
            )

    points_recovered = T.transform_grid(points, theta_est.detach())
    final_point_diff = (points_recovered - points_target).abs()

    print("\nRecovery summary:")
    print(f"  Final loss: {loss.item():.2e}")
    print(f"  Final max point abs diff: {final_point_diff.max().item():.2e}")
    print(f"  Final mean point abs diff: {final_point_diff.mean().item():.2e}")

    html_path = os.path.join(get_dir(__file__), "toy_demo_3d.html")
    save_mesh_viewer(verts, faces, uv, points_target, points_recovered, html_path)
    print(f"Saved interactive viewer: {html_path}")


if __name__ == "__main__":
    main()
