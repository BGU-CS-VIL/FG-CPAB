# -*- coding: utf-8 -*-
"""
Benchmark for the published CPAB transformer implementation.

This benchmark measures forward and backward runtime for the
CUDA transformer implementation used by the package.
"""

import sys
import os
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from cpab import Cpab
from cpab.pytorch.transformer import CPAB_transformer
from cpab.pytorch.interpolation import interpolate


def save_images_grid(images, output_path, cols='auto'):
    """Save a list of images as a grid to a file.
    
    Arguments:
        images: List/tensor of np.arrays (H, W, C) with values in [0, 1].
        output_path: Path to save the output image.
        cols: Number of columns in the grid (default 'auto').
    """
    n_images = len(images)
    cols = int(np.round(np.sqrt(n_images))) if cols == 'auto' else cols
    rows = int(np.ceil(n_images / float(cols)))
    
    # Get image dimensions
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if images[0].ndim == 3 else 1
    
    # Create grid
    grid = np.zeros((rows * h, cols * w, channels), dtype=np.uint8)
    
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        # Convert to uint8
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img_uint8
    
    # Save image
    Image.fromarray(grid).save(output_path)
    print(f"Saved grid to: {output_path}")


def benchmark_forward_speed(func, points, theta, params, n_warmup=10, n_iterations=100):
    """Benchmark the speed of forward pass.
    
    Args:
        func: Transformer function to benchmark
        points: Input points tensor
        theta: Transformation parameters tensor
        params: CPAB parameters
        n_warmup: Number of warmup iterations
        n_iterations: Number of timed iterations
        
    Returns:
        dict with timing statistics
    """
    # Warmup
    for _ in range(n_warmup):
        output = func(points, theta, params)
        torch.cuda.synchronize()
    
    # Forward pass timing
    forward_times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = func(points, theta, params)
        torch.cuda.synchronize()
        end = time.perf_counter()
        forward_times.append(end - start)
    
    return {
        'mean_ms': np.mean(forward_times) * 1000,
        'std_ms': np.std(forward_times) * 1000,
        'min_ms': np.min(forward_times) * 1000,
        'max_ms': np.max(forward_times) * 1000,
    }


def benchmark_backward_speed(func, points, theta, params, n_warmup=10, n_iterations=100):
    """Benchmark the speed of backward pass.
    
    Args:
        func: Transformer function to benchmark
        points: Input points tensor
        theta: Transformation parameters tensor (must require grad)
        params: CPAB parameters
        n_warmup: Number of warmup iterations
        n_iterations: Number of timed iterations
        
    Returns:
        dict with timing statistics
    """
    # Warmup
    for _ in range(n_warmup):
        theta_clone = theta.clone().detach().requires_grad_(True)
        output = func(points, theta_clone, params)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
    
    # Backward pass timing
    backward_times = []
    for _ in range(n_iterations):
        theta_clone = theta.clone().detach().requires_grad_(True)
        output = func(points, theta_clone, params)
        loss = output.sum()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        backward_times.append(end - start)
    
    return {
        'mean_ms': np.mean(backward_times) * 1000,
        'std_ms': np.std(backward_times) * 1000,
        'min_ms': np.min(backward_times) * 1000,
        'max_ms': np.max(backward_times) * 1000,
    }

def create_test_data(grid_size, ndim, device='cuda'):
    """Create an ndim uniform grid of points for benchmarking.

    Args:
        grid_size: Number of points per axis
        ndim: Number of spatial dimensions (1, 2, or 3)
        device: Device to create tensors on

    Returns:
        points tensor with shape [ndim, grid_size**ndim]
    """
    axes = [torch.linspace(0, 1, grid_size, device=device) for _ in range(ndim)]
    mesh = torch.meshgrid(*axes, indexing='ij')
    points = torch.stack([coord.flatten() for coord in mesh], dim=0).contiguous()
    return points


def initialize_benchmark_state(args):
    """Build params/theta for the configured dimensionality."""
    tess = [args.tess] * args.ndim
    T = Cpab(tess, zero_boundary=False, volume_perservation=True, override=False)
    T.set_solver_params(
        nstepsolver=args.nstepsolver,
        adaptive_substeps=args.adaptive_substeps,
    )
    theta = T.sample_transformation(args.batch_size)
    return T.params, theta


def print_timing_stats(name, stats):
    """Print timing statistics in a consistent format."""
    print(f"  {name}: {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")
    print(f"  {name} range: [{stats['min_ms']:.3f}, {stats['max_ms']:.3f}] ms")





def run_benchmark(args):
    """Run the complete benchmark suite."""
    print("=" * 70)
    print("CPAB Transformer Benchmark")
    print("=" * 70)
    n_points = args.grid_size ** args.ndim
    tess_display = "x".join([str(args.tess)] * args.ndim)
    print(f"\nConfiguration:")
    print(f"  Dimensions: {args.ndim}D")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grid size: {args.grid_size}^{args.ndim} = {n_points} points")
    print(f"  Tessellation: {tess_display}")
    print(f"  Solver steps: {args.nstepsolver}")
    print(f"  Adaptive substeps: {args.adaptive_substeps}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Benchmark iterations: {args.iterations}")
    print()

    # Initialize CPAB params/theta and create test grid
    params, theta = initialize_benchmark_state(args)
    points = create_test_data(args.grid_size, args.ndim, device="cuda")
    
    print(f"Points shape: {points.shape}")
    print(f"Theta shape: {theta.shape}")
    print()
    

    
    # ========================================================================
    # SPEED BENCHMARKS - FORWARD
    # ========================================================================
    
    print("-" * 70)
    print("Forward Speed Benchmark")
    print("-" * 70)
    
    forward_stats = benchmark_forward_speed(
        CPAB_transformer, points, theta, params,
        n_warmup=args.warmup, n_iterations=args.iterations
    )
    print_timing_stats("Forward", forward_stats)
    print()
    
    # ========================================================================
    # SPEED BENCHMARKS - BACKWARD
    # ========================================================================
    
    print("-" * 70)
    print("Backward Speed Benchmark")
    print("-" * 70)
    
    backward_stats = benchmark_backward_speed(
        CPAB_transformer, points, theta, params,
        n_warmup=args.warmup, n_iterations=args.iterations
    )
    print_timing_stats("Backward", backward_stats)
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\nForward Pass:  {forward_stats['mean_ms']:.3f} ms")
    print(f"Backward Pass: {backward_stats['mean_ms']:.3f} ms")
    print()


def run_image_demo(args):
    """Run the demo with an actual image, similar to simple_demo.py.
    
    Saves output images from the current implementation.
    """
    print("=" * 70)
    print("Image Transformation Demo")
    print("=" * 70)

    if args.ndim != 2:
        print("Image demo is only supported for 2D. Skipping demo.")
        print()
        return
    
    N = 9  # Number of transformed samples
    outsize = (350, 350)
    
    # Load image data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cat.jpg')
    if not os.path.exists(data_path):
        print(f"Warning: Image not found at {data_path}")
        print("Skipping image demo")
        return
    
    data = np.array(Image.open(data_path)) / 255.0
    data = np.tile(data[None], [N, 1, 1, 1])
    
    # Create transformer
    T = Cpab([args.tess, args.tess], zero_boundary=True, volume_perservation=False, override=False)
    T.set_solver_params(
        nstepsolver=args.nstepsolver,
        adaptive_substeps=args.adaptive_substeps,
    )
    
    # Convert to PyTorch tensor (N, C, H, W) format
    data_tensor = torch.tensor(data, dtype=torch.float32, device='cuda')
    data_tensor = data_tensor.permute(0, 3, 1, 2)
    
    # Sample transformation
    theta = T.sample_transformation(N)
    
    print(f"\nTransforming {N} images with outsize {outsize}...")
    
    # Create the grid for transformation
    grid = T.uniform_meshgrid(outsize)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    points_transformed = CPAB_transformer(grid, theta, T.params)
    torch.cuda.synchronize()
    transform_time = time.perf_counter() - start

    # Interpolate to get final images.
    t_data_transformed = interpolate(2, data_tensor, points_transformed, outsize)
    
    print(f"\nPoint transformation:")
    print(f"  Runtime: {transform_time * 1000:.2f} ms")
    
    # Convert to numpy for saving (N, H, W, C) format
    t_data_transformed_np = t_data_transformed.permute(0, 2, 3, 1).cpu().numpy()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    transformed_output_path = os.path.join(output_dir, 'transformed.png')
    save_images_grid(t_data_transformed_np, transformed_output_path)
    
    # Also save original (untransformed) images for reference
    original_data_np = data_tensor.permute(0, 2, 3, 1).cpu().numpy()
    # Resize to match output size
    from PIL import Image as PILImage
    original_resized = []
    for i in range(N):
        img = PILImage.fromarray((original_data_np[i] * 255).astype(np.uint8))
        img_resized = img.resize(outsize, PILImage.Resampling.BILINEAR)
        original_resized.append(np.array(img_resized) / 255.0)
    original_input_path = os.path.join(output_dir, 'original_input.png')
    save_images_grid(original_resized, original_input_path)
    
    print(f"\nOutput images saved to: {output_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPAB Transformer Benchmark')
    parser.add_argument('--ndim', type=int, choices=[1, 2, 3], default=1, help='Spatial dimensionality')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--grid-size', type=int, default=128, help='Points per axis (creates grid_size^ndim points)')
    parser.add_argument('--tess', type=int, default=10, help='Tessellation size per axis')
    parser.add_argument('--nstepsolver', type=int, default=50, help='Number of solver steps')
    parser.add_argument('--adaptive-substeps', action='store_true', help='Enable adaptive substeps (if supported)')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=1, help='Benchmark iterations')
    parser.add_argument('--demo', action='store_true', help='Run image demo')
    
    args = parser.parse_args()
    
    if args.demo:
        run_image_demo(args)
    
    run_benchmark(args)
