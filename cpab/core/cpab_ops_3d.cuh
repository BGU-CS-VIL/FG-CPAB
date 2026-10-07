/**
 * CPAB CUDA operations header - Optimized 3D extension
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "cpab_ops_2d.cuh"

// Forward kernel (3D)
__global__ void cpab_cuda_kernel_forward_3D_optimized(
    const int nP,
    const int batch_size,
    float* __restrict__ newpoints,
    const float* __restrict__ points,
    const float* __restrict__ Trels,
    const float* __restrict__ As,
    const int nStepSolver,
    const int ncx,
    const int ncy,
    const int ncz,
    const int adaptive_substeps,
    const int broadcast);

// Fused backward kernel (3D): segmented checkpoints + adjoint moments
__global__ void cpab_cuda_kernel_fused_backward_3D(
    const int nP,
    const int n_theta,
    const int nC,
    const int nStepSolver,
    float* __restrict__ moments,
    float* __restrict__ grad_points,
    const float* __restrict__ grad_output,
    const float* __restrict__ points,
    const float* __restrict__ As,
    const int ncx,
    const int ncy,
    const int ncz,
    const int adaptive_substeps,
    const int broadcast);

// Optimized 4x4 matrix exponential kernel for affine 3D CPAB matrices
__global__ void cpab_cuda_kernel_expm4x4(
    const int n_matrices,
    const float dT,
    float* __restrict__ Trels_out,   // [n_matrices, 12] output (top 3 rows)
    const float* __restrict__ As_in  // [n_matrices, 12] input velocity fields
);
