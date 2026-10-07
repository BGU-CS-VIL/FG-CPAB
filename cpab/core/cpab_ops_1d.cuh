#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "cpab_ops_2d.cuh"

// Dedicated closed-form forward kernel (1D)
__global__ void cpab_cuda_kernel_forward_1D_closed_form(
    const int nP,
    const int batch_size,
    float* __restrict__ newpoints,
    const float* __restrict__ points,
    const float* __restrict__ As,
    const int ncx,
    const int broadcast);

// Fused backward kernel (1D): segmented checkpoints + adjoint moments
__global__ void cpab_cuda_kernel_fused_backward_1D(
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
    const int adaptive_substeps,
    const int broadcast);

// Optimized 2x2 matrix exponential kernel for affine 1D CPAB matrices
__global__ void cpab_cuda_kernel_expm2x2(
    const int n_matrices,
    const float dT,
    float* __restrict__ Trels_out,   // [n_matrices, 2] output (top row)
    const float* __restrict__ As_in  // [n_matrices, 2] input velocity fields
);
