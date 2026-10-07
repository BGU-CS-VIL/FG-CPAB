#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Compile-time limits used by the optimized backward kernel.
// If you raise this, rebuild the extension so launcher checks and kernels stay in sync.
constexpr int CPAB_MAX_STEPS = 100;
constexpr int CKPT_SEGMENT_SIZE = 8;
constexpr int CKPT_MAX_SEGMENTS =
    (CPAB_MAX_STEPS + CKPT_SEGMENT_SIZE - 1) / CKPT_SEGMENT_SIZE;

// Forward kernel
__global__ void cpab_cuda_kernel_forward_2D_optimized(
    const int nP, 
    const int batch_size,
    float* __restrict__ newpoints, 
    const float* __restrict__ points,
    const float* __restrict__ Trels,
    const float* __restrict__ As,
    const int nStepSolver,
    const int ncx,
    const int ncy,
    const int adaptive_substeps,
    const int broadcast);

// Fused backward kernel (forward trajectory in local memory + adjoint moments)
__global__ void cpab_cuda_kernel_fused_backward(
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
    const int adaptive_substeps,
    const int broadcast);

// Optimized 3x3 matrix exponential kernel for CPAB affine velocity matrices
__global__ void cpab_cuda_kernel_expm3x3(
    const int n_matrices,
    const float dT,
    float* __restrict__ Trels_out,   // [n_matrices, 2, 3] output (top 2 rows)
    const float* __restrict__ As_in); // [n_matrices, 2, 3] input velocity fields
