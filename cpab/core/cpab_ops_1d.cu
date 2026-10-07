#include <cuda.h>
#include <cuda_runtime.h>
#include "cpab_ops_1d.cuh"

// ============================================================================
// WARP-AGGREGATED ATOMIC ADD FOR 2-ELEMENT MOMENTS
// ============================================================================

__device__ __forceinline__ void warp_aggregated_moment_flush_2(
    float* __restrict__ moments_base,
    const int cell,
    const float local_m[2])
{
    const unsigned active = __activemask();
    const unsigned match = __match_any_sync(active, cell);
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(match) - 1;

    if (match == (1u << lane)) {
        const int idx = cell * 2;
        atomicAdd(&moments_base[idx + 0], local_m[0]);
        atomicAdd(&moments_base[idx + 1], local_m[1]);
        return;
    }

    float s0 = 0.0f;
    float s1 = 0.0f;

    unsigned peers = match;
    while (peers) {
        const int src = __ffs(peers) - 1;
        const float v0 = __shfl_sync(match, local_m[0], src);
        const float v1 = __shfl_sync(match, local_m[1], src);
        if (lane == leader) {
            s0 += v0;
            s1 += v1;
        }
        peers &= peers - 1;
    }

    if (lane == leader) {
        const int idx = cell * 2;
        atomicAdd(&moments_base[idx + 0], s0);
        atomicAdd(&moments_base[idx + 1], s1);
    }
}

// ============================================================================
// CELL INDEX LOOKUP
// ============================================================================

__device__ __forceinline__ int cuda_findcellidx_1D_fast(const float px, const int ncx)
{
    const float eps = 1e-7f;
    const float x_clamped = fminf(fmaxf(px, 0.0f), 1.0f - eps);
    int cell_x = __float2int_rd(x_clamped * (float)ncx);
    cell_x = min(cell_x, ncx - 1);
    return cell_x;
}

// ============================================================================
// FORWARD KERNEL
// ============================================================================

__device__ __forceinline__ int cpab_signf_1D(const float x)
{
    return (x > 0.0f) - (x < 0.0f);
}

__device__ __forceinline__ bool cpab_cmpf0_1D(const float x)
{
    return fabsf(x) < 1e-7f;
}

__device__ __forceinline__ float cpab_right_boundary_1D(const int c, const int ncx)
{
    return ((float)(c + 1) / (float)ncx) + 1e-7f;
}

__device__ __forceinline__ float cpab_left_boundary_1D(const int c, const int ncx)
{
    return ((float)c / (float)ncx) - 1e-7f;
}

__device__ __forceinline__ float cpab_get_psi_1D(const float x, const float t, const float a, const float b)
{
    if (cpab_cmpf0_1D(a)) {
        return __fmaf_rn(t, b, x);
    }

    const float b_over_a = b / a;
    return expf(t * a) * (x + b_over_a) - b_over_a;
}

__device__ __forceinline__ float cpab_get_hit_time_1D(
    const float x,
    const float t,
    const int c,
    const float a,
    const float b,
    const int ncx,
    float& xc,
    int& cc)
{
    const int s = cpab_signf_1D(t);
    const float inf_t = INFINITY * (float)s;

    const float v = __fmaf_rn(a, x, b);
    if (cpab_cmpf0_1D(v)) return inf_t;

    cc = c + cpab_signf_1D(v) * s;
    if (cc < 0 || cc >= ncx) return inf_t;

    if (t > 0.0f) {
        xc = (v > 0.0f) ? cpab_right_boundary_1D(c, ncx) : cpab_left_boundary_1D(c, ncx);
    } else {
        xc = (v > 0.0f) ? cpab_left_boundary_1D(c, ncx) : cpab_right_boundary_1D(c, ncx);
    }

    const float vc = __fmaf_rn(a, xc, b);
    if (cpab_cmpf0_1D(vc)) return inf_t;
    if (cpab_signf_1D(v) != cpab_signf_1D(vc)) return inf_t;
    if (xc <= 0.0f || xc >= 1.0f) return inf_t;

    if (cpab_cmpf0_1D(a)) {
        return (xc - x) / b;
    }
    return logf(vc / v) / a;
}

__device__ __forceinline__ float cpab_integrate_closed_form_1D(
    float x,
    float t,
    const float* __restrict__ As,
    const int As_batch_offset,
    const int ncx)
{
    int c = cuda_findcellidx_1D_fast(x, ncx);
    int cont = 0;
    const int contmax = max(c, ncx - 1 - c) + 1;
    const int s = cpab_signf_1D(t);

    while (true) {
        const float* A = As + As_batch_offset + 2 * c;
        const float a = A[0];
        const float b = A[1];

        float xc = x;
        float thit = 0.0f;
        int cc = c;

        thit = cpab_get_hit_time_1D(x, t, c, a, b, ncx, xc, cc);
        if ((float)s * thit > (float)s * t) {
            return cpab_get_psi_1D(x, t, a, b);
        }

        x = xc;
        c = cc;
        t -= thit;

        cont++;
        if (cont > contmax) break;
    }

    return x;
}

__device__ __forceinline__ void cpab_midpoint_step_1D_cell(
    float& px,
    const float* __restrict__ As,
    const int As_batch_offset,
    const int cellidx,
    const float dt)
{
    const float* A = As + As_batch_offset + 2 * cellidx;
    const float half_dt = 0.5f * dt;

    const float vx = __fmaf_rn(A[0], px, A[1]);
    const float pMid = __fmaf_rn(half_dt, vx, px);
    const float vMid = __fmaf_rn(A[0], pMid, A[1]);
    px = __fmaf_rn(dt, vMid, px);
}

__global__ void cpab_cuda_kernel_forward_1D_closed_form(
    const int nP,
    const int batch_size,
    float* __restrict__ newpoints,
    const float* __restrict__ points,
    const float* __restrict__ As,
    const int ncx,
    const int broadcast)
{
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (point_index < nP && batch_index < batch_size) {
        const int input_offset = broadcast * batch_index * nP;
        float px = points[input_offset + point_index];

        const int As_batch_offset = batch_index * 2 * ncx;
        px = cpab_integrate_closed_form_1D(px, 1.0f, As, As_batch_offset, ncx);

        newpoints[nP * batch_index + point_index] = px;
    }
}

// ============================================================================
// FUSED BACKWARD KERNEL — Segmented checkpointing (1D)
// ============================================================================

__device__ __forceinline__ int cpab_forward_step_1D(
    float& px,
    const float* __restrict__ As,
    const int As_batch_offset,
    const float h,
    const float h_half,
    const int ncx)
{
    const int cellidx = cuda_findcellidx_1D_fast(px, ncx);
    const float* A = As + As_batch_offset + 2 * cellidx;

    const float vx = __fmaf_rn(A[0], px, A[1]);
    const float pMid = __fmaf_rn(h_half, vx, px);
    const float vMid = __fmaf_rn(A[0], pMid, A[1]);

    px = __fmaf_rn(h, vMid, px);
    return cellidx;
}

__device__ __forceinline__ void cpab_forward_step_1D_adaptive(
    float& px,
    const float* __restrict__ As,
    const int As_batch_offset,
    const float h,
    const float h_half,
    const int ncx,
    const int adaptive_substeps)
{
    if (adaptive_substeps == 0) {
        cpab_forward_step_1D(px, As, As_batch_offset, h, h_half, ncx);
        return;
    }

    const int cellidx_start = cuda_findcellidx_1D_fast(px, ncx);
    float step_px = px;
    cpab_midpoint_step_1D_cell(step_px, As, As_batch_offset, cellidx_start, h);

    const int cellidx_end = cuda_findcellidx_1D_fast(step_px, ncx);
    if (cellidx_end == cellidx_start) {
        px = step_px;
        return;
    }

    const float* A0 = As + As_batch_offset + 2 * cellidx_start;
    const float v0 = __fmaf_rn(A0[0], px, A0[1]);

    const float* A1 = As + As_batch_offset + 2 * cellidx_end;
    const float v1 = __fmaf_rn(A1[0], step_px, A1[1]);

    px = __fmaf_rn(h_half, v0 + v1, px);
}

__device__ __forceinline__ void cpab_accumulate_moments_and_adjoint_1D(
    const float* __restrict__ A,
    const float px,
    const float h,
    const float h_half,
    float* __restrict__ local_m,
    float& lam)
{
    const float A0 = A[0];
    const float A1 = A[1];

    const float vx = __fmaf_rn(A0, px, A1);
    const float pMid = __fmaf_rn(h_half, vx, px);

    float At_lam = A0 * lam;

    const float w_pMid = h * lam;
    const float w_p = h_half * h * At_lam;

    local_m[0] += w_pMid * pMid;
    local_m[1] += w_pMid;

    local_m[0] += w_p * px;
    local_m[1] += w_p;

    const float lam_mid = lam + h_half * At_lam;
    At_lam = A0 * lam_mid;
    lam += h * At_lam;
}

__global__ void cpab_cuda_kernel_fused_backward_1D(
    const int nP,
    const int n_theta,
    const int nC,
    const int nStepSolver,
    float* __restrict__ moments,           // [n_theta, nC, 2]
    float* __restrict__ grad_points,       // [n_theta, 1, nP] or nullptr
    const float* __restrict__ grad_output, // [n_theta, 1, nP]
    const float* __restrict__ points,
    const float* __restrict__ As,
    const int ncx,
    const int adaptive_substeps,
    const int broadcast)
{
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_index = blockIdx.y;

    if (point_index >= nP || batch_index >= n_theta) return;

    const float h = 1.0f / (float)nStepSolver;
    const float h_half = h * 0.5f;
    const int As_batch_offset = batch_index * 2 * nC;

    const int input_offset = broadcast * batch_index * nP;
    float px = points[input_offset + point_index];

    const int n_segments = (nStepSolver + CKPT_SEGMENT_SIZE - 1) / CKPT_SEGMENT_SIZE;
    float ckpt_x[CKPT_MAX_SEGMENTS + 1];
    ckpt_x[0] = px;

    int ckpt_idx = 1;
    for (int t = 0; t < nStepSolver; t++) {
        cpab_forward_step_1D_adaptive(
            px, As, As_batch_offset, h, h_half, ncx, adaptive_substeps);

        if ((t + 1) % CKPT_SEGMENT_SIZE == 0) {
            ckpt_x[ckpt_idx] = px;
            ckpt_idx++;
        }
    }

    const int moment_batch_offset = batch_index * nC * 2;
    const int grad_idx = batch_index * nP + point_index;
    float lam = grad_output[grad_idx];

    int prev_cell = -1;
    float local_m[2] = {0.0f, 0.0f};

    float seg_x[CKPT_SEGMENT_SIZE + 1];
    int seg_cells[CKPT_SEGMENT_SIZE];

    for (int seg = n_segments - 1; seg >= 0; seg--) {
        const int seg_start = seg * CKPT_SEGMENT_SIZE;
        const int seg_end = min(seg_start + CKPT_SEGMENT_SIZE, nStepSolver);
        const int seg_len = seg_end - seg_start;

        px = ckpt_x[seg];
        seg_x[0] = px;

        for (int s = 0; s < seg_len; s++) {
            seg_cells[s] = cuda_findcellidx_1D_fast(px, ncx);
            cpab_forward_step_1D_adaptive(
                px, As, As_batch_offset, h, h_half, ncx, adaptive_substeps);
            seg_x[s + 1] = px;
        }

        for (int s = seg_len - 1; s >= 0; s--) {
            px = seg_x[s];
            const int cellidx = seg_cells[s];

            if (cellidx != prev_cell && prev_cell >= 0) {
                warp_aggregated_moment_flush_2(
                    moments + moment_batch_offset, prev_cell, local_m);
                local_m[0] = 0.0f;
                local_m[1] = 0.0f;
            }
            prev_cell = cellidx;

            const float* A = As + As_batch_offset + 2 * cellidx;
            cpab_accumulate_moments_and_adjoint_1D(
                A, px, h, h_half, local_m, lam);
        }
    }

    if (prev_cell >= 0) {
        warp_aggregated_moment_flush_2(moments + moment_batch_offset, prev_cell, local_m);
    }

    if (grad_points != nullptr) {
        grad_points[grad_idx] = lam;
    }
}

// ============================================================================
// MOMENTS -> GRADIENT PROJECTION
// ============================================================================

// ============================================================================
// OPTIMIZED 2x2 MATRIX EXPONENTIAL KERNEL
// ============================================================================

/**
 * Compute matrix exponential for 1D affine CPAB velocity matrices.
 *
 * Input matrices have the form:
 *   M = dT * [ A[0] A[1] ]
 *            [  0    0   ]
 *
 * exp(M) top row is:
 *   [exp(m00), m01 * exprel(m00)] where exprel(x) = (exp(x)-1)/x.
 */
__global__ void cpab_cuda_kernel_expm2x2(
    const int n_matrices,
    const float dT,
    float* __restrict__ Trels_out,   // [n_matrices, 2]
    const float* __restrict__ As_in) // [n_matrices, 2]
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_matrices) return;

    const float* A = As_in + idx * 2;
    const float m00 = dT * A[0];
    const float m01 = dT * A[1];

    const float e00 = expf(m00);
    float exprel;
    if (fabsf(m00) > 1e-6f) {
        exprel = expm1f(m00) / m00;
    } else {
        const float m00_2 = m00 * m00;
        exprel = 1.0f + 0.5f * m00 + (1.0f / 6.0f) * m00_2 + (1.0f / 24.0f) * m00_2 * m00;
    }

    float* out = Trels_out + idx * 2;
    out[0] = e00;
    out[1] = m01 * exprel;
}
