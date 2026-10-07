#include <cuda.h>
#include <cuda_runtime.h>
#include "cpab_ops_3d.cuh"

// ============================================================================
// WARP-AGGREGATED ATOMIC ADD FOR 12-ELEMENT MOMENTS
// ============================================================================

__device__ __forceinline__ void warp_aggregated_moment_flush_12(
    float* __restrict__ moments_base,
    const int cell,
    const float local_m[12])
{
    const unsigned active = __activemask();
    const unsigned match = __match_any_sync(active, cell);
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(match) - 1;

    if (match == (1u << lane)) {
        const int idx = cell * 12;
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            atomicAdd(&moments_base[idx + i], local_m[i]);
        }
        return;
    }

    float sums[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) sums[i] = 0.0f;

    unsigned peers = match;
    while (peers) {
        const int src = __ffs(peers) - 1;
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            const float v = __shfl_sync(match, local_m[i], src);
            if (lane == leader) sums[i] += v;
        }
        peers &= peers - 1;
    }

    if (lane == leader) {
        const int idx = cell * 12;
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            atomicAdd(&moments_base[idx + i], sums[i]);
        }
    }
}

// ============================================================================
// CELL INDEX LOOKUP
// ============================================================================

/**
 * Fast cell index lookup for 3D CPAB tessellation.
 * Returns tetrahedron index in [0, 5*nx*ny*nz).
 *
 * Outside the domain a point uses the cell that owns the nearest boundary point,
 * which the clamping below finds beyond a face or an edge. Beyond a corner, the
 * cells that own the corner's three edges meet, and each pair of them agrees on
 * the plane where their two overshoots past the domain, in cell units, are equal.
 * So the point goes to the edge along the axis with the smallest overshoot: that
 * coordinate is moved half a cell inside the domain and the other two are clamped.
 */
__device__ __forceinline__ int cuda_findcellidx_3D_fast(
    float px,
    float py,
    float pz,
    const int nx,
    const int ny,
    const int nz)
{
    const float ox = (fabsf(px - 0.5f) - 0.5f) * (float)nx;
    const float oy = (fabsf(py - 0.5f) - 0.5f) * (float)ny;
    const float oz = (fabsf(pz - 0.5f) - 0.5f) * (float)nz;

    if (ox > 0.0f && oy > 0.0f && oz > 0.0f) {
        if (ox <= oy && ox <= oz) {
            px = (px < 0.5f) ? 0.5f / (float)nx : 1.0f - 0.5f / (float)nx;
        } else if (oy <= oz) {
            py = (py < 0.5f) ? 0.5f / (float)ny : 1.0f - 0.5f / (float)ny;
        } else {
            pz = (pz < 0.5f) ? 0.5f / (float)nz : 1.0f - 0.5f / (float)nz;
        }
    }

    // Clamp onto the boundary itself, not to 1 - eps: a point clamped just inside a
    // face can land in the central tetrahedron next to the face diagonal, whose
    // extension outside is not continuous. The min() below keeps the index in range.
    const float x_clamped = fminf(fmaxf(px, 0.0f), 1.0f);
    const float y_clamped = fminf(fmaxf(py, 0.0f), 1.0f);
    const float z_clamped = fminf(fmaxf(pz, 0.0f), 1.0f);

    const float sx = x_clamped * (float)nx;
    const float sy = y_clamped * (float)ny;
    const float sz = z_clamped * (float)nz;

    int i = __float2int_rd(sx);
    int j = __float2int_rd(sy);
    int k = __float2int_rd(sz);

    i = min(i, nx - 1);
    j = min(j, ny - 1);
    k = min(k, nz - 1);

    float x = sx - (float)i;
    float y = sy - (float)j;
    float z = sz - (float)k;

    int cell_idx = 5 * (i + j * nx + k * nx * ny);

    // Parity-dependent local coordinate remapping (same pattern as original).
    if (((i ^ j ^ k) & 1) != 0) {
        const float tmp = x;
        x = y;
        y = 1.0f - tmp;
    }

    if (-x - y + z >= 0.0f) {
        cell_idx += 1;
    } else if (x + y + z - 2.0f >= 0.0f) {
        cell_idx += 2;
    } else if (-x + y - z >= 0.0f) {
        cell_idx += 3;
    } else if (x - y - z >= 0.0f) {
        cell_idx += 4;
    }
    return cell_idx;
}

// ============================================================================
// FORWARD KERNEL
// ============================================================================

__device__ __forceinline__ void cpab_midpoint_step_3D_cell(
    float& px,
    float& py,
    float& pz,
    const float* __restrict__ As,
    const int As_batch_offset,
    const int cellidx,
    const float dt)
{
    const float* A = As + As_batch_offset + 12 * cellidx;
    const float half_dt = 0.5f * dt;

    const float vx = __fmaf_rn(A[0], px, __fmaf_rn(A[1], py, __fmaf_rn(A[2], pz, A[3])));
    const float vy = __fmaf_rn(A[4], px, __fmaf_rn(A[5], py, __fmaf_rn(A[6], pz, A[7])));
    const float vz = __fmaf_rn(A[8], px, __fmaf_rn(A[9], py, __fmaf_rn(A[10], pz, A[11])));

    const float pMid_x = __fmaf_rn(half_dt, vx, px);
    const float pMid_y = __fmaf_rn(half_dt, vy, py);
    const float pMid_z = __fmaf_rn(half_dt, vz, pz);

    const float vMid_x = __fmaf_rn(
        A[0], pMid_x, __fmaf_rn(A[1], pMid_y, __fmaf_rn(A[2], pMid_z, A[3])));
    const float vMid_y = __fmaf_rn(
        A[4], pMid_x, __fmaf_rn(A[5], pMid_y, __fmaf_rn(A[6], pMid_z, A[7])));
    const float vMid_z = __fmaf_rn(
        A[8], pMid_x, __fmaf_rn(A[9], pMid_y, __fmaf_rn(A[10], pMid_z, A[11])));

    px = __fmaf_rn(dt, vMid_x, px);
    py = __fmaf_rn(dt, vMid_y, py);
    pz = __fmaf_rn(dt, vMid_z, pz);
}

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
    const int broadcast)
{
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (point_index < nP && batch_index < batch_size) {
        const int input_offset = broadcast * batch_index * nP * 3;
        float px = points[input_offset + point_index];
        float py = points[input_offset + point_index + nP];
        float pz = points[input_offset + point_index + 2 * nP];

        const int nC = 5 * ncx * ncy * ncz;
        const int trels_batch_offset = batch_index * 12 * nC;
        const int As_batch_offset = batch_index * 12 * nC;
        const float h = 1.0f / (float)nStepSolver;

        #pragma unroll 4
        for (int n = 0; n < nStepSolver; n++) {
            const int cellidx_start = cuda_findcellidx_3D_fast(px, py, pz, ncx, ncy, ncz);
            const float* T = Trels + trels_batch_offset + 12 * cellidx_start;

            const float step_px = __fmaf_rn(
                T[0], px, __fmaf_rn(T[1], py, __fmaf_rn(T[2], pz, T[3])));
            const float step_py = __fmaf_rn(
                T[4], px, __fmaf_rn(T[5], py, __fmaf_rn(T[6], pz, T[7])));
            const float step_pz = __fmaf_rn(
                T[8], px, __fmaf_rn(T[9], py, __fmaf_rn(T[10], pz, T[11])));

            if (adaptive_substeps != 0) {
                const int cellidx_end = cuda_findcellidx_3D_fast(step_px, step_py, step_pz, ncx, ncy, ncz);
                if (cellidx_end != cellidx_start) {
                    // Fast crossing correction (Heun/trapezoid): one start/end velocity average.
                    const float* A0 = As + As_batch_offset + 12 * cellidx_start;
                    const float v0x = __fmaf_rn(A0[0], px, __fmaf_rn(A0[1], py, __fmaf_rn(A0[2], pz, A0[3])));
                    const float v0y = __fmaf_rn(A0[4], px, __fmaf_rn(A0[5], py, __fmaf_rn(A0[6], pz, A0[7])));
                    const float v0z = __fmaf_rn(A0[8], px, __fmaf_rn(A0[9], py, __fmaf_rn(A0[10], pz, A0[11])));

                    const float* A1 = As + As_batch_offset + 12 * cellidx_end;
                    const float v1x = __fmaf_rn(A1[0], step_px, __fmaf_rn(A1[1], step_py, __fmaf_rn(A1[2], step_pz, A1[3])));
                    const float v1y = __fmaf_rn(A1[4], step_px, __fmaf_rn(A1[5], step_py, __fmaf_rn(A1[6], step_pz, A1[7])));
                    const float v1z = __fmaf_rn(A1[8], step_px, __fmaf_rn(A1[9], step_py, __fmaf_rn(A1[10], step_pz, A1[11])));

                    const float h_half = 0.5f * h;
                    px = __fmaf_rn(h_half, v0x + v1x, px);
                    py = __fmaf_rn(h_half, v0y + v1y, py);
                    pz = __fmaf_rn(h_half, v0z + v1z, pz);
                    continue;
                }
            }

            px = step_px;
            py = step_py;
            pz = step_pz;
        }

        const int output_offset = 3 * nP * batch_index;
        newpoints[output_offset + point_index] = px;
        newpoints[output_offset + point_index + nP] = py;
        newpoints[output_offset + point_index + 2 * nP] = pz;
    }
}

// ============================================================================
// FUSED BACKWARD KERNEL — Segmented checkpointing (3D)
// ============================================================================

__device__ __forceinline__ int cpab_forward_step_3D(
    float& px,
    float& py,
    float& pz,
    const float* __restrict__ As,
    const int As_batch_offset,
    const float h,
    const float h_half,
    const int ncx,
    const int ncy,
    const int ncz)
{
    const int cellidx = cuda_findcellidx_3D_fast(px, py, pz, ncx, ncy, ncz);
    const float* A = As + As_batch_offset + 12 * cellidx;

    const float vx = __fmaf_rn(A[0], px, __fmaf_rn(A[1], py, __fmaf_rn(A[2], pz, A[3])));
    const float vy = __fmaf_rn(A[4], px, __fmaf_rn(A[5], py, __fmaf_rn(A[6], pz, A[7])));
    const float vz = __fmaf_rn(A[8], px, __fmaf_rn(A[9], py, __fmaf_rn(A[10], pz, A[11])));

    const float pMid_x = __fmaf_rn(h_half, vx, px);
    const float pMid_y = __fmaf_rn(h_half, vy, py);
    const float pMid_z = __fmaf_rn(h_half, vz, pz);

    const float vMid_x = __fmaf_rn(
        A[0], pMid_x, __fmaf_rn(A[1], pMid_y, __fmaf_rn(A[2], pMid_z, A[3])));
    const float vMid_y = __fmaf_rn(
        A[4], pMid_x, __fmaf_rn(A[5], pMid_y, __fmaf_rn(A[6], pMid_z, A[7])));
    const float vMid_z = __fmaf_rn(
        A[8], pMid_x, __fmaf_rn(A[9], pMid_y, __fmaf_rn(A[10], pMid_z, A[11])));

    px = __fmaf_rn(h, vMid_x, px);
    py = __fmaf_rn(h, vMid_y, py);
    pz = __fmaf_rn(h, vMid_z, pz);

    return cellidx;
}

__device__ __forceinline__ int cpab_forward_step_3D_adaptive(
    float& px,
    float& py,
    float& pz,
    const float* __restrict__ As,
    const int As_batch_offset,
    const float h,
    const float h_half,
    const int ncx,
    const int ncy,
    const int ncz,
    const int adaptive_substeps)
{
    if (adaptive_substeps == 0) {
        return cpab_forward_step_3D(px, py, pz, As, As_batch_offset, h, h_half, ncx, ncy, ncz);
    }

    const int cellidx_start = cuda_findcellidx_3D_fast(px, py, pz, ncx, ncy, ncz);
    float step_px = px;
    float step_py = py;
    float step_pz = pz;
    cpab_midpoint_step_3D_cell(step_px, step_py, step_pz, As, As_batch_offset, cellidx_start, h);

    const int cellidx_end = cuda_findcellidx_3D_fast(step_px, step_py, step_pz, ncx, ncy, ncz);
    if (cellidx_end == cellidx_start) {
        px = step_px;
        py = step_py;
        pz = step_pz;
        return cellidx_start;
    }

    const float* A0 = As + As_batch_offset + 12 * cellidx_start;
    const float v0x = __fmaf_rn(A0[0], px, __fmaf_rn(A0[1], py, __fmaf_rn(A0[2], pz, A0[3])));
    const float v0y = __fmaf_rn(A0[4], px, __fmaf_rn(A0[5], py, __fmaf_rn(A0[6], pz, A0[7])));
    const float v0z = __fmaf_rn(A0[8], px, __fmaf_rn(A0[9], py, __fmaf_rn(A0[10], pz, A0[11])));

    const float* A1 = As + As_batch_offset + 12 * cellidx_end;
    const float v1x = __fmaf_rn(A1[0], step_px, __fmaf_rn(A1[1], step_py, __fmaf_rn(A1[2], step_pz, A1[3])));
    const float v1y = __fmaf_rn(A1[4], step_px, __fmaf_rn(A1[5], step_py, __fmaf_rn(A1[6], step_pz, A1[7])));
    const float v1z = __fmaf_rn(A1[8], step_px, __fmaf_rn(A1[9], step_py, __fmaf_rn(A1[10], step_pz, A1[11])));

    px = __fmaf_rn(h_half, v0x + v1x, px);
    py = __fmaf_rn(h_half, v0y + v1y, py);
    pz = __fmaf_rn(h_half, v0z + v1z, pz);
    return cellidx_start;
}

__device__ __forceinline__ void cpab_accumulate_moments_and_adjoint_3D(
    const float* __restrict__ A,
    const float px,
    const float py,
    const float pz,
    const float h,
    const float h_half,
    float* __restrict__ local_m,
    float& lam_x,
    float& lam_y,
    float& lam_z)
{
    const float A0 = A[0], A1 = A[1], A2 = A[2], A3 = A[3];
    const float A4 = A[4], A5 = A[5], A6 = A[6], A7 = A[7];
    const float A8 = A[8], A9 = A[9], A10 = A[10], A11 = A[11];

    const float vx = __fmaf_rn(A0, px, __fmaf_rn(A1, py, __fmaf_rn(A2, pz, A3)));
    const float vy = __fmaf_rn(A4, px, __fmaf_rn(A5, py, __fmaf_rn(A6, pz, A7)));
    const float vz = __fmaf_rn(A8, px, __fmaf_rn(A9, py, __fmaf_rn(A10, pz, A11)));

    const float pMid_x = __fmaf_rn(h_half, vx, px);
    const float pMid_y = __fmaf_rn(h_half, vy, py);
    const float pMid_z = __fmaf_rn(h_half, vz, pz);

    float At_lam_x = A0 * lam_x + A4 * lam_y + A8 * lam_z;
    float At_lam_y = A1 * lam_x + A5 * lam_y + A9 * lam_z;
    float At_lam_z = A2 * lam_x + A6 * lam_y + A10 * lam_z;

    const float w_pMid_x = h * lam_x;
    const float w_pMid_y = h * lam_y;
    const float w_pMid_z = h * lam_z;
    const float w_p_x = h_half * h * At_lam_x;
    const float w_p_y = h_half * h * At_lam_y;
    const float w_p_z = h_half * h * At_lam_z;

    local_m[0] += w_pMid_x * pMid_x;
    local_m[1] += w_pMid_x * pMid_y;
    local_m[2] += w_pMid_x * pMid_z;
    local_m[3] += w_pMid_x;
    local_m[4] += w_pMid_y * pMid_x;
    local_m[5] += w_pMid_y * pMid_y;
    local_m[6] += w_pMid_y * pMid_z;
    local_m[7] += w_pMid_y;
    local_m[8] += w_pMid_z * pMid_x;
    local_m[9] += w_pMid_z * pMid_y;
    local_m[10] += w_pMid_z * pMid_z;
    local_m[11] += w_pMid_z;

    local_m[0] += w_p_x * px;
    local_m[1] += w_p_x * py;
    local_m[2] += w_p_x * pz;
    local_m[3] += w_p_x;
    local_m[4] += w_p_y * px;
    local_m[5] += w_p_y * py;
    local_m[6] += w_p_y * pz;
    local_m[7] += w_p_y;
    local_m[8] += w_p_z * px;
    local_m[9] += w_p_z * py;
    local_m[10] += w_p_z * pz;
    local_m[11] += w_p_z;

    const float lam_mid_x = lam_x + h_half * At_lam_x;
    const float lam_mid_y = lam_y + h_half * At_lam_y;
    const float lam_mid_z = lam_z + h_half * At_lam_z;

    At_lam_x = A0 * lam_mid_x + A4 * lam_mid_y + A8 * lam_mid_z;
    At_lam_y = A1 * lam_mid_x + A5 * lam_mid_y + A9 * lam_mid_z;
    At_lam_z = A2 * lam_mid_x + A6 * lam_mid_y + A10 * lam_mid_z;

    lam_x += h * At_lam_x;
    lam_y += h * At_lam_y;
    lam_z += h * At_lam_z;
}

__global__ void cpab_cuda_kernel_fused_backward_3D(
    const int nP,
    const int n_theta,
    const int nC,
    const int nStepSolver,
    float* __restrict__ moments,           // [n_theta, nC, 12]
    float* __restrict__ grad_points,       // [n_theta, 3, nP] or nullptr
    const float* __restrict__ grad_output, // [n_theta, 3, nP]
    const float* __restrict__ points,
    const float* __restrict__ As,
    const int ncx,
    const int ncy,
    const int ncz,
    const int adaptive_substeps,
    const int broadcast)
{
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_index = blockIdx.y;

    if (point_index >= nP || batch_index >= n_theta) return;

    const float h = 1.0f / (float)nStepSolver;
    const float h_half = 0.5f * h;
    const int As_batch_offset = batch_index * 12 * nC;

    const int input_offset = broadcast * batch_index * nP * 3;
    float px = points[input_offset + point_index];
    float py = points[input_offset + point_index + nP];
    float pz = points[input_offset + point_index + 2 * nP];

    const int n_segments = (nStepSolver + CKPT_SEGMENT_SIZE - 1) / CKPT_SEGMENT_SIZE;

    float ckpt_x[CKPT_MAX_SEGMENTS + 1];
    float ckpt_y[CKPT_MAX_SEGMENTS + 1];
    float ckpt_z[CKPT_MAX_SEGMENTS + 1];

    ckpt_x[0] = px;
    ckpt_y[0] = py;
    ckpt_z[0] = pz;

    int ckpt_idx = 1;
    for (int t = 0; t < nStepSolver; t++) {
        cpab_forward_step_3D_adaptive(
            px, py, pz,
            As, As_batch_offset,
            h, h_half,
            ncx, ncy, ncz,
            adaptive_substeps);
        if ((t + 1) % CKPT_SEGMENT_SIZE == 0) {
            ckpt_x[ckpt_idx] = px;
            ckpt_y[ckpt_idx] = py;
            ckpt_z[ckpt_idx] = pz;
            ckpt_idx++;
        }
    }

    const int moment_batch_offset = batch_index * nC * 12;
    const int grad_idx = batch_index * 3 * nP + point_index;
    float lam_x = grad_output[grad_idx];
    float lam_y = grad_output[grad_idx + nP];
    float lam_z = grad_output[grad_idx + 2 * nP];

    int prev_cell = -1;
    float local_m[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) local_m[i] = 0.0f;

    float seg_x[CKPT_SEGMENT_SIZE + 1];
    float seg_y[CKPT_SEGMENT_SIZE + 1];
    float seg_z[CKPT_SEGMENT_SIZE + 1];
    int seg_cells[CKPT_SEGMENT_SIZE];
    for (int seg = n_segments - 1; seg >= 0; seg--) {
        const int seg_start = seg * CKPT_SEGMENT_SIZE;
        const int seg_end = min(seg_start + CKPT_SEGMENT_SIZE, nStepSolver);
        const int seg_len = seg_end - seg_start;

        px = ckpt_x[seg];
        py = ckpt_y[seg];
        pz = ckpt_z[seg];
        seg_x[0] = px;
        seg_y[0] = py;
        seg_z[0] = pz;

        for (int s = 0; s < seg_len; s++) {
            seg_cells[s] = cpab_forward_step_3D_adaptive(
                px, py, pz,
                As, As_batch_offset,
                h, h_half,
                ncx, ncy, ncz,
                adaptive_substeps);
            seg_x[s + 1] = px;
            seg_y[s + 1] = py;
            seg_z[s + 1] = pz;
        }

        for (int s = seg_len - 1; s >= 0; s--) {
            px = seg_x[s];
            py = seg_y[s];
            pz = seg_z[s];
            const int cellidx = seg_cells[s];

            if (cellidx != prev_cell && prev_cell >= 0) {
                warp_aggregated_moment_flush_12(
                    moments + moment_batch_offset, prev_cell, local_m);
                #pragma unroll
                for (int i = 0; i < 12; i++) local_m[i] = 0.0f;
            }
            prev_cell = cellidx;

            const float* A = As + As_batch_offset + 12 * cellidx;
            cpab_accumulate_moments_and_adjoint_3D(
                A,
                px, py, pz,
                h, h_half,
                local_m, lam_x, lam_y, lam_z);
        }
    }

    if (prev_cell >= 0) {
        warp_aggregated_moment_flush_12(moments + moment_batch_offset, prev_cell, local_m);
    }

    if (grad_points != nullptr) {
        grad_points[grad_idx] = lam_x;
        grad_points[grad_idx + nP] = lam_y;
        grad_points[grad_idx + 2 * nP] = lam_z;
    }
}

// ============================================================================
// OPTIMIZED 4x4 MATRIX EXPONENTIAL KERNEL (AFFINE 3D STRUCTURE)
// ============================================================================

__global__ void cpab_cuda_kernel_expm4x4(
    const int n_matrices,
    const float dT,
    float* __restrict__ Trels_out,
    const float* __restrict__ As_in)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_matrices) return;

    const float* A = As_in + idx * 12;

    float a00 = dT * A[0],  a01 = dT * A[1],  a02 = dT * A[2],  a03 = dT * A[3];
    float a10 = dT * A[4],  a11 = dT * A[5],  a12 = dT * A[6],  a13 = dT * A[7];
    float a20 = dT * A[8],  a21 = dT * A[9],  a22 = dT * A[10], a23 = dT * A[11];

    const float norm_sq =
        a00*a00 + a01*a01 + a02*a02 + a03*a03 +
        a10*a10 + a11*a11 + a12*a12 + a13*a13 +
        a20*a20 + a21*a21 + a22*a22 + a23*a23;
    const float norm = sqrtf(norm_sq);

    int n_squarings = 0;
    if (norm > 0.5f) {
        n_squarings = (int)ceilf(log2f(norm * 2.0f));
    }

    float scale = 1.0f;
    for (int i = 0; i < n_squarings; i++) scale *= 0.5f;

    float m00 = a00 * scale, m01 = a01 * scale, m02 = a02 * scale, m03 = a03 * scale;
    float m10 = a10 * scale, m11 = a11 * scale, m12 = a12 * scale, m13 = a13 * scale;
    float m20 = a20 * scale, m21 = a21 * scale, m22 = a22 * scale, m23 = a23 * scale;

    float mm00 = m00*m00 + m01*m10 + m02*m20;
    float mm01 = m00*m01 + m01*m11 + m02*m21;
    float mm02 = m00*m02 + m01*m12 + m02*m22;
    float mm10 = m10*m00 + m11*m10 + m12*m20;
    float mm11 = m10*m01 + m11*m11 + m12*m21;
    float mm12 = m10*m02 + m11*m12 + m12*m22;
    float mm20 = m20*m00 + m21*m10 + m22*m20;
    float mm21 = m20*m01 + m21*m11 + m22*m21;
    float mm22 = m20*m02 + m21*m12 + m22*m22;

    float mt0 = m00*m03 + m01*m13 + m02*m23;
    float mt1 = m10*m03 + m11*m13 + m12*m23;
    float mt2 = m20*m03 + m21*m13 + m22*m23;

    const float inv12 = 1.0f / 12.0f;
    const float half = 0.5f;

    float p00 = 1.0f + half*m00 + inv12*mm00;
    float p01 =        half*m01 + inv12*mm01;
    float p02 =        half*m02 + inv12*mm02;
    float p10 =        half*m10 + inv12*mm10;
    float p11 = 1.0f + half*m11 + inv12*mm11;
    float p12 =        half*m12 + inv12*mm12;
    float p20 =        half*m20 + inv12*mm20;
    float p21 =        half*m21 + inv12*mm21;
    float p22 = 1.0f + half*m22 + inv12*mm22;

    float pt0 = half*m03 + inv12*mt0;
    float pt1 = half*m13 + inv12*mt1;
    float pt2 = half*m23 + inv12*mt2;

    float q00 = 1.0f - half*m00 + inv12*mm00;
    float q01 =      - half*m01 + inv12*mm01;
    float q02 =      - half*m02 + inv12*mm02;
    float q10 =      - half*m10 + inv12*mm10;
    float q11 = 1.0f - half*m11 + inv12*mm11;
    float q12 =      - half*m12 + inv12*mm12;
    float q20 =      - half*m20 + inv12*mm20;
    float q21 =      - half*m21 + inv12*mm21;
    float q22 = 1.0f - half*m22 + inv12*mm22;

    float qt0 = -half*m03 + inv12*mt0;
    float qt1 = -half*m13 + inv12*mt1;
    float qt2 = -half*m23 + inv12*mt2;

    const float det =
        q00 * (q11 * q22 - q12 * q21) -
        q01 * (q10 * q22 - q12 * q20) +
        q02 * (q10 * q21 - q11 * q20);
    const float inv_det = 1.0f / det;

    float qi00 =  (q11*q22 - q12*q21) * inv_det;
    float qi01 =  (q02*q21 - q01*q22) * inv_det;
    float qi02 =  (q01*q12 - q02*q11) * inv_det;
    float qi10 =  (q12*q20 - q10*q22) * inv_det;
    float qi11 =  (q00*q22 - q02*q20) * inv_det;
    float qi12 =  (q02*q10 - q00*q12) * inv_det;
    float qi20 =  (q10*q21 - q11*q20) * inv_det;
    float qi21 =  (q01*q20 - q00*q21) * inv_det;
    float qi22 =  (q00*q11 - q01*q10) * inv_det;

    float qit0 = -(qi00 * qt0 + qi01 * qt1 + qi02 * qt2);
    float qit1 = -(qi10 * qt0 + qi11 * qt1 + qi12 * qt2);
    float qit2 = -(qi20 * qt0 + qi21 * qt1 + qi22 * qt2);

    float r00 = qi00*p00 + qi01*p10 + qi02*p20;
    float r01 = qi00*p01 + qi01*p11 + qi02*p21;
    float r02 = qi00*p02 + qi01*p12 + qi02*p22;
    float r10 = qi10*p00 + qi11*p10 + qi12*p20;
    float r11 = qi10*p01 + qi11*p11 + qi12*p21;
    float r12 = qi10*p02 + qi11*p12 + qi12*p22;
    float r20 = qi20*p00 + qi21*p10 + qi22*p20;
    float r21 = qi20*p01 + qi21*p11 + qi22*p21;
    float r22 = qi20*p02 + qi21*p12 + qi22*p22;

    float r03 = qi00*pt0 + qi01*pt1 + qi02*pt2 + qit0;
    float r13 = qi10*pt0 + qi11*pt1 + qi12*pt2 + qit1;
    float r23 = qi20*pt0 + qi21*pt1 + qi22*pt2 + qit2;

    for (int i = 0; i < n_squarings; i++) {
        float new_r00 = r00*r00 + r01*r10 + r02*r20;
        float new_r01 = r00*r01 + r01*r11 + r02*r21;
        float new_r02 = r00*r02 + r01*r12 + r02*r22;
        float new_r10 = r10*r00 + r11*r10 + r12*r20;
        float new_r11 = r10*r01 + r11*r11 + r12*r21;
        float new_r12 = r10*r02 + r11*r12 + r12*r22;
        float new_r20 = r20*r00 + r21*r10 + r22*r20;
        float new_r21 = r20*r01 + r21*r11 + r22*r21;
        float new_r22 = r20*r02 + r21*r12 + r22*r22;

        float new_r03 = r00*r03 + r01*r13 + r02*r23 + r03;
        float new_r13 = r10*r03 + r11*r13 + r12*r23 + r13;
        float new_r23 = r20*r03 + r21*r13 + r22*r23 + r23;

        r00 = new_r00; r01 = new_r01; r02 = new_r02; r03 = new_r03;
        r10 = new_r10; r11 = new_r11; r12 = new_r12; r13 = new_r13;
        r20 = new_r20; r21 = new_r21; r22 = new_r22; r23 = new_r23;
    }

    float* out = Trels_out + idx * 12;
    out[0] = r00; out[1] = r01; out[2] = r02; out[3] = r03;
    out[4] = r10; out[5] = r11; out[6] = r12; out[7] = r13;
    out[8] = r20; out[9] = r21; out[10] = r22; out[11] = r23;
}
