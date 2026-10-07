#include <cuda.h>
#include <cuda_runtime.h>
#include "cpab_ops_2d.cuh"


__device__ __forceinline__ void warp_aggregated_moment_flush(
    float* __restrict__ moments_base,   // &moments[moment_batch_offset]
    int cell,
    const float local_m[6])
{
    const unsigned active = __activemask();
    const unsigned match  = __match_any_sync(active, cell);
    const int      lane   = threadIdx.x & 31;
    const int      leader = __ffs(match) - 1;   // lowest lane in group

    // Fast path: if this thread is the only one targeting this cell, just atomicAdd directly.
    if (match == (1u << lane)) {
        const int idx = cell * 6;
        atomicAdd(&moments_base[idx + 0], local_m[0]);
        atomicAdd(&moments_base[idx + 1], local_m[1]);
        atomicAdd(&moments_base[idx + 2], local_m[2]);
        atomicAdd(&moments_base[idx + 3], local_m[3]);
        atomicAdd(&moments_base[idx + 4], local_m[4]);
        atomicAdd(&moments_base[idx + 5], local_m[5]);
        return;
    }

    // Collect values from all matching peers into the leader.
    // All threads in the match group call __shfl_sync together;
    // only the leader accumulates.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    float s3 = 0.0f, s4 = 0.0f, s5 = 0.0f;

    unsigned peers = match;
    while (peers) {
        const int src = __ffs(peers) - 1;
        const float v0 = __shfl_sync(match, local_m[0], src);
        const float v1 = __shfl_sync(match, local_m[1], src);
        const float v2 = __shfl_sync(match, local_m[2], src);
        const float v3 = __shfl_sync(match, local_m[3], src);
        const float v4 = __shfl_sync(match, local_m[4], src);
        const float v5 = __shfl_sync(match, local_m[5], src);
        if (lane == leader) {
            s0 += v0; s1 += v1; s2 += v2;
            s3 += v3; s4 += v4; s5 += v5;
        }
        peers &= peers - 1;          // clear lowest set bit
    }

    if (lane == leader) {
        const int idx = cell * 6;
        atomicAdd(&moments_base[idx + 0], s0);
        atomicAdd(&moments_base[idx + 1], s1);
        atomicAdd(&moments_base[idx + 2], s2);
        atomicAdd(&moments_base[idx + 3], s3);
        atomicAdd(&moments_base[idx + 4], s4);
        atomicAdd(&moments_base[idx + 5], s5);
    }
}

// ============================================================================
// CELL INDEX LOOKUP
// ============================================================================

/**
 * Fast cell index lookup for 2D CPAB tessellation.
 * 
 * The domain [0,1]² is divided into ncx × ncy cells, each split into 4 triangles.
 * Returns the triangle index (0 to 4*ncx*ncy - 1).
 */
__device__ __forceinline__ int cuda_findcellidx_2D_fast(
    const float px, const float py, 
    const int ncx, const int ncy) {
    
    const float inv_inc_x = (float)ncx;
    const float inv_inc_y = (float)ncy;
    
    // Clamp to [0, 1) range
    const float eps = 1e-7f;
    float x_clamped = fminf(fmaxf(px, 0.0f), 1.0f - eps);
    float y_clamped = fminf(fmaxf(py, 0.0f), 1.0f - eps);
    
    // Fast cell index computation using intrinsic floor
    int cell_x = __float2int_rd(x_clamped * inv_inc_x);
    int cell_y = __float2int_rd(y_clamped * inv_inc_y);
    
    cell_x = min(cell_x, ncx - 1);
    cell_y = min(cell_y, ncy - 1);
    
    // Local coordinates within cell [0, 1)
    float local_x = fmaf(x_clamped, inv_inc_x, -(float)cell_x);
    float local_y = fmaf(y_clamped, inv_inc_y, -(float)cell_y);
    
    // Base cell index (4 triangles per cell)
    int cell_idx = (cell_x + cell_y * ncx) * 4;
    
    // Handle out-of-bounds cases
    if (px <= 0.0f) {
        if (py <= 0.0f && py * inv_inc_y < px * inv_inc_x) {
            return cell_idx;
        } else if (py >= 1.0f && (py * inv_inc_y - ncy) > -px * inv_inc_x) {
            return cell_idx + 2;
        }
        return cell_idx + 3;
    }
    
    if (px >= 1.0f) {
        if (py <= 0.0f && -py * inv_inc_y > (px * inv_inc_x - ncx)) {
            return cell_idx;
        } else if (py >= 1.0f && (py * inv_inc_y - ncy) > (px * inv_inc_x - ncx)) {
            return cell_idx + 2;
        }
        return cell_idx + 1;
    }
    
    if (py <= 0.0f) return cell_idx;
    if (py >= 1.0f) return cell_idx + 2;
    
    // Inbound: branchless triangle selection
    //   bit1=1 when x<y, bit0=1 when (1-x)<y
    //   raw: 0→tri0, 1→tri1, 2→tri3, 3→tri2
    //   XOR lowest bit with bit1 to swap 2↔3
    const int bit1 = (int)(local_x < local_y);
    const int bit0 = (int)((1.0f - local_x) < local_y);
    int tri = (bit1 << 1) | bit0;
    tri ^= (tri >> 1) & 1;
    return cell_idx + tri;
}

// ============================================================================
// FORWARD KERNEL
// ============================================================================

__device__ __forceinline__ void cpab_midpoint_step_2D_cell(
    float& px,
    float& py,
    const float* __restrict__ As,
    const int As_batch_offset,
    const int cellidx,
    const float dt)
{
    const float* A = As + As_batch_offset + 6 * cellidx;
    const float half_dt = 0.5f * dt;

    const float vx = __fmaf_rn(A[0], px, __fmaf_rn(A[1], py, A[2]));
    const float vy = __fmaf_rn(A[3], px, __fmaf_rn(A[4], py, A[5]));

    const float pMid_x = __fmaf_rn(half_dt, vx, px);
    const float pMid_y = __fmaf_rn(half_dt, vy, py);

    const float vMid_x = __fmaf_rn(A[0], pMid_x, __fmaf_rn(A[1], pMid_y, A[2]));
    const float vMid_y = __fmaf_rn(A[3], pMid_x, __fmaf_rn(A[4], pMid_y, A[5]));

    px = __fmaf_rn(dt, vMid_x, px);
    py = __fmaf_rn(dt, vMid_y, py);
}

/**
 * Forward transformation kernel.
 * 
 * Transforms points through CPAB velocity field integration.
 * Uses midpoint method for numerical integration.
 */
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
    const int broadcast) {

    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (point_index < nP && batch_index < batch_size) {
        // Load point
        const int input_offset = broadcast * batch_index * nP * 2;
        float px = points[input_offset + point_index];
        float py = points[input_offset + point_index + nP];
    
        // Batch offsets for transformation matrices and velocity fields
        const int nC = 4 * ncx * ncy;
        const int trels_batch_offset = batch_index * 6 * nC;
        const int As_batch_offset = batch_index * 6 * nC;
        const float h = 1.0f / (float)nStepSolver;
        
        // Integration loop
        #pragma unroll 4
        for (int n = 0; n < nStepSolver; n++) {
            const int cellidx_start = cuda_findcellidx_2D_fast(px, py, ncx, ncy);
            const float* T = Trels + trels_batch_offset + 6 * cellidx_start;
            
            // Apply affine transformation: [x', y'] = T * [x, y, 1]
            const float step_px = __fmaf_rn(T[0], px, __fmaf_rn(T[1], py, T[2]));
            const float step_py = __fmaf_rn(T[3], px, __fmaf_rn(T[4], py, T[5]));

            if (adaptive_substeps != 0) {
                const int cellidx_end = cuda_findcellidx_2D_fast(step_px, step_py, ncx, ncy);
                if (cellidx_end != cellidx_start) {
                    // Fast crossing correction (Heun/trapezoid): one start/end velocity average.
                    const float* A0 = As + As_batch_offset + 6 * cellidx_start;
                    const float v0x = __fmaf_rn(A0[0], px, __fmaf_rn(A0[1], py, A0[2]));
                    const float v0y = __fmaf_rn(A0[3], px, __fmaf_rn(A0[4], py, A0[5]));

                    const float* A1 = As + As_batch_offset + 6 * cellidx_end;
                    const float v1x = __fmaf_rn(A1[0], step_px, __fmaf_rn(A1[1], step_py, A1[2]));
                    const float v1y = __fmaf_rn(A1[3], step_px, __fmaf_rn(A1[4], step_py, A1[5]));

                    const float h_half = 0.5f * h;
                    px = __fmaf_rn(h_half, v0x + v1x, px);
                    py = __fmaf_rn(h_half, v0y + v1y, py);
                    continue;
                }
            }
            
            px = step_px;
            py = step_py;
        }
    
        // Write output
        const int output_offset = 2 * nP * batch_index;
        newpoints[output_offset + point_index] = px;
        newpoints[output_offset + point_index + nP] = py;
    }
}

// ============================================================================
// FUSED BACKWARD KERNEL  —  Segmented Checkpointing
// ============================================================================

/**
 * Fused forward-checkpoint + adjoint-moment kernel.
 *
 * Instead of storing the FULL trajectory in per-thread local arrays
 * (which consumed ~1552 bytes/thread and dominated local-memory traffic),
 * this version uses **segmented checkpointing**:
 *
 *   Phase 1 — Forward pass saving only K = ceil(N/S) checkpoints,
 *             one every CKPT_SEGMENT_SIZE (S) steps.
 *             Storage: K × 2 floats = small.
 *
 *   Phase 2 — For each segment (last to first):
 *             (a) Recompute forward from checkpoint → rebuild S positions + cells
 *                 Storage: (S+1)×2 floats + S ints  (small, compile-time sized)
 *             (b) Walk backward through the segment, propagating λ and
 *                 accumulating moments — identical math to the original kernel.
 *
 * Memory comparison  (CPAB_MAX_STEPS = 128, CKPT_SEGMENT_SIZE = 8):
 *   Old:  traj_x[129] + traj_y[129] + traj_cells[128] = 1544 bytes  →  ptxas: 1552 B stack
 *   New:  ckpt_x[17] + ckpt_y[17] + seg_x[9] + seg_y[9] + seg_cells[8] = 240 bytes
 *
 * Extra compute: one additional forward pass (N steps) per thread, which
 * corresponds to ~10% of the current backward time.  The massive reduction
 * in local-memory pressure and improved occupancy more than compensate.
 *
 * Result: **bit-identical** gradients to the original kernel (same FP ops
 * in the same order within each segment).
 */

/**
 * Inline helper: one midpoint-method forward step.
 * Updates (px, py) in-place and returns the cell index.
 */
__device__ __forceinline__ int cpab_forward_step(
    float& px, float& py,
    const float* __restrict__ As,
    const int As_batch_offset,
    const float h, const float h_half,
    const int ncx, const int ncy)
{
    const int cellidx = cuda_findcellidx_2D_fast(px, py, ncx, ncy);
    const float* A = As + As_batch_offset + 6 * cellidx;

    float vx = __fmaf_rn(A[0], px, __fmaf_rn(A[1], py, A[2]));
    float vy = __fmaf_rn(A[3], px, __fmaf_rn(A[4], py, A[5]));

    float pMid_x = __fmaf_rn(h_half, vx, px);
    float pMid_y = __fmaf_rn(h_half, vy, py);

    float vMid_x = __fmaf_rn(A[0], pMid_x, __fmaf_rn(A[1], pMid_y, A[2]));
    float vMid_y = __fmaf_rn(A[3], pMid_x, __fmaf_rn(A[4], pMid_y, A[5]));

    px = __fmaf_rn(h, vMid_x, px);
    py = __fmaf_rn(h, vMid_y, py);

    return cellidx;
}

__device__ __forceinline__ void cpab_forward_step_adaptive(
    float& px,
    float& py,
    const float* __restrict__ As,
    const int As_batch_offset,
    const float h,
    const float h_half,
    const int ncx,
    const int ncy,
    const int adaptive_substeps)
{
    if (adaptive_substeps == 0) {
        cpab_forward_step(px, py, As, As_batch_offset, h, h_half, ncx, ncy);
        return;
    }

    const int cellidx_start = cuda_findcellidx_2D_fast(px, py, ncx, ncy);
    float step_px = px;
    float step_py = py;
    cpab_midpoint_step_2D_cell(step_px, step_py, As, As_batch_offset, cellidx_start, h);

    const int cellidx_end = cuda_findcellidx_2D_fast(step_px, step_py, ncx, ncy);
    if (cellidx_end == cellidx_start) {
        px = step_px;
        py = step_py;
        return;
    }

    const float* A0 = As + As_batch_offset + 6 * cellidx_start;
    const float v0x = __fmaf_rn(A0[0], px, __fmaf_rn(A0[1], py, A0[2]));
    const float v0y = __fmaf_rn(A0[3], px, __fmaf_rn(A0[4], py, A0[5]));

    const float* A1 = As + As_batch_offset + 6 * cellidx_end;
    const float v1x = __fmaf_rn(A1[0], step_px, __fmaf_rn(A1[1], step_py, A1[2]));
    const float v1y = __fmaf_rn(A1[3], step_px, __fmaf_rn(A1[4], step_py, A1[5]));

    px = __fmaf_rn(h_half, v0x + v1x, px);
    py = __fmaf_rn(h_half, v0y + v1y, py);
}

__device__ __forceinline__ void cpab_accumulate_moments_and_adjoint_2D(
    const float* __restrict__ A,
    const float px,
    const float py,
    const float h,
    const float h_half,
    float* __restrict__ local_m,
    float& lam_x,
    float& lam_y)
{
    const float A0 = A[0], A1 = A[1], A2 = A[2];
    const float A3 = A[3], A4 = A[4], A5 = A[5];

    const float vx = __fmaf_rn(A0, px, __fmaf_rn(A1, py, A2));
    const float vy = __fmaf_rn(A3, px, __fmaf_rn(A4, py, A5));
    const float pMid_x = __fmaf_rn(h_half, vx, px);
    const float pMid_y = __fmaf_rn(h_half, vy, py);

    float At_lam_x = A0 * lam_x + A3 * lam_y;
    float At_lam_y = A1 * lam_x + A4 * lam_y;

    const float w_pMid_x = h * lam_x;
    const float w_pMid_y = h * lam_y;
    const float w_p_x = h_half * h * At_lam_x;
    const float w_p_y = h_half * h * At_lam_y;

    local_m[0] += w_pMid_x * pMid_x;
    local_m[1] += w_pMid_x * pMid_y;
    local_m[2] += w_pMid_x;
    local_m[3] += w_pMid_y * pMid_x;
    local_m[4] += w_pMid_y * pMid_y;
    local_m[5] += w_pMid_y;

    local_m[0] += w_p_x * px;
    local_m[1] += w_p_x * py;
    local_m[2] += w_p_x;
    local_m[3] += w_p_y * px;
    local_m[4] += w_p_y * py;
    local_m[5] += w_p_y;

    const float lam_mid_x = lam_x + h_half * At_lam_x;
    const float lam_mid_y = lam_y + h_half * At_lam_y;

    At_lam_x = A0 * lam_mid_x + A3 * lam_mid_y;
    At_lam_y = A1 * lam_mid_x + A4 * lam_mid_y;

    lam_x += h * At_lam_x;
    lam_y += h * At_lam_y;
}

__global__ void cpab_cuda_kernel_fused_backward(
    const int nP,
    const int n_theta,
    const int nC,
    const int nStepSolver,
    float* __restrict__ moments,           // [n_theta, nC, 6]
    float* __restrict__ grad_points,       // [n_theta, 2, nP] or nullptr
    const float* __restrict__ grad_output, // [n_theta, 2, nP]
    const float* __restrict__ points,
    const float* __restrict__ As,
    const int ncx,
    const int ncy,
    const int adaptive_substeps,
    const int broadcast) {

    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_index = blockIdx.y;

    if (point_index >= nP || batch_index >= n_theta) return;

    const float h = 1.0f / (float)nStepSolver;
    const float h_half = h * 0.5f;
    const int As_batch_offset = batch_index * 6 * nC;

    // ---- Load initial point ----
    const int input_offset = broadcast * batch_index * nP * 2;
    float px = points[input_offset + point_index];
    float py = points[input_offset + point_index + nP];

    // ================================================================
    // Phase 1: Forward pass — save only checkpoints every S steps
    // ================================================================
    const int n_segments = (nStepSolver + CKPT_SEGMENT_SIZE - 1) / CKPT_SEGMENT_SIZE;

    // Checkpoint arrays — much smaller than full trajectory
    float ckpt_x[CKPT_MAX_SEGMENTS + 1];
    float ckpt_y[CKPT_MAX_SEGMENTS + 1];

    ckpt_x[0] = px;
    ckpt_y[0] = py;

    int ckpt_idx = 1;
    for (int t = 0; t < nStepSolver; t++) {
        cpab_forward_step_adaptive(
            px, py,
            As, As_batch_offset,
            h, h_half,
            ncx, ncy,
            adaptive_substeps);

        // Save checkpoint at segment boundaries
        if ((t + 1) % CKPT_SEGMENT_SIZE == 0) {
            ckpt_x[ckpt_idx] = px;
            ckpt_y[ckpt_idx] = py;
            ckpt_idx++;
        }
    }
    // If nStepSolver is not a multiple of CKPT_SEGMENT_SIZE, the last
    // partial segment's end is never checkpointed — that's fine, we
    // don't need it.

    // ================================================================
    // Phase 2: Backward pass — segment by segment, last to first
    // ================================================================
    const int moment_batch_offset = batch_index * nC * 6;
    const int grad_idx = batch_index * 2 * nP + point_index;
    float lam_x = grad_output[grad_idx];
    float lam_y = grad_output[grad_idx + nP];

    int prev_cell = -1;
    float local_m[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Small per-segment buffers (compile-time sized)
    float seg_x[CKPT_SEGMENT_SIZE + 1];
    float seg_y[CKPT_SEGMENT_SIZE + 1];
    int   seg_cells[CKPT_SEGMENT_SIZE];
    for (int seg = n_segments - 1; seg >= 0; seg--) {
        const int seg_start = seg * CKPT_SEGMENT_SIZE;
        const int seg_end   = min(seg_start + CKPT_SEGMENT_SIZE, nStepSolver);
        const int seg_len   = seg_end - seg_start;

        // ---- (a) Recompute forward from checkpoint to fill segment ----
        px = ckpt_x[seg];
        py = ckpt_y[seg];
        seg_x[0] = px;
        seg_y[0] = py;

        for (int s = 0; s < seg_len; s++) {
            seg_cells[s] = cuda_findcellidx_2D_fast(px, py, ncx, ncy);
            cpab_forward_step_adaptive(
                px, py,
                As, As_batch_offset,
                h, h_half,
                ncx, ncy,
                adaptive_substeps);
            seg_x[s + 1] = px;
            seg_y[s + 1] = py;
        }

        // ---- (b) Backward through this segment ----
        for (int s = seg_len - 1; s >= 0; s--) {
            px = seg_x[s];
            py = seg_y[s];
            const int cellidx = seg_cells[s];

            // Flush local buffer if cell changed
            if (cellidx != prev_cell && prev_cell >= 0) {
                warp_aggregated_moment_flush(moments + moment_batch_offset,
                                             prev_cell, local_m);
                local_m[0] = local_m[1] = local_m[2] =
                local_m[3] = local_m[4] = local_m[5] = 0.0f;
            }
            prev_cell = cellidx;

            const float* A = As + As_batch_offset + 6 * cellidx;
            cpab_accumulate_moments_and_adjoint_2D(
                A, px, py, h, h_half, local_m, lam_x, lam_y);
        }
    }

    // Flush remaining moments
    if (prev_cell >= 0) {
        warp_aggregated_moment_flush(moments + moment_batch_offset, prev_cell, local_m);
    }

    if (grad_points != nullptr) {
        grad_points[grad_idx] = lam_x;
        grad_points[grad_idx + nP] = lam_y;
    }
}

// ============================================================================
// OPTIMIZED 3x3 MATRIX EXPONENTIAL KERNEL
// ============================================================================

/**
 * Compute matrix exponential for 3x3 CPAB affine velocity matrices.
 *
 * Input matrices have the form:
 *   M = dT * [ A[0] A[1] A[2] ]   (row 0)
 *            [ A[3] A[4] A[5] ]   (row 1)
 *            [  0    0    0   ]   (row 2)
 *
 * Uses Padé(3,3) approximation with scaling and squaring.
 * For a matrix M, the Padé(3,3) approximant is:
 *   R_{3,3}(M) = (I - M/2 + M²/12)^{-1} * (I + M/2 + M²/12)
 * which equals [Q]^{-1} * [P] where Q = I - M/2 + M²/12, P = I + M/2 + M²/12.
 *
 * With scaling: we scale M by 2^(-s) so ||M/2^s|| < 0.5, compute Padé, then square s times.
 *
 * Since row 2 is always [0,0,1] in the result (affine structure), we only
 * compute and output the top 2 rows.
 */
__global__ void cpab_cuda_kernel_expm3x3(
    const int n_matrices,
    const float dT,
    float* __restrict__ Trels_out,   // [n_matrices, 6] output (top 2 rows, row-major)
    const float* __restrict__ As_in) // [n_matrices, 6] input velocity fields
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_matrices) return;

    // Load the 2x3 velocity field and scale by dT
    const float* A = As_in + idx * 6;
    float a00 = dT * A[0], a01 = dT * A[1], a02 = dT * A[2];
    float a10 = dT * A[3], a11 = dT * A[4], a12 = dT * A[5];
    // Row 2 is [0, 0, 0] (homogeneous coordinate row is zero in velocity)

    // Compute Frobenius norm of the 3x3 matrix (row 2 is zero)
    float norm_sq = a00*a00 + a01*a01 + a02*a02 + a10*a10 + a11*a11 + a12*a12;
    float norm = sqrtf(norm_sq);

    // Determine number of scalings: find s such that ||M/2^s|| <= 0.5
    // s = max(0, ceil(log2(norm / 0.5))) = max(0, ceil(log2(2*norm)))
    int n_squarings = 0;
    if (norm > 0.5f) {
        n_squarings = (int)ceilf(log2f(norm * 2.0f));
    }

    // Scale the matrix
    float scale = 1.0f;
    for (int i = 0; i < n_squarings; i++) scale *= 0.5f;
    float m00 = a00 * scale, m01 = a01 * scale, m02 = a02 * scale;
    float m10 = a10 * scale, m11 = a11 * scale, m12 = a12 * scale;
    // m20=0, m21=0, m22=0

    // Compute M² (3x3, but row 2 of M is zero)
    // M²[i][j] = Σ_k M[i][k] * M[k][j]
    // Since M[2][k] = 0, M²[2][j] = 0
    // Since M[k][2] only has m02, m12 (and m22=0):
    float mm00 = m00*m00 + m01*m10;             // M²[0][0]
    float mm01 = m00*m01 + m01*m11;             // M²[0][1]
    float mm02 = m00*m02 + m01*m12;             // M²[0][2] (m22=0)
    float mm10 = m10*m00 + m11*m10;             // M²[1][0]
    float mm11 = m10*m01 + m11*m11;             // M²[1][1]
    float mm12 = m10*m02 + m11*m12;             // M²[1][2]
    // mm20=mm21=mm22=0

    // Padé(3,3): P = I + M/2 + M²/12,  Q = I - M/2 + M²/12
    // For 3x3 with structure, compute P and Q (only rows 0,1 differ from identity pattern)
    float inv12 = 1.0f / 12.0f;
    float half = 0.5f;

    // P = I + M/2 + M²/12
    float p00 = 1.0f + half*m00 + inv12*mm00;
    float p01 =        half*m01 + inv12*mm01;
    float p02 =        half*m02 + inv12*mm02;
    float p10 =        half*m10 + inv12*mm10;
    float p11 = 1.0f + half*m11 + inv12*mm11;
    float p12 =        half*m12 + inv12*mm12;
    // p20=0, p21=0, p22=1

    // Q = I - M/2 + M²/12
    float q00 = 1.0f - half*m00 + inv12*mm00;
    float q01 =      - half*m01 + inv12*mm01;
    float q02 =      - half*m02 + inv12*mm02;
    float q10 =      - half*m10 + inv12*mm10;
    float q11 = 1.0f - half*m11 + inv12*mm11;
    float q12 =      - half*m12 + inv12*mm12;
    // q20=0, q21=0, q22=1

    // Solve Q * R = P for R, i.e. R = Q^{-1} * P
    // Q is 3x3 with last row [0,0,1], so we only need to invert the 2x2 top-left
    // block and handle the affine part.
    //
    // Q = [ q00 q01 q02 ]    Q^{-1} = [ Qi00 Qi01  -(Qi*q_t) ]
    //     [ q10 q11 q12 ]              [ Qi10 Qi11  -(Qi*q_t) ]
    //     [  0   0   1  ]              [   0    0       1     ]
    // where Qi is inverse of 2x2 top-left, q_t = [q02, q12]^T

    float det = q00 * q11 - q01 * q10;
    float inv_det = 1.0f / det;

    float qi00 =  q11 * inv_det;
    float qi01 = -q01 * inv_det;
    float qi10 = -q10 * inv_det;
    float qi11 =  q00 * inv_det;

    // Q^{-1} third column (top 2 elements): -(Qi * [q02, q12]^T)
    float qi02 = -(qi00 * q02 + qi01 * q12);
    float qi12 = -(qi10 * q02 + qi11 * q12);

    // R = Q^{-1} * P  (only need top 2 rows since last row of both is [0,0,1])
    // R[0][j] = qi0k * P[k][j], R[1][j] = qi1k * P[k][j]
    // P last row is [0,0,1]
    float r00 = qi00*p00 + qi01*p10;           // + qi02*0
    float r01 = qi00*p01 + qi01*p11;
    float r02 = qi00*p02 + qi01*p12 + qi02;    // + qi02*1
    float r10 = qi10*p00 + qi11*p10;
    float r11 = qi10*p01 + qi11*p11;
    float r12 = qi10*p02 + qi11*p12 + qi12;    // + qi12*1

    // Squaring step: R = R^s (s times squaring)
    // R is 3x3 with last row [0,0,1], so squaring preserves this structure
    for (int i = 0; i < n_squarings; i++) {
        float new_r00 = r00*r00 + r01*r10;
        float new_r01 = r00*r01 + r01*r11;
        float new_r02 = r00*r02 + r01*r12 + r02;  // + r02*1
        float new_r10 = r10*r00 + r11*r10;
        float new_r11 = r10*r01 + r11*r11;
        float new_r12 = r10*r02 + r11*r12 + r12;  // + r12*1
        r00 = new_r00; r01 = new_r01; r02 = new_r02;
        r10 = new_r10; r11 = new_r11; r12 = new_r12;
    }

    // Write output (top 2 rows only)
    float* out = Trels_out + idx * 6;
    out[0] = r00; out[1] = r01; out[2] = r02;
    out[3] = r10; out[4] = r11; out[5] = r12;
}
