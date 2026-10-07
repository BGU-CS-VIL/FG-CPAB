#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "../core/cpab_ops_1d.cuh"
#include "../core/cpab_ops_2d.cuh"
#include "../core/cpab_ops_3d.cuh"

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * Forward pass - transforms points through CPAB velocity field.
 */
at::Tensor cpab_cuda_forward_optimized(
    at::Tensor points_in,
    at::Tensor trels_in,
    at::Tensor As_in,
    const int nstepsolver,
    const int ncx,
    const int ncy,
    const int adaptive_substeps,
    const int broadcast,
    at::Tensor output) {

    const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto batch_size = trels_in.size(0);

    TORCH_CHECK(
        nstepsolver > 0,
        "Invalid nstepsolver=", nstepsolver, ". Expected nstepsolver > 0.");
    TORCH_CHECK(
        nstepsolver <= CPAB_MAX_STEPS,
        "Invalid nstepsolver=", nstepsolver,
        ". Optimized CUDA backward supports at most CPAB_MAX_STEPS=",
        CPAB_MAX_STEPS,
        ". Reduce params.nstepsolver or increase CPAB_MAX_STEPS in "
        "libcpab/core/cpab_ops_2d.cuh and rebuild.");
    TORCH_CHECK(ndim == 2 || ndim == 3, "Optimized CUDA supports ndim=2,3 here, got ", ndim);

    dim3 bc(DIV_UP(nP, 256), batch_size);
    dim3 tpb(256, 1);

    if (ndim == 2) {
        cpab_cuda_kernel_forward_2D_optimized<<<bc, tpb>>>(
            nP, batch_size,
            output.data<float>(),
            points_in.data<float>(),
            trels_in.data<float>(),
            As_in.data<float>(),
            nstepsolver,
            ncx, ncy,
            adaptive_substeps,
            broadcast);
    } else {
        const int64_t nC = trels_in.size(1);
        const int64_t cells_per_z = (int64_t)5 * ncx * ncy;

        TORCH_CHECK(cells_per_z > 0, "Invalid ncx/ncy for 3D: ", ncx, " x ", ncy);
        TORCH_CHECK(
            nC % cells_per_z == 0,
            "Could not infer ncz from nC=", nC, ", ncx=", ncx, ", ncy=", ncy,
            " (expected nC to be divisible by 5*ncx*ncy=", cells_per_z, ").");

        const int ncz = (int)(nC / cells_per_z);
        TORCH_CHECK(ncz > 0, "Invalid inferred ncz=", ncz, " from nC=", nC);

        cpab_cuda_kernel_forward_3D_optimized<<<bc, tpb>>>(
            nP, batch_size,
            output.data<float>(),
            points_in.data<float>(),
            trels_in.data<float>(),
            As_in.data<float>(),
            nstepsolver,
            ncx, ncy, ncz,
            adaptive_substeps,
            broadcast);
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}

/**
 * Dedicated 1D closed-form forward.
 * Avoids generic ndim dispatch and Trels plumbing.
 */
at::Tensor cpab_cuda_forward_1D_closed_form(
    at::Tensor points_in,
    at::Tensor As_in,
    const int ncx,
    const int broadcast,
    at::Tensor output)
{
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto batch_size = As_in.size(0);

    dim3 bc(DIV_UP(nP, 256), batch_size);
    dim3 tpb(256, 1);

    cpab_cuda_kernel_forward_1D_closed_form<<<bc, tpb>>>(
        nP,
        batch_size,
        output.data<float>(),
        points_in.data<float>(),
        As_in.data<float>(),
        ncx,
        broadcast);

    gpuErrchk(cudaPeekAtLastError());
    return output;
}

/**
 * Compute matrix exponential for a batch of affine CPAB velocity matrices.
 * Dispatches to 3x3 (2D) or 4x4 (3D) optimized kernels.
 */
at::Tensor cpab_cuda_expm_optimized(
    at::Tensor As_in,
    const float dT,
    at::Tensor output)
{
    const int n_matrices = As_in.size(0);
    const int ndim = As_in.size(1);

    dim3 tpb(256);
    dim3 bc(DIV_UP(n_matrices, 256));

    if (ndim == 1) {
        cpab_cuda_kernel_expm2x2<<<bc, tpb>>>(
            n_matrices,
            dT,
            output.data<float>(),
            As_in.data<float>());
    } else if (ndim == 2) {
        cpab_cuda_kernel_expm3x3<<<bc, tpb>>>(
            n_matrices,
            dT,
            output.data<float>(),
            As_in.data<float>());
    } else if (ndim == 3) {
        cpab_cuda_kernel_expm4x4<<<bc, tpb>>>(
            n_matrices,
            dT,
            output.data<float>(),
            As_in.data<float>());
    } else {
        TORCH_CHECK(false, "Optimized expm supports ndim=1,2,3, got ", ndim);
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}

/**
 * Backward pass (moments only): computes cell-wise moments without projection.
 */
at::Tensor cpab_cuda_backward_moments(
    at::Tensor points_in,
    at::Tensor As_in,
    at::Tensor grad_output_in,
    const int nstepsolver,
    const int ncx,
    const int ncy,
    const int adaptive_substeps,
    const int broadcast,
    at::Tensor moments) {

    const int ndim = As_in.size(2);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto n_theta = As_in.size(0);
    const auto nC = As_in.size(1);

    TORCH_CHECK(
        nstepsolver > 0,
        "Invalid nstepsolver=", nstepsolver, ". Expected nstepsolver > 0.");
    TORCH_CHECK(
        nstepsolver <= CPAB_MAX_STEPS,
        "Invalid nstepsolver=", nstepsolver,
        ". Optimized CUDA backward supports at most CPAB_MAX_STEPS=",
        CPAB_MAX_STEPS,
        ". Reduce params.nstepsolver or increase CPAB_MAX_STEPS in "
        "libcpab/core/cpab_ops_2d.cuh and rebuild.");
    TORCH_CHECK(ndim == 1 || ndim == 2 || ndim == 3, "Optimized CUDA supports ndim=1,2,3, got ", ndim);
    TORCH_CHECK(
        adaptive_substeps == 0 || adaptive_substeps == 1,
        "adaptive_substeps must be 0 or 1");
    TORCH_CHECK(grad_output_in.size(1) == ndim, "grad_output ndim mismatch");

    dim3 bc1(DIV_UP(nP, 256), n_theta);
    dim3 tpb1(256, 1);

    if (ndim == 1) {
        TORCH_CHECK(moments.size(2) == 2, "moments ppc mismatch for 1D");
        cpab_cuda_kernel_fused_backward_1D<<<bc1, tpb1>>>(
            nP, n_theta, nC, nstepsolver,
            moments.data<float>(),
            nullptr,
            grad_output_in.data<float>(),
            points_in.data<float>(),
            As_in.data<float>(),
            ncx,
            adaptive_substeps,
            broadcast);
        gpuErrchk(cudaPeekAtLastError());
    } else if (ndim == 2) {
        TORCH_CHECK(moments.size(2) == 6, "moments ppc mismatch for 2D");
        cpab_cuda_kernel_fused_backward<<<bc1, tpb1>>>(
            nP, n_theta, nC, nstepsolver,
            moments.data<float>(),
            nullptr,
            grad_output_in.data<float>(),
            points_in.data<float>(),
            As_in.data<float>(),
            ncx, ncy,
            adaptive_substeps,
            broadcast);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        TORCH_CHECK(moments.size(2) == 12, "moments ppc mismatch for 3D");
        const int64_t cells_per_z = (int64_t)5 * ncx * ncy;
        TORCH_CHECK(cells_per_z > 0, "Invalid ncx/ncy for 3D: ", ncx, " x ", ncy);
        TORCH_CHECK(
            nC % cells_per_z == 0,
            "Could not infer ncz from nC=", nC, ", ncx=", ncx, ", ncy=", ncy,
            " (expected nC to be divisible by 5*ncx*ncy=", cells_per_z, ").");
        const int ncz = (int)(nC / cells_per_z);
        TORCH_CHECK(ncz > 0, "Invalid inferred ncz=", ncz, " from nC=", nC);

        cpab_cuda_kernel_fused_backward_3D<<<bc1, tpb1>>>(
            nP, n_theta, nC, nstepsolver,
            moments.data<float>(),
            nullptr,
            grad_output_in.data<float>(),
            points_in.data<float>(),
            As_in.data<float>(),
            ncx, ncy, ncz,
            adaptive_substeps,
            broadcast);
        gpuErrchk(cudaPeekAtLastError());
    }

    return moments;
}

/**
 * Backward pass (moments + grad_points): computes cell-wise moments and
 * adjoint at t=0 for each input point.
 */
std::vector<at::Tensor> cpab_cuda_backward_moments_and_points(
    at::Tensor points_in,
    at::Tensor As_in,
    at::Tensor grad_output_in,
    const int nstepsolver,
    const int ncx,
    const int ncy,
    const int adaptive_substeps,
    const int broadcast,
    at::Tensor moments,
    at::Tensor grad_points) {

    const int ndim = As_in.size(2);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto n_theta = As_in.size(0);
    const auto nC = As_in.size(1);

    TORCH_CHECK(
        nstepsolver > 0,
        "Invalid nstepsolver=", nstepsolver, ". Expected nstepsolver > 0.");
    TORCH_CHECK(
        nstepsolver <= CPAB_MAX_STEPS,
        "Invalid nstepsolver=", nstepsolver,
        ". Optimized CUDA backward supports at most CPAB_MAX_STEPS=",
        CPAB_MAX_STEPS,
        ". Reduce params.nstepsolver or increase CPAB_MAX_STEPS in "
        "libcpab/core/cpab_ops_2d.cuh and rebuild.");
    TORCH_CHECK(ndim == 1 || ndim == 2 || ndim == 3, "Optimized CUDA supports ndim=1,2,3, got ", ndim);
    TORCH_CHECK(
        adaptive_substeps == 0 || adaptive_substeps == 1,
        "adaptive_substeps must be 0 or 1");
    TORCH_CHECK(grad_output_in.size(1) == ndim, "grad_output ndim mismatch");
    TORCH_CHECK(grad_points.size(0) == n_theta, "grad_points batch mismatch");
    TORCH_CHECK(grad_points.size(1) == ndim, "grad_points ndim mismatch");
    TORCH_CHECK(grad_points.size(2) == nP, "grad_points nP mismatch");

    dim3 bc1(DIV_UP(nP, 256), n_theta);
    dim3 tpb1(256, 1);

    if (ndim == 1) {
        TORCH_CHECK(moments.size(2) == 2, "moments ppc mismatch for 1D");
        cpab_cuda_kernel_fused_backward_1D<<<bc1, tpb1>>>(
            nP, n_theta, nC, nstepsolver,
            moments.data<float>(),
            grad_points.data<float>(),
            grad_output_in.data<float>(),
            points_in.data<float>(),
            As_in.data<float>(),
            ncx,
            adaptive_substeps,
            broadcast);
        gpuErrchk(cudaPeekAtLastError());
    } else if (ndim == 2) {
        TORCH_CHECK(moments.size(2) == 6, "moments ppc mismatch for 2D");
        cpab_cuda_kernel_fused_backward<<<bc1, tpb1>>>(
            nP, n_theta, nC, nstepsolver,
            moments.data<float>(),
            grad_points.data<float>(),
            grad_output_in.data<float>(),
            points_in.data<float>(),
            As_in.data<float>(),
            ncx, ncy,
            adaptive_substeps,
            broadcast);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        TORCH_CHECK(moments.size(2) == 12, "moments ppc mismatch for 3D");
        const int64_t cells_per_z = (int64_t)5 * ncx * ncy;
        TORCH_CHECK(cells_per_z > 0, "Invalid ncx/ncy for 3D: ", ncx, " x ", ncy);
        TORCH_CHECK(
            nC % cells_per_z == 0,
            "Could not infer ncz from nC=", nC, ", ncx=", ncx, ", ncy=", ncy,
            " (expected nC to be divisible by 5*ncx*ncy=", cells_per_z, ").");
        const int ncz = (int)(nC / cells_per_z);
        TORCH_CHECK(ncz > 0, "Invalid inferred ncz=", ncz, " from nC=", nC);

        cpab_cuda_kernel_fused_backward_3D<<<bc1, tpb1>>>(
            nP, n_theta, nC, nstepsolver,
            moments.data<float>(),
            grad_points.data<float>(),
            grad_output_in.data<float>(),
            points_in.data<float>(),
            As_in.data<float>(),
            ncx, ncy, ncz,
            adaptive_substeps,
            broadcast);
        gpuErrchk(cudaPeekAtLastError());
    }

    return {moments, grad_points};
}
