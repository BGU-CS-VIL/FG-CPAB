#include <torch/extension.h>
#include <vector>

// CUDA function declarations
at::Tensor cpab_cuda_forward_optimized(at::Tensor points_in, at::Tensor trels_in,
                                        at::Tensor As_in, const int nstepsolver,
                                        const int ncx, const int ncy,
                                        const int adaptive_substeps,
                                        const int broadcast, at::Tensor output);
at::Tensor cpab_cuda_forward_1D_closed_form(at::Tensor points_in, at::Tensor As_in,
                                            const int ncx, const int broadcast,
                                            at::Tensor output);
at::Tensor cpab_cuda_backward_moments(at::Tensor points_in, at::Tensor As_in,
                                      at::Tensor grad_output_in,
                                      const int nstepsolver, const int ncx, const int ncy,
                                      const int adaptive_substeps,
                                      const int broadcast, at::Tensor moments);
std::vector<at::Tensor> cpab_cuda_backward_moments_and_points(
    at::Tensor points_in, at::Tensor As_in, at::Tensor grad_output_in,
    const int nstepsolver, const int ncx, const int ncy,
    const int adaptive_substeps, const int broadcast,
    at::Tensor moments, at::Tensor grad_points);
at::Tensor cpab_cuda_expm_optimized(at::Tensor As_in, const float dT, at::Tensor output);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Forward: Transform points through CPAB velocity field.
 */
at::Tensor cpab_forward(at::Tensor points_in,        // [ndim, nP] or [batch, ndim, nP]
                        at::Tensor trels_in,         // [batch, nC, ndim, ndim+1]
                        at::Tensor As_in,            // [batch, nC, ndim, ndim+1]
                        const int nstepsolver,       // number of solver steps
                        const int ncx,               // tessellation cells in x
                        const int ncy,               // tessellation cells in y
                        const int adaptive_substeps) { // Heun correction on cell-crossing
    CHECK_INPUT(points_in);
    CHECK_INPUT(trels_in);
    CHECK_INPUT(As_in);
    
    const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == trels_in.size(0));
    const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto batch_size = trels_in.size(0);

    AT_ASSERTM(As_in.size(0) == batch_size, "As_in batch dimension mismatch");
    AT_ASSERTM(As_in.size(1) == trels_in.size(1), "As_in nC dimension mismatch");
    AT_ASSERTM(As_in.size(2) == trels_in.size(2), "As_in ndim dimension mismatch");
    AT_ASSERTM(As_in.size(3) == trels_in.size(3), "As_in affine dimension mismatch");
    AT_ASSERTM(adaptive_substeps == 0 || adaptive_substeps == 1, "adaptive_substeps must be 0 or 1");
       
    auto output = torch::zeros({batch_size, ndim, nP}, at::kCUDA);
    if (ndim == 1) {
        return cpab_cuda_forward_1D_closed_form(points_in, As_in, ncx, broadcast, output);
    }
    return cpab_cuda_forward_optimized(points_in, trels_in, As_in, nstepsolver, ncx, ncy,
                                       adaptive_substeps,
                                       broadcast, output);
}

/**
 * Forward (1D closed-form): dedicated fast path that bypasses generic forward dispatch.
 */
at::Tensor cpab_forward_1d_closed_form(at::Tensor points_in, // [1,nP] or [batch,1,nP]
                                       at::Tensor As_in,     // [batch, nC, 1, 2]
                                       const int ncx) {
    CHECK_INPUT(points_in);
    CHECK_INPUT(As_in);

    AT_ASSERTM(As_in.dim() == 4, "As_in must have shape [batch, nC, 1, 2]");
    AT_ASSERTM(As_in.size(2) == 1 && As_in.size(3) == 2,
               "As_in must have affine 1D shape [batch, nC, 1, 2]");

    const auto batch_size = As_in.size(0);
    const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == batch_size);
    const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);

    AT_ASSERTM(ndim == 1, "points_in must be 1D for forward_1d_closed_form");

    auto output = torch::zeros({batch_size, 1, nP}, at::kCUDA);
    return cpab_cuda_forward_1D_closed_form(points_in, As_in, ncx, broadcast, output);
}

/**
 * Backward moments only: compute cell-wise moments without dense basis projection.
 * Returns moments with shape [n_theta, nC, 2] (1D), [n_theta, nC, 6] (2D),
 * or [n_theta, nC, 12] (3D).
 */
at::Tensor cpab_backward_moments(at::Tensor points_in,      // [ndim, nP] or [batch, ndim, nP]
                                 at::Tensor As_in,          // [n_theta, nC, ndim, ndim+1]
                                 at::Tensor grad_output_in, // [n_theta, ndim, nP]
                                 const int nstepsolver,     // number of solver steps
                                 const int ncx,             // tessellation cells in x
                                 const int ncy,             // tessellation cells in y
                                 const int adaptive_substeps) {
    CHECK_INPUT(points_in);
    CHECK_INPUT(As_in);
    CHECK_INPUT(grad_output_in);

    AT_ASSERTM(adaptive_substeps == 0 || adaptive_substeps == 1, "adaptive_substeps must be 0 or 1");

    const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == As_in.size(0));
    const int ndim = As_in.size(2);
    const auto n_theta = As_in.size(0);
    const auto nC = As_in.size(1);
    const int ppc = ndim * (ndim + 1);

    auto moments = torch::zeros({n_theta, nC, ppc}, at::kCUDA);
    return cpab_cuda_backward_moments(points_in, As_in, grad_output_in,
                                      nstepsolver, ncx, ncy,
                                      adaptive_substeps,
                                      broadcast, moments);
}

/**
 * Backward moments + point gradients:
 * returns [moments, grad_points] where grad_points has shape [n_theta, ndim, nP].
 */
std::vector<at::Tensor> cpab_backward_moments_and_points(
    at::Tensor points_in,      // [ndim, nP] or [batch, ndim, nP]
    at::Tensor As_in,          // [n_theta, nC, ndim, ndim+1]
    at::Tensor grad_output_in, // [n_theta, ndim, nP]
    const int nstepsolver,     // number of solver steps
    const int ncx,             // tessellation cells in x
    const int ncy,             // tessellation cells in y
    const int adaptive_substeps) {
    CHECK_INPUT(points_in);
    CHECK_INPUT(As_in);
    CHECK_INPUT(grad_output_in);

    AT_ASSERTM(adaptive_substeps == 0 || adaptive_substeps == 1, "adaptive_substeps must be 0 or 1");

    const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == As_in.size(0));
    const int ndim = As_in.size(2);
    const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
    const auto n_theta = As_in.size(0);
    const auto nC = As_in.size(1);
    const int ppc = ndim * (ndim + 1);

    auto moments = torch::zeros({n_theta, nC, ppc}, at::kCUDA);
    auto grad_points = torch::zeros({n_theta, ndim, nP}, at::kCUDA);

    return cpab_cuda_backward_moments_and_points(
        points_in, As_in, grad_output_in,
        nstepsolver, ncx, ncy,
        adaptive_substeps,
        broadcast,
        moments,
        grad_points);
}

/**
 * Matrix exponential for CPAB velocity field matrices.
 * Supports [n,1,2] (1D), [n,2,3] (2D), and [n,3,4] (3D) affine forms.
 */
at::Tensor cpab_expm(at::Tensor As_in,   // [n_matrices, ndim, ndim+1] velocity matrices
                     const float dT) {   // time step
    CHECK_INPUT(As_in);

    const int ndim = As_in.size(1);
    AT_ASSERTM(ndim == 1 || ndim == 2 || ndim == 3, "As_in must have ndim=1,2,3 in dim 1");
    AT_ASSERTM(As_in.size(2) == ndim + 1, "As_in must be affine with shape [n, ndim, ndim+1]");

    const int n_matrices = As_in.size(0);
    auto output = torch::zeros({n_matrices, ndim, ndim + 1}, at::kCUDA);
    return cpab_cuda_expm_optimized(As_in, dT, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cpab_forward, "CPAB forward (CUDA)");
    m.def("forward_1d_closed_form", &cpab_forward_1d_closed_form, "CPAB forward 1D closed-form (CUDA)");
    m.def("backward_moments", &cpab_backward_moments, "CPAB backward moments only (CUDA)");
    m.def("backward_moments_and_points", &cpab_backward_moments_and_points,
          "CPAB backward moments and points gradients (CUDA)");
    m.def("expm", &cpab_expm, "CPAB optimized affine matrix exponential (CUDA)");
}
