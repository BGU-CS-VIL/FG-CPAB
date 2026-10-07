import torch
from torch.utils.cpp_extension import load
from ..core.utility import get_dir

# JIT compile CUDA source
_dir = get_dir(__file__)
_verbose = False

try:
    cpab_gpu = load(
        name='cpab_gpu',
        sources=[
            _dir + '/transformer_cuda.cpp',
            _dir + '/transformer_cuda.cu',
            _dir + '/../core/cpab_ops_1d.cu',
            _dir + '/../core/cpab_ops_2d.cu',
            _dir + '/../core/cpab_ops_3d.cu'
        ],
        verbose=_verbose,
        with_cuda=True
    )
    _gpu_success = True
except Exception as e:
    _gpu_success = False
    print(f'Failed to compile CUDA source: {e}')


def CPAB_transformer(points, theta, params, compute_points_grad=False):
    """Transform points through CPAB velocity field.
    
    Arguments:
        points: [ndim, n_points] or [n_batch, ndim, n_points] tensor
        theta: [n_batch, d] tensor of transformation parameters
        params: parameter object from Cpab class
        compute_points_grad: bool, request gradient w.r.t. input points in backward
        
    Returns:
        transformed_points: [n_batch, ndim, n_points] tensor
    """
    assert points.is_cuda and theta.is_cuda, "Only GPU tensors supported"
    assert _gpu_success, "CUDA compilation failed"
    assert params.ndim in (1, 2, 3), "GPU transformer supports 1D, 2D, and 3D"
    assert type(compute_points_grad) == bool, "compute_points_grad must be bool"
    
    return _CPABFunction.apply(points, theta, params, compute_points_grad)

def _get_sparse_basis(params, device, dtype):
    """Get sparse basis tensor on the requested CUDA device and dtype."""
    B_sparse = getattr(params, "basis_gpu_sparse", None)
    if B_sparse is not None and B_sparse.device == device and B_sparse.dtype == dtype:
        return B_sparse

    basis = params.basis.tocoo()
    indices_np = torch.stack((
        torch.from_numpy(basis.row),
        torch.from_numpy(basis.col),
    )).to(device=device, dtype=torch.int64)
    values = torch.from_numpy(basis.data).to(device=device, dtype=dtype)
    B_sparse = torch.sparse_coo_tensor(
        indices_np, values, size=params.basis.shape, device=device, dtype=dtype
    ).coalesce()

    if device.type == "cuda" and device.index in (None, 0) and dtype == torch.float32:
        params.basis_gpu_sparse = B_sparse
        params.basis_gpu_sparse_t = B_sparse.transpose(0, 1).coalesce()
    return B_sparse


class _CPABFunction(torch.autograd.Function):
    """Autograd function with analytical gradients using adjoint method."""
    
    @staticmethod
    def forward(ctx, points, theta, params, compute_points_grad=False):
        n_theta = theta.shape[0]
        B_sparse = _get_sparse_basis(params, theta.device, theta.dtype)
        B_sparse_t = getattr(params, "basis_gpu_sparse_t", None)
        if B_sparse_t is None or B_sparse_t.device != theta.device or B_sparse_t.dtype != theta.dtype:
            B_sparse_t = B_sparse.transpose(0, 1).coalesce()
            if theta.device.type == "cuda" and theta.device.index in (None, 0) and theta.dtype == torch.float32:
                params.basis_gpu_sparse_t = B_sparse_t
        ctx.B_sparse_t = B_sparse_t

        # Compute velocity field matrices: A = B @ theta.
        Avees = torch.sparse.mm(B_sparse, theta.t())

        As = Avees.t().reshape(n_theta * params.nC, *params.Ashape)
        
        As_batch = As.view(n_theta, params.nC, *params.Ashape)

        # Call CUDA forward with host integers (avoids device-to-host memcpy)
        nstepsolver = params.nstepsolver
        ncx = int(params.nc[0])
        ncy = int(params.nc[1]) if params.ndim > 1 else 1
        adaptive_substeps = int(getattr(params, "adaptive_substeps", False))

        # 1D: dedicated closed-form entrypoint to avoid generic forward dispatch.
        if params.ndim == 1:
            newpoints = cpab_gpu.forward_1d_closed_form(
                points.contiguous(),
                As_batch.contiguous(),
                ncx,
            )
        else:
            dT = 1.0 / params.nstepsolver
            Trels = cpab_gpu.expm(As.contiguous(), dT)
            Trels = Trels.view(n_theta, params.nC, *params.Ashape)

            newpoints = cpab_gpu.forward(
                points.contiguous(), 
                Trels.contiguous(), 
                As_batch.contiguous(),
                nstepsolver, 
                ncx,
                ncy,
                adaptive_substeps
            )

        ctx.nstepsolver = nstepsolver
        ctx.ncx = ncx
        ctx.ncy = ncy
        ctx.adaptive_substeps = adaptive_substeps
        ctx.compute_points_grad = compute_points_grad
        ctx.params = params
        ctx.save_for_backward(points, As_batch)
        
        return newpoints

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        points, As = ctx.saved_tensors
        nstepsolver = ctx.nstepsolver
        ncx = ctx.ncx
        ncy = ctx.ncy
        adaptive_substeps = ctx.adaptive_substeps

        # Sparse runtime path:
        # 1) CUDA fused backward computes moments [n_theta, nC, ppc]
        # 2) sparse projection grad = moments_flat @ B
        if ctx.compute_points_grad:
            moments, grad_points = cpab_gpu.backward_moments_and_points(
                points.contiguous(),
                As.contiguous(),
                grad.contiguous(),
                nstepsolver,
                ncx,
                ncy,
                adaptive_substeps
            )
            # Broadcast input points [ndim, nP] are reused across all theta batches.
            if points.dim() == 2:
                grad_points = grad_points.sum(dim=0).contiguous()
        else:
            moments = cpab_gpu.backward_moments(
                points.contiguous(),
                As.contiguous(),
                grad.contiguous(),
                nstepsolver,
                ncx,
                ncy,
                adaptive_substeps
            )
            grad_points = None

        moments_flat = moments.reshape(moments.shape[0], -1)
        grad_theta = torch.sparse.mm(ctx.B_sparse_t, moments_flat.t()).t().contiguous()
            
        return grad_points, grad_theta, None, None
