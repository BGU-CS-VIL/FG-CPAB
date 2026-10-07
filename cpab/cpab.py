from .core.tesselation import Tesselation1D, Tesselation2D, Tesselation3D
from .pytorch.transformer import CPAB_transformer as transformer
from .core.utility import params, get_dir, create_dir
from .pytorch.interpolation import interpolate
import numpy as np
import torch

# GPU device constant
_DEVICE = torch.device("cuda")

class Cpab(object):
    """Continuous piecewise-affine transformations with a PyTorch GPU backend.

    Arguments:
        tess_size: list of 1, 2, or 3 integers defining the number of cells
            per spatial dimension
        zero_boundary: bool, whether the boundary velocity is constrained to 0
        volume_perservation: bool, whether the transform is volume preserving
        override: bool, whether to rebuild and overwrite the cached basis
    """
    def __init__(self, 
                 tess_size,
                 zero_boundary=True,
                 volume_perservation=False,
                 override=False):
        # Check input
        self._check_input(tess_size, zero_boundary, volume_perservation,
                          override)
                
        # Parameters
        self.params = params()
        self.params.nc = tess_size
        self.params.ndim = len(tess_size)
        self.params.Ashape = [self.params.ndim, self.params.ndim + 1]
        self.params.valid_outside = not(zero_boundary)
        self.params.zero_boundary = zero_boundary
        self.params.volume_perservation = volume_perservation
        self.params.domain_max = [1] * self.params.ndim
        self.params.domain_min = [0] * self.params.ndim
        self.params.inc = [(self.params.domain_max[i] - self.params.domain_min[i]) / 
                           self.params.nc[i] for i in range(self.params.ndim)]
        self.params.nstepsolver = 50
        self.params.adaptive_substeps = False
        
        # For saving the basis
        self._dir = get_dir(__file__) + '/../basis_files/'
        create_dir(self._dir)
            
        # Initialize tesselation
        if self.params.ndim == 1:
            self.params.nC = np.prod(self.params.nc)
            self.params.params_pr_cell = 2
            tesselation = Tesselation1D
        elif self.params.ndim == 2:
            self.params.nC = 4*np.prod(self.params.nc)
            self.params.params_pr_cell = 6
            tesselation = Tesselation2D
        elif self.params.ndim == 3:
            self.params.nC = 5*np.prod(self.params.nc)
            self.params.params_pr_cell = 12
            tesselation = Tesselation3D
        else:
            raise ValueError('Only 1D, 2D, and 3D tessellations are supported')
        
        self.tesselation = tesselation(
            self.params.nc, self.params.domain_min, 
            self.params.domain_max, self.params.zero_boundary, 
            self.params.volume_perservation,
            self._dir, override
        )
        
        # Extract parameters from tesselation
        self.params.constrain_mat = self.tesselation.L
        self.params.basis = self.tesselation.B
        self.params.D, self.params.d = self.params.basis.shape
        self.params.basis_gpu_sparse = None
        self.params.basis_gpu_sparse_t = None
        basis_sparse = self._get_basis_sparse_gpu(dtype=torch.float32, device=_DEVICE)
        self.params.basis_gpu_sparse_t = basis_sparse.transpose(0, 1).coalesce()
        
    def get_theta_dim(self):
        """Return the dimensionality of the transformation parameters."""
        return self.params.d
    
    def get_params(self):
        """Return the parameter container for the transformation."""
        return self.params
    
    def get_basis(self):
        """Return the basis matrix of the transformation."""
        return self.params.basis

    def _get_basis_sparse_gpu(self, dtype=torch.float32, device=_DEVICE):
        """Get sparse basis tensor on device, materializing as needed."""
        B = self.params.basis_gpu_sparse
        if B is not None and B.dtype == dtype and B.device == device:
            return B

        basis_coo = self.params.basis.tocoo()
        indices_np = np.vstack((basis_coo.row, basis_coo.col)).astype(np.int64)
        indices = torch.from_numpy(indices_np).to(device=device)
        values = torch.from_numpy(basis_coo.data).to(device=device, dtype=dtype)
        B = torch.sparse_coo_tensor(
            indices, values, size=self.params.basis.shape, device=device, dtype=dtype
        ).coalesce()

        if device.type == "cuda" and device.index in (None, 0) and dtype == torch.float32:
            self.params.basis_gpu_sparse = B
            self.params.basis_gpu_sparse_t = B.transpose(0, 1).coalesce()
        return B
    
    def set_solver_params(
        self,
        nstepsolver=50,
        adaptive_substeps=False,
    ):
        """Set parameters controlling the integration algorithm.

        Arguments:
            nstepsolver: int, number of integration steps. Higher values improve
                the approximation at increased cost
            adaptive_substeps: bool, enable Heun crossing correction in the
                CUDA path when a point crosses into a different cell
        """
        assert nstepsolver > 0, 'nstepsolver must be a positive number'
        assert type(nstepsolver) == int, 'nstepsolver must be integer'
        assert type(adaptive_substeps) == bool, 'adaptive_substeps must be bool'
        self.params.nstepsolver = nstepsolver
        self.params.adaptive_substeps = adaptive_substeps
        
    def uniform_meshgrid(self, n_points):
        """Constructs a meshgrid.
        
        Arguments:
            n_points: list of ndim integers, number of points in each dimension
            
        Output:
            grid: [ndim, nP] matrix of points, where nP = product(n_points)
        """
        lin = [torch.linspace(self.params.domain_min[i], self.params.domain_max[i], 
                              n_points[i], device=_DEVICE) for i in range(self.params.ndim)]
        mesh = torch.meshgrid(lin[::-1], indexing='ij')
        grid = torch.cat([g.reshape(1, -1) for g in mesh[::-1]], dim=0)
        return grid
      
    def sample_transformation(self, n_sample=1, mean=None, cov=None):
        """Sample transformation from multivariate gaussian.
        
        Arguments:
            n_sample: integer, number of transformations to sample
            mean: [d,] vector, mean of multivariate gaussian
            cov: [d,d] matrix, covariance of multivariate gaussian
            
        Output:
            samples: [n_sample, d] matrix. Each row is an independent sample
        """
        d = self.params.d
        mean = torch.zeros(d, dtype=torch.float32, device=_DEVICE) if mean is None else mean
        cov = torch.eye(d, dtype=torch.float32, device=_DEVICE) if cov is None else cov
        distribution = torch.distributions.MultivariateNormal(mean, cov)
        samples = distribution.sample((n_sample,))
        return samples
        
    def sample_transformation_with_prior(self, n_sample=1, mean=None, 
                                         length_scale=0.1, output_variance=1):
        """Sample smooth transformations using squared exponential kernel.
        
        Arguments:
            n_sample: integer, number of transformations to sample
            mean: [d,] vector, mean of multivariate gaussian
            length_scale: float>0, determines how fast the covariance declines 
            output_variance: float>0, determines the overall variance from the mean
            
        Output:
            samples: [n_sample, d] matrix
        """
        # Get cell centers and convert to tensor
        centers = torch.tensor(self.tesselation.get_cell_centers(), 
                               dtype=torch.float32, device=_DEVICE)
        
        # Get distance between cell centers (pdist)
        norm = torch.sum(centers * centers, 1)
        norm = torch.reshape(norm, (-1, 1))
        dist = norm - 2 * centers.mm(centers.t()) + norm.t()
        
        # Make into a covariance matrix between parameters
        ppc = self.params.params_pr_cell
        cov_init = torch.zeros(self.params.D, self.params.D, device=_DEVICE)
        
        for i in range(self.params.nC):
            for j in range(self.params.nC):
                block = 100 * dist.max() * torch.ones(ppc, ppc, device=_DEVICE)
                block[torch.arange(ppc), torch.arange(ppc)] = dist[i, j].repeat(ppc)
                cov_init[ppc*i:ppc*(i+1), ppc*j:ppc*(j+1)] = block
        
        # Squared exponential kernel
        cov_avees = output_variance**2 * torch.exp(-(cov_init / (2*length_scale**2)))

        # Transform covariance to theta space via sparse basis multiplications.
        B_t = self._get_basis_sparse_gpu(dtype=torch.float32, device=_DEVICE).transpose(0, 1).coalesce()
        cov_avees_B = torch.sparse.mm(B_t, cov_avees.t()).t()
        cov_theta = torch.sparse.mm(B_t, cov_avees_B)
        
        # Sample
        samples = self.sample_transformation(n_sample, mean=mean, cov=cov_theta)
        return samples
    
    def identity(self, n_sample=1, epsilon=0):
        """Get identity parameters (vector of zeros).
        
        Arguments:
            n_sample: integer, number of identity transformations
            epsilon: float>=0, small number to add for stability during training
            
        Output:
            samples: [n_sample, d] matrix
        """
        assert epsilon >= 0, "epsilon need to be larger than 0"
        return torch.zeros(n_sample, self.params.d, dtype=torch.float32, device=_DEVICE) + epsilon
    
    def transform_grid(self, grid, theta):
        """Integrate a grid using the transformation parameters in ``theta``.

        Arguments:
            grid: [ndim, n_points] matrix or [n_batch, ndim, n_points] tensor
            theta: [n_batch, d] matrix
            
        Output:
            transformed_grid: [n_batch, ndim, n_points] tensor
        """
        if len(grid.shape) == 3:
            assert grid.shape[0] == theta.shape[0], \
                'When passing a batched grid, the first dimension must match theta'
        transformed_grid = transformer(grid, theta, self.params, compute_points_grad=True)
        return transformed_grid

    def transform_grid_chain(self, grid, theta_chain, scale_by_dt=False):
        """Apply a time-ordered chain of CPAB transforms to a grid.

        This supports piecewise-constant-in-time velocity fields by composing
        multiple CPAB stages. If ``scale_by_dt=True``, each stage theta is
        multiplied by ``dt = 1 / n_transforms`` so the total integration horizon
        remains 1.0 when using ``n_transforms`` stages.

        Arguments:
            grid: [ndim, n_points] or [n_batch, ndim, n_points] tensor
            theta_chain: [n_transforms, d] or [n_transforms, n_batch, d] tensor
            scale_by_dt: bool, apply dt scaling per stage

        Output:
            transformed_grid: [n_batch, ndim, n_points] tensor
        """
        assert type(scale_by_dt) == bool, 'scale_by_dt must be bool'
        assert theta_chain.dim() in (2, 3), \
            'theta_chain must have shape [n_transforms, d] or [n_transforms, n_batch, d]'

        if theta_chain.dim() == 2:
            theta_chain = theta_chain[:, None, :]

        n_transforms, n_batch, d = theta_chain.shape
        assert n_transforms > 0, 'theta_chain must contain at least one transform'
        assert d == self.params.d, \
            f'theta_chain has d={d}, expected d={self.params.d}'

        if grid.dim() == 3:
            assert grid.shape[0] == n_batch, \
                'When passing a 3D grid, its batch dimension must match theta_chain'

        dt = (1.0 / n_transforms) if scale_by_dt else 1.0
        transformed_grid = grid
        for i in range(n_transforms):
            theta_i = theta_chain[i] * dt
            transformed_grid = transformer(
                transformed_grid,
                theta_i,
                self.params,
                compute_points_grad=True
            )

        return transformed_grid
    
    def interpolate(self, data, grid, outsize):
        """Linear interpolation method.
        
        Arguments:
            data: [n_batch, n_channels, *in_size] tensor (PyTorch format)
            grid: [n_batch, ndim, n_points] tensor with grid points
            outsize: list of ndim integers, output size per dimension
            
        Output:
            interpolated: [n_batch, n_channels, *outsize] tensor
        """            
        return interpolate(self.params.ndim, data, grid, outsize)
    
    def transform_data(self, data, theta, outsize):
        """Combination of transform_grid and interpolate for easy data transformation.
        
        Arguments:
            data: [n_batch, n_channels, *in_size] tensor (PyTorch format)
            theta: [n_batch, d] matrix with transformation parameters
            outsize: list of ndim integers, output size per dimension
            
        Output:
            data_t: [n_batch, n_channels, *outsize] tensor
        """
        grid = self.uniform_meshgrid(outsize)
        grid_t = self.transform_grid(grid, theta)
        data_t = self.interpolate(data, grid_t, outsize)
        return data_t
     
    def _check_input(self, tess_size, zero_boundary, volume_perservation,
                     override):
        """Validate input parameters."""
        assert type(tess_size) in [list, tuple], 'tess_size must be a list or tuple'
        assert len(tess_size) in [1, 2, 3], 'Only 1D, 2D, and 3D tesselations are supported'
        assert all([type(e) == int for e in tess_size]), 'All elements must be integers'
        assert all([e > 0 for e in tess_size]), 'All elements must be positive'
        assert type(zero_boundary) == bool, 'zero_boundary must be True or False'
        assert type(volume_perservation) == bool, 'volume_perservation must be True or False'
        assert type(override) == bool, 'override must be True or False'
            
    def __repr__(self):
        return f'''
        CPAB transformer class ({self.params.ndim}D PyTorch GPU). 
            Parameters:
                Tesselation size:           {self.params.nc}
                Total number of cells:      {self.params.nC}
                Theta size:                 {self.params.d}
                Domain lower bound:         {self.params.domain_min}
                Domain upper bound:         {self.params.domain_max}
                Zero Boundary:              {self.params.zero_boundary}
                Volume perservation:        {self.params.volume_perservation}
        '''
