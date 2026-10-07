import numpy as np
import scipy.sparse as sp
from .utility import (
    make_hashable,
    check_if_file_exist,
    null,
    save_obj,
    load_obj,
    drop_zero_rows,
)
from itertools import combinations
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

class Tesselation(object):
    """ Base tesselation class. This function is not meant to be called,
        but descripes the base structure that needs to be implemented in
        1D, 2D, and 3D. Additionally, some functionallity is shared across
        the different dimensions.
        
    Args:
        nc: list with number of cells
        domain_min: value of the lower bound(s) of the domain
        domain_max: value of the upper bound(s) of the domain
        zero_boundary: bool, if true the velocity is zero on the boundary
        volume_perservation: bool, if true volume is perserved
    Methods that should not be implemented in subclasses:
        @get_cell_centers:
        @create_continuity_constrains:
        @create_zero_trace_constrains:
            
    Methods that should be implemented in subclasses:
        @find_verts:
        @find_verts_outside:
        @create_zero_boundary_constrains:
        
    """
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        """ Initilization of the class that create the constrain matrix L
        Arguments:
            nc: list, number of cells in each dimension
            domain_min: list, lower domain bound in each dimension
            domain_max: list, upper domain bound in each dimension
            zero_boundary: bool, determines is the velocity at the boundary is zero
            volume_perservation: bool, determine if the transformation is
                volume perservating
            direc: string, where to store the basis
            override: bool, determines if we should calculate the basis even
                if it already exists
        """
        
        # Save parameters
        self.nc = nc
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
        self.dir = direc
        basis_version = 'nodal_v2'
        self._basis_file = self.dir + \
                            'cpab_basis_dim' + str(len(self.nc)) + '_tess' + \
                            '_'.join([str(e) for e in self.nc]) + '_' + \
                            'vo' + str(int(not self.zero_boundary)) + '_' + \
                            'zb' + str(int(self.zero_boundary)) + '_' + \
                            'vp' + str(int(self.volume_perservation)) + '_' + \
                            basis_version

        # Check if file exist else calculate the basis
        if not check_if_file_exist(self._basis_file+'.pkl') or override:
            # Get vertices
            self.find_verts()
            
            # Find shared vertices
            self.find_shared_verts()
            
            # find auxility vertices, if transformation is valid outside
            if not zero_boundary: self.find_verts_outside()
            
            # Get continuity constrains
            self.L = self.create_continuity_constrains()
            
            # If zero boundary, add constrains
            if zero_boundary:
                temp = self.create_zero_boundary_constrains()
                self.L = sp.vstack((self.L, temp), format='csr')
                
            # If volume perservation, add constrains
            if volume_perservation:
                temp = self.create_zero_trace_constrains()
                self.L = sp.vstack((self.L, temp), format='csr')
            
            # Build sparse nodal basis
            self.B = self.create_nodal_basis_sparse()
        
            # Save to file
            save_obj(self.__dict__, self._basis_file)
        
        else:
            self.__dict__ = load_obj(self._basis_file)
            if not sp.issparse(self.L):
                self.L = sp.csr_matrix(self.L)
            if not sp.issparse(self.B):
                self.B = sp.csr_matrix(self.B)
    
    def get_cell_centers(self):
        """ Get the centers of all the cells """
        return np.mean(self.verts[:,:,:self.ndim], axis=1)
    
    def find_verts(self):
        """ Function that should find the different vertices of all cells in
            the tesselation """
        raise NotImplementedError
        
    def find_shared_verts(self):
        """ Find pairs of cells that share ndim-vertices. It is these pairs,
            where we need to add continuity constrains at """
        shared_v, shared_v_idx = [ ], [ ]
        face_to_cells = {}
        for i in tqdm(range(self.nC), total=self.nC, desc="Finding shared vertices"):
            vi = make_hashable(self.verts[i])
            for face in combinations(vi, self.ndim):
                sorted_face = tuple(sorted(face))
                if sorted_face in face_to_cells:
                    face_to_cells[sorted_face].append(i)
                else:
                    face_to_cells[sorted_face] = [i]
                    
        for face, cells in face_to_cells.items():
            if len(cells) > 1:
                for idx1 in range(len(cells)):
                    for idx2 in range(idx1 + 1, len(cells)):
                        c1, c2 = cells[idx1], cells[idx2]
                        shared_v.append(list(face))
                        shared_v_idx.append((c1, c2))
                        
        # Save result
        if len(shared_v) > 0:
            self.shared_v = np.asarray(shared_v)
        else:
            self.shared_v = np.zeros((0, self.ndim, self.ndim))
        self.shared_v_idx = shared_v_idx
        
    def find_verts_outside(self):
        """ If the transformation should be valid outside, this function should
            add additional auxilliry points to the tesselation that secures
            continuity outside the domain """
        raise NotImplementedError
        
    def create_continuity_constrains(self):
        """ This function goes through all pairs (i,j) of cells that share a
            boundary. In N dimension we need to add N*N constrains (one for each
            dimension times one of each vertex in the boundary) """
        n_pairs = len(self.shared_v_idx)
        n_rows = n_pairs * self.ndim * self.ndim
        n_cols = self.n_params * self.nC
        rows, cols, vals = [ ], [ ], [ ]

        row_idx = 0
        for idx, (i,j) in enumerate(tqdm(self.shared_v_idx, total=n_pairs,
                                         desc="Building continuity constraints")):
            base_i = self.n_params * i
            base_j = self.n_params * j
            for vidx in range(self.ndim):
                v = np.asarray(self.shared_v[idx][vidx], dtype=np.float64)
                for k in range(self.ndim):
                    index1 = base_i + k * (self.ndim + 1)
                    index2 = base_j + k * (self.ndim + 1)
                    for q, value in enumerate(v):
                        if value != 0.0:
                            rows.append(row_idx)
                            cols.append(index1 + q)
                            vals.append(value)
                            rows.append(row_idx)
                            cols.append(index2 + q)
                            vals.append(-value)
                    row_idx += 1
        Ltemp = sp.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        return Ltemp.tocsr()
        
    def create_zero_boundary_constrains(self):
        """ Function that creates a constrain matrix L, containing constrains that
            secure 0 velocity at the boundary """
        raise NotImplementedError
        
    def create_zero_trace_constrains(self):
        """ The volume perservation constrains, that corresponds to the trace
            of each matrix being 0. These can be written general for all dims."""
        rows, cols, vals = [ ], [ ], [ ]
        for c in tqdm(range(self.nC), total=self.nC, desc="Building volume constraints"):
            base = self.n_params * c
            for k in range(self.ndim):
                rows.append(c)
                cols.append(base + k*(self.ndim + 1) + k)
                vals.append(1.0)
        Ltemp = sp.coo_matrix((vals, (rows, cols)),
                              shape=(self.nC, self.n_params * self.nC))
        return Ltemp.tocsr()

    def _vertex_key(self, vertex):
        """Stable hashable key for a vertex in homogeneous coordinates."""
        return tuple(np.asarray(vertex, dtype=np.float64).tolist())

    def _is_boundary_vertex(self, vertex, atol=1e-12):
        """Check if a vertex lies on the domain boundary."""
        for d in range(self.ndim):
            if np.isclose(vertex[d], self.domain_min[d], atol=atol) or \
               np.isclose(vertex[d], self.domain_max[d], atol=atol):
                return True
        return False

    def create_nodal_prolongation(self):
        """Create sparse map from nodal velocity DOFs to cell-wise affine params."""
        # Discover unique vertices in deterministic order
        all_vertices = [ ]
        vertex_seen = { }
        for c in range(self.nC):
            for v in self.verts[c]:
                key = self._vertex_key(v)
                if key not in vertex_seen:
                    vertex_seen[key] = True
                    all_vertices.append(key)

        # Active nodal DOFs: interior vertices if zero-boundary, else all vertices
        active_vertices = [ ]
        active_index = { }
        for key in all_vertices:
            if self.zero_boundary and self._is_boundary_vertex(key):
                continue
            active_index[key] = len(active_vertices)
            active_vertices.append(key)

        n_dofs = self.ndim * len(active_vertices)
        D = self.n_params * self.nC
        if n_dofs == 0:
            self.nodal_vertices = np.asarray(active_vertices)
            return sp.csr_matrix((D, 0))

        rows, cols, vals = [ ], [ ], [ ]
        for c in tqdm(range(self.nC), total=self.nC, desc="Building nodal prolongation"):
            verts = np.asarray(self.verts[c], dtype=np.float64)
            v_inv = np.linalg.inv(verts.T)
            base = self.n_params * c

            for local_vidx, v in enumerate(verts):
                key = self._vertex_key(v)
                vtx_idx = active_index.get(key)
                if vtx_idx is None:
                    continue

                coeffs = v_inv[local_vidx]
                for k in range(self.ndim):
                    col = self.ndim * vtx_idx + k
                    row_base = base + k * (self.ndim + 1)
                    for q, value in enumerate(coeffs):
                        if value != 0.0:
                            rows.append(row_base + q)
                            cols.append(col)
                            vals.append(value)

        N = sp.coo_matrix((vals, (rows, cols)), shape=(D, n_dofs)).tocsr()
        N.eliminate_zeros()
        self.nodal_vertices = np.asarray(active_vertices)
        return N

    def create_nodal_basis_sparse(self, eps=1e-6):
        """Build sparse basis in nodal form and project residual constraints."""
        N = self.create_nodal_prolongation()
        C = (self.L @ N).tocsr()
        if C.nnz:
            C.data[np.abs(C.data) <= eps] = 0.0
            C.eliminate_zeros()
        C = drop_zero_rows(C)

        if C.shape[0] == 0 or C.shape[1] == 0:
            return self._normalize_basis_columns(N)

        Z = null(C, eps=eps, zero_tol=eps)
        if Z.shape[1] == 0:
            return sp.csr_matrix((N.shape[0], 0))

        B = (N @ Z).tocsr()
        if B.nnz:
            B.data[np.abs(B.data) <= eps] = 0.0
            B.eliminate_zeros()
        return self._normalize_basis_columns(B)

    def _normalize_basis_columns(self, B):
        """Scale each basis column to unit L2 norm while preserving sparsity."""
        if B.shape[1] == 0:
            return B.tocsr()

        Bc = B.tocsc(copy=True)
        norms = np.sqrt(np.asarray(Bc.power(2).sum(axis=0)).ravel())
        norms[norms == 0.0] = 1.0

        for j in range(Bc.shape[1]):
            start, end = Bc.indptr[j], Bc.indptr[j + 1]
            if end > start:
                Bc.data[start:end] /= norms[j]

        Bc.eliminate_zeros()
        return Bc.tocsr()
        
class Tesselation1D(Tesselation):
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        # 1D parameters
        self.n_params = 2
        self.nC = np.prod(nc)
        self.ndim = 1
        
        # Initialize super class
        super(Tesselation1D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation, direc, override)
        
    def find_verts(self):
        Vx = np.linspace(self.domain_min[0], self.domain_max[0], self.nc[0]+1)
        
        # Find cell index and verts for each cell
        cells, verts = [ ], [ ]
        for i in range(self.nc[0]):
            v1 = tuple([Vx[i], 1])
            v2 = tuple([Vx[i+1], 1])
            verts.append((v1, v2))
            cells.append((i))
        
        # Convert to array
        self.verts = np.asarray(verts)
        self.cells = cells
        
    def find_verts_outside(self):
        pass # in 1D, we do not need auxilliry points
        
    def create_zero_boundary_constrains(self):
        n_cols = 2 * self.nC
        rows = [0, 0, 1, 1]
        cols = [0, 1, n_cols - 2, n_cols - 1]
        vals = [self.domain_min[0], 1.0, self.domain_max[0], 1.0]
        Ltemp = sp.coo_matrix((vals, (rows, cols)), shape=(2, n_cols))
        return Ltemp.tocsr()

#%%
class Tesselation2D(Tesselation):
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        # 1D parameters
        self.n_params = 6 
        self.nC = 4*np.prod(nc) # 4 triangle per cell
        self.ndim = 2
        
        # Initialize super class
        super(Tesselation2D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation, direc, override)
    
    def find_verts(self):
        Vx = np.linspace(self.domain_min[0], self.domain_max[0], self.nc[0]+1)
        Vy = np.linspace(self.domain_min[1], self.domain_max[1], self.nc[1]+1)
        
        # Find cell index and verts for each cell
        cells, verts = [ ], [ ]
        for i in range(self.nc[1]):
            for j in range(self.nc[0]):
                ul = tuple([Vx[j],Vy[i],1])
                ur = tuple([Vx[j+1],Vy[i],1])
                ll = tuple([Vx[j],Vy[i+1],1])
                lr = tuple([Vx[j+1],Vy[i+1],1])
                
                center = [(Vx[j]+Vx[j+1])/2,(Vy[i]+Vy[i+1])/2,1]
                center = tuple(center)                 
                
                verts.append((center,ul,ur))  # order matters!
                verts.append((center,ur,lr))  # order matters!
                verts.append((center,lr,ll))  # order matters!
                verts.append((center,ll,ul))  # order matters!                
        
                cells.append((j,i,0))
                cells.append((j,i,1))
                cells.append((j,i,2))
                cells.append((j,i,3))
                
        # Convert to array
        self.verts = np.asarray(verts)
        self.cells = cells
        
    def find_verts_outside(self):
        shared_v, shared_v_idx = [ ], [ ]
        cell_to_idx = {tuple(self.cells[k]): k for k in range(self.nC)}
        
        for i in tqdm(range(self.nC), total=self.nC, desc="Finding outside vertices (2D)"):
            mi = self.cells[i]
            potential_j = []
            
            if mi[0] == 0 and mi[2] == 3:
                potential_j.extend([(0, mi[1] + 1, 3), (0, mi[1] - 1, 3)])
            if mi[0] == self.nc[0] - 1 and mi[2] == 1:
                potential_j.extend([(self.nc[0] - 1, mi[1] + 1, 1), (self.nc[0] - 1, mi[1] - 1, 1)])
            if mi[1] == 0 and mi[2] == 0:
                potential_j.extend([(mi[0] + 1, 0, 0), (mi[0] - 1, 0, 0)])
            if mi[1] == self.nc[1] - 1 and mi[2] == 2:
                potential_j.extend([(mi[0] + 1, self.nc[1] - 1, 2), (mi[0] - 1, self.nc[1] - 1, 2)])
            
            for mj in potential_j:
                if mj in cell_to_idx:
                    j = cell_to_idx[mj]
                    if i < j:
                        vi = make_hashable(self.verts[i])
                        vj = make_hashable(self.verts[j])
                        shared_verts = set(vi).intersection(vj)
                        if len(shared_verts) == 1:
                            v_aux = list(list(shared_verts)[0])
                            if (mi[0] == 0 and mi[2] == 3) or (mi[0] == self.nc[0] - 1 and mi[2] == 1):
                                v_aux[0] -= 10
                            elif (mi[1] == 0 and mi[2] == 0) or (mi[1] == self.nc[1] - 1 and mi[2] == 2):
                                v_aux[1] -= 10
                            shared_verts = [tuple(shared_verts)[0], tuple(v_aux)]
                            shared_v.append(shared_verts)
                            shared_v_idx.append((i, j))
                            
        # Concat to the current list of vertices
        if shared_v:
            self.shared_v = np.concatenate((self.shared_v, shared_v))
            self.shared_v_idx += shared_v_idx
        
    def create_zero_boundary_constrains(self):
        xmin, ymin = self.domain_min
        xmax, ymax = self.domain_max
        n_cols = 6 * self.nC
        rows, cols, vals = [ ], [ ], [ ]
        row_idx = 0
        for c in tqdm(range(self.nC), total=self.nC, desc="Building boundary constraints (2D)"):
            base = 6 * c
            for v in self.verts[c]:
                if v[0] == xmin or v[0] == xmax:
                    for q, value in enumerate(v):
                        if value != 0.0:
                            rows.append(row_idx)
                            cols.append(base + q)
                            vals.append(value)
                    row_idx += 1
                if v[1] == ymin or v[1] == ymax:
                    for q, value in enumerate(v):
                        if value != 0.0:
                            rows.append(row_idx)
                            cols.append(base + 3 + q)
                            vals.append(value)
                    row_idx += 1
        Ltemp = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, n_cols))
        return Ltemp.tocsr()

class Tesselation3D(Tesselation):
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        # 1D parameters
        self.n_params = 12
        self.nC = 5*np.prod(nc) # 5 triangle per cell
        self.ndim = 3
        
        # Initialize super class
        super(Tesselation3D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation, direc, override)
    
    def find_verts(self):
        Vx = np.linspace(self.domain_min[0], self.domain_max[0], self.nc[0]+1)
        Vy = np.linspace(self.domain_min[1], self.domain_max[1], self.nc[1]+1)
        Vz = np.linspace(self.domain_min[2], self.domain_max[2], self.nc[2]+1)
        
        # Find cell index and verts for each cell
        cells, verts = [ ], [ ]
        for i in range(self.nc[2]):
            for j in range(self.nc[1]):        
                for k in range(self.nc[0]):
                    ul0 = tuple([Vx[k],Vy[j],Vz[i],1])
                    ur0 = tuple([Vx[k+1],Vy[j],Vz[i],1])
                    ll0 = tuple([Vx[k],Vy[j+1],Vz[i],1])
                    lr0 = tuple([Vx[k+1],Vy[j+1],Vz[i],1])
                    ul1 = tuple([Vx[k],Vy[j],Vz[i+1],1])
                    ur1 = tuple([Vx[k+1],Vy[j],Vz[i+1],1])
                    ll1 = tuple([Vx[k],Vy[j+1],Vz[i+1],1])
                    lr1 = tuple([Vx[k+1],Vy[j+1],Vz[i+1],1])

                    tf=False                    
                    if k%2==0:
                        if (i%2==0 and j%2==1) or  (i%2==1 and j%2==0):
                            tf=True
                    else:
                        if (i%2==0 and j%2==0) or  (i%2==1 and j%2==1):
                            tf=True
                    
                    if tf:
                        ul0,ur0,lr0,ll0 = ur0,lr0,ll0,ul0
                        ul1,ur1,lr1,ll1 = ur1,lr1,ll1,ul1
                    
                    # ORDER MATTERS 
                    verts.append((ll1,ur1,ul0,lr0))  # central part
                    verts.append((ul1,ur1,ll1,ul0))
                    verts.append((lr1,ur1,ll1,lr0))
                    verts.append((ll0,ul0,lr0,ll1))
                    verts.append((ur0,ul0,lr0,ur1))
                    
                    for l in range(5):
                        cells.append((k,j,i,l))
        
        # Convert to array
        self.verts = np.asarray(verts)
        self.cells = cells

    def find_verts_outside(self):
        shared_verts, shared_verts_idx = [ ], [ ]
        # In 3D dist_cond implies it must be the same block of 5 tetrahedra
        for voxel_idx in tqdm(range(self.nC // 5), total=self.nC // 5, desc="Finding outside vertices (3D)"):
            base_idx = voxel_idx * 5
            for i_local in range(5):
                i = base_idx + i_local
                for j_local in range(i_local + 1, 5):
                    j = base_idx + j_local
                    for d in range(self.ndim):
                        vi = self.verts[i]    
                        vj = self.verts[j]
                        upper_cond = sum(vi[:,d]==self.domain_min[d]) == 3 and \
                                     sum(vj[:,d]==self.domain_min[d]) == 3
                        lower_cond = sum(vi[:,d]==self.domain_max[d]) == 3 and \
                                     sum(vj[:,d]==self.domain_max[d]) == 3
                        
                        if upper_cond or lower_cond:
                            vi_h = make_hashable(vi)
                            vj_h = make_hashable(vj)
                            sv = set(vi_h).intersection(vj_h)
                            center = [(v1 + v2) / 2.0 for v1, v2 in zip(vi_h[0], vj_h[0])]
                            center[d] += (-1) if upper_cond else (+1)
                            shared_verts.append(list(sv.union([tuple(center)])))
                            shared_verts_idx.append((i,j))
                            
        # Add to already found pairs
        if shared_verts:
            self.shared_v = np.concatenate((self.shared_v, np.asarray(shared_verts)))
            self.shared_v_idx += shared_verts_idx

            
    def create_zero_boundary_constrains(self):
        xmin, ymin, zmin = self.domain_min
        xmax, ymax, zmax = self.domain_max
        n_cols = 12 * self.nC
        rows, cols, vals = [ ], [ ], [ ]
        row_idx = 0
        for c in tqdm(range(self.nC), total=self.nC, desc="Building boundary constraints (3D)"):
            base = 12 * c
            for v in self.verts[c]:
                if v[0] == xmin or v[0] == xmax:
                    for q, value in enumerate(v):
                        if value != 0.0:
                            rows.append(row_idx)
                            cols.append(base + q)
                            vals.append(value)
                    row_idx += 1
                if v[1] == ymin or v[1] == ymax:
                    for q, value in enumerate(v):
                        if value != 0.0:
                            rows.append(row_idx)
                            cols.append(base + 4 + q)
                            vals.append(value)
                    row_idx += 1
                if v[2] == zmin or v[2] == zmax:
                    for q, value in enumerate(v):
                        if value != 0.0:
                            rows.append(row_idx)
                            cols.append(base + 8 + q)
                            vals.append(value)
                    row_idx += 1
        Ltemp = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, n_cols))
        return Ltemp.tocsr()
                    
if __name__ == "__main__":
    tess1 = Tesselation1D([5], [0], [1], zero_boundary=True, volume_perservation=True)
    tess2 = Tesselation2D([2,2], [0,0], [1,1], zero_boundary=False, volume_perservation=True)
    tess3 = Tesselation3D([2,2,2], [0,0,0], [1,1,1], zero_boundary=True, volume_perservation=False)
