try:
    import cPickle as pkl
except:
    import pickle as pkl
import os
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


class params:
    """Simple parameter container class."""
    pass
    
    def __repr__(self):
        return str(self.__dict__)


def drop_zero_rows(A):
    """Remove all-zero rows from a sparse or dense matrix."""
    if sp.issparse(A):
        A = A.tocsr()
        nnz_per_row = np.diff(A.indptr)
        keep = np.flatnonzero(nnz_per_row)
        if keep.size == A.shape[0]:
            return A
        return A[keep]

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    keep = np.flatnonzero(np.any(np.abs(A) > 0, axis=1))
    return A[keep]


def null(A, eps=1e-6, zero_tol=None):
    """Find the null space of a matrix and return it as sparse CSR."""
    if sp.issparse(A):
        A = A.toarray()
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    if A.shape[1] == 0:
        return sp.csr_matrix((0, 0))

    u, s, vh = la.svd(A)
    padding = np.max([0, np.shape(A)[-1] - np.shape(s)[0]])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
    null_space = np.compress(null_mask, vh, axis=0)
    null_space = np.transpose(null_space)

    tol = eps if zero_tol is None else zero_tol
    null_space[np.abs(null_space) <= tol] = 0.0
    null_sparse = sp.csr_matrix(null_space)
    null_sparse.eliminate_zeros()
    return null_sparse


def make_hashable(arr):
    """Make an array hashable for use with set() and intersection()."""
    return tuple([tuple(r.tolist()) for r in arr])


def load_obj(name):
    """Load a variable from a pickle file."""
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)


def save_obj(obj, name):
    """Save a variable to a pickle file."""
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def get_dir(file):
    """Get directory of the input file."""
    return os.path.dirname(os.path.realpath(file))


def create_dir(direc):
    """Create a directory if it does not already exist."""
    if not os.path.exists(direc):
        os.mkdir(direc)


def check_if_file_exist(file):
    """Check if a file exists."""
    return os.path.isfile(file)
