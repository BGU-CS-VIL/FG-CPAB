/**
 * 2D CPAB tessellation and sparse CPA basis.
 *
 * Port of cpab/core/tesselation.py (Tesselation2D.find_verts,
 * create_nodal_prolongation, _normalize_basis_columns) for
 * zero_boundary=True, volume_perservation=False.
 *
 * In that setting every continuity and boundary constraint in C = L N vanishes
 * identically (shared vertices make the field continuous, and boundary vertices
 * carry no DOFs), so the Python basis is exactly the column-normalised nodal map
 * B = N D^-1. theta therefore has the same ordering and scaling as
 * `Cpab(tess, zero_boundary=True)`: theta[2*v + k] = D_v * u_v[k], where u_v is
 * the velocity at the v-th active (interior) vertex.
 */

// np.linspace(0, 1, n + 1): i * (1 / n), with the last entry pinned to 1.
function linspace01(n) {
  const step = 1 / n;
  const out = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) out[i] = i * step;
  out[n] = 1;
  return out;
}

// Inverse of the 3x3 matrix X = [x~_0 x~_1 x~_2] whose columns are homogeneous
// vertices. Row j of X^-1 holds the affine coefficients of vertex j's hat function.
function invertVerts(xy, v0, v1, v2) {
  const a = xy[2 * v0], b = xy[2 * v1], c = xy[2 * v2];
  const d = xy[2 * v0 + 1], e = xy[2 * v1 + 1], f = xy[2 * v2 + 1];
  // X = [[a b c], [d e f], [1 1 1]]
  const c00 = e - f, c01 = c - b, c02 = b * f - c * e;
  const c10 = f - d, c11 = a - c, c12 = c * d - a * f;
  const c20 = d - e, c21 = b - a, c22 = a * e - b * d;
  const det = a * c00 + b * c10 + c * c20;
  const s = 1 / det;
  return [
    c00 * s, c01 * s, c02 * s,
    c10 * s, c11 * s, c12 * s,
    c20 * s, c21 * s, c22 * s,
  ];
}

export function buildTessellation2D(nx, ny) {
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 1 || ny < 1) {
    throw new Error(`tessellation must be positive integers, got [${nx}, ${ny}]`);
  }
  const nC = 4 * nx * ny;
  const nCorners = (nx + 1) * (ny + 1);
  const nVerts = nCorners + nx * ny;
  const corner = (j, i) => j + i * (nx + 1);
  const center = (j, i) => nCorners + j + i * nx;

  const Vx = linspace01(nx);
  const Vy = linspace01(ny);
  const xy = new Float64Array(2 * nVerts);
  for (let i = 0; i <= ny; i++) {
    for (let j = 0; j <= nx; j++) {
      const v = corner(j, i);
      xy[2 * v] = Vx[j];
      xy[2 * v + 1] = Vy[i];
    }
  }
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const v = center(j, i);
      xy[2 * v] = (Vx[j] + Vx[j + 1]) / 2;
      xy[2 * v + 1] = (Vy[i] + Vy[i + 1]) / 2;
    }
  }

  // Same triangle order as Tesselation2D.find_verts ("order matters!"), which is
  // also the order assumed by the cell lookup in the kernels:
  // cell = 4 * (j + i * nx) + {0: top, 1: right, 2: bottom, 3: left}.
  const cellVerts = new Int32Array(3 * nC);
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const ul = corner(j, i), ur = corner(j + 1, i);
      const ll = corner(j, i + 1), lr = corner(j + 1, i + 1);
      const ce = center(j, i);
      const c = 4 * (j + i * nx);
      cellVerts.set([ce, ul, ur, ce, ur, lr, ce, lr, ll, ce, ll, ul], 3 * c);
    }
  }

  // Active vertices in first-seen order over cells (create_nodal_prolongation).
  const isBoundary = (v) => {
    if (v >= nCorners) return false;
    const j = v % (nx + 1), i = Math.floor(v / (nx + 1));
    return j === 0 || j === nx || i === 0 || i === ny;
  };
  const seen = new Uint8Array(nVerts);
  const activeIndex = new Int32Array(nVerts).fill(-1);
  const activeVerts = [];
  for (let s = 0; s < 3 * nC; s++) {
    const v = cellVerts[s];
    if (seen[v]) continue;
    seen[v] = 1;
    if (!isBoundary(v)) {
      activeIndex[v] = activeVerts.length;
      activeVerts.push(v);
    }
  }
  const nActive = activeVerts.length;
  const d = 2 * nActive;

  // Per-cell hat-function coefficients X_c^-1 and the column norms of N.
  // Both DOFs (k = 0, 1) of a vertex have identical columns up to the row block,
  // so a single norm per vertex suffices.
  const xinv = new Float64Array(9 * nC);
  const normSq = new Float64Array(nActive);
  for (let c = 0; c < nC; c++) {
    const m = invertVerts(xy, cellVerts[3 * c], cellVerts[3 * c + 1], cellVerts[3 * c + 2]);
    xinv.set(m, 9 * c);
    for (let j = 0; j < 3; j++) {
      const a = activeIndex[cellVerts[3 * c + j]];
      if (a < 0) continue;
      normSq[a] += m[3 * j] ** 2 + m[3 * j + 1] ** 2 + m[3 * j + 2] ** 2;
    }
  }
  const vertexNorm = normSq.map(Math.sqrt);

  // GPU tables. Each cell's three vertices are sorted by active index (boundary
  // vertices last) so the A = B theta sum runs in the same column order as the
  // coalesced torch sparse matrix. cellCoef[c, slot, q] = X_c^-1[j, q] / D_v,
  // i.e. exactly the float32 values of B.
  const cellDof = new Int32Array(4 * nC).fill(-1);
  const cellCoef = new Float32Array(9 * nC);
  const incidence = Array.from({ length: nActive }, () => []);
  for (let c = 0; c < nC; c++) {
    const order = [0, 1, 2]
      .map((j) => ({ j, a: activeIndex[cellVerts[3 * c + j]] }))
      .sort((p, q) => (p.a < 0) - (q.a < 0) || p.a - q.a);
    order.forEach(({ j, a }, slot) => {
      if (a < 0) return;
      cellDof[4 * c + slot] = a;
      for (let q = 0; q < 3; q++) {
        cellCoef[9 * c + 3 * slot + q] = xinv[9 * c + 3 * j + q] / vertexNorm[a];
      }
      incidence[a].push(4 * c + slot);
    });
  }
  // CSR incidence (active vertex -> packed cell * 4 + slot, cells ascending) for
  // the gradient projection grad_theta = B^T vec(M).
  const incPtr = new Uint32Array(nActive + 1);
  for (let a = 0; a < nActive; a++) incPtr[a + 1] = incPtr[a] + incidence[a].length;
  const incList = new Uint32Array(incPtr[nActive]);
  for (let a = 0; a < nActive; a++) incList.set(incidence[a], incPtr[a]);

  return {
    nx, ny, nC, nVerts, nActive, d,
    xy: Float32Array.from(xy),
    cellVerts,
    activeVerts: Int32Array.from(activeVerts),
    activeIndex,
    vertexNorm,
    cellDof,
    cellCoef,
    incPtr,
    incList,
  };
}

/** Nodal velocities u [nActive, 2] (interleaved) -> theta [d] in the Python basis. */
export function thetaFromVelocities(tess, u, out = new Float32Array(tess.d)) {
  for (let a = 0; a < tess.nActive; a++) {
    out[2 * a] = u[2 * a] * tess.vertexNorm[a];
    out[2 * a + 1] = u[2 * a + 1] * tess.vertexNorm[a];
  }
  return out;
}

/** theta [d] in the Python basis -> nodal velocities u [nActive, 2] (interleaved). */
export function velocitiesFromTheta(tess, theta, out = new Float32Array(tess.d)) {
  for (let a = 0; a < tess.nActive; a++) {
    out[2 * a] = theta[2 * a] / tess.vertexNorm[a];
    out[2 * a + 1] = theta[2 * a + 1] / tess.vertexNorm[a];
  }
  return out;
}
