/**
 * WGSL kernels for 2D CPAB, ported from cpab/core/cpab_ops_2d.cu.
 *
 * Buffer layouts match the CUDA extension exactly:
 *   theta      [batch, d]
 *   As, Trels  [batch, nC, 6]   (row-major 2x3 affine blocks)
 *   points     [2, nP] or [batch, 2, nP]   (x row, then y row)
 *   out, grad  [batch, 2, nP]
 *   moments    [batch, nC, 6]   (f32 bits; accumulated with CAS atomics)
 */

export const WORKGROUP_SIZE = 256;
export const CKPT_SEGMENT_SIZE = 8;
export const MAX_STEPS = 128;

const PARAMS = /* wgsl */ `
struct Params {
  nP: u32,
  batch: u32,          // batch entries handled by this dispatch
  batchOffset: u32,    // first theta / As entry used by this dispatch
  pointsBatched: u32,  // 1: points are [batch, 2, nP]; 0: shared [2, nP]
  nC: u32,
  ncx: u32,
  ncy: u32,
  nSteps: u32,
  adaptive: u32,
  writeGradPoints: u32,
  nActive: u32,
  d: u32,
  h: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
`;

const AFFINE_HELPERS = /* wgsl */ `
struct Affine { r0: vec3<f32>, r1: vec3<f32> };

fn apply(M: Affine, p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(fma(M.r0.x, p.x, fma(M.r0.y, p.y, M.r0.z)),
                   fma(M.r1.x, p.x, fma(M.r1.y, p.y, M.r1.z)));
}
`;

// Port of cuda_findcellidx_2D_fast: [0,1]^2 split into ncx x ncy squares of
// 4 triangles each; points outside the domain map to the nearest boundary triangle.
const FIND_CELL = /* wgsl */ `
fn find_cell(p: vec2<f32>) -> u32 {
  let fx = f32(P.ncx);
  let fy = f32(P.ncy);
  let eps = 1e-7;
  let xc = min(max(p.x, 0.0), 1.0 - eps);
  let yc = min(max(p.y, 0.0), 1.0 - eps);
  let cx = min(i32(floor(xc * fx)), i32(P.ncx) - 1);
  let cy = min(i32(floor(yc * fy)), i32(P.ncy) - 1);
  let lx = fma(xc, fx, -f32(cx));
  let ly = fma(yc, fy, -f32(cy));
  let base = u32(cx + cy * i32(P.ncx)) * 4u;

  if (p.x <= 0.0) {
    if (p.y <= 0.0 && p.y * fy < p.x * fx) { return base; }
    if (p.y >= 1.0 && (p.y * fy - f32(P.ncy)) > -p.x * fx) { return base + 2u; }
    return base + 3u;
  }
  if (p.x >= 1.0) {
    if (p.y <= 0.0 && -p.y * fy > (p.x * fx - f32(P.ncx))) { return base; }
    if (p.y >= 1.0 && (p.y * fy - f32(P.ncy)) > (p.x * fx - f32(P.ncx))) { return base + 2u; }
    return base + 1u;
  }
  if (p.y <= 0.0) { return base; }
  if (p.y >= 1.0) { return base + 2u; }

  // bit1 = x < y, bit0 = (1 - x) < y; raw 0,1,2,3 -> tri 0,1,3,2.
  let bit1 = select(0u, 1u, lx < ly);
  let bit0 = select(0u, 1u, (1.0 - lx) < ly);
  var tri = (bit1 << 1u) | bit0;
  tri = tri ^ ((tri >> 1u) & 1u);
  return base + tri;
}
`;

// theta -> As (A = B theta), one invocation per (batch, cell).
const AFFINE = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> theta: array<f32>;
@group(0) @binding(2) var<storage, read> cellDof: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> cellCoef: array<f32>;
@group(0) @binding(4) var<storage, read_write> As: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= P.batch * P.nC) { return; }
  let b = idx / P.nC;
  let c = idx - b * P.nC;
  let dof = cellDof[c];
  let tb = (b + P.batchOffset) * P.d;
  var r0 = vec3<f32>(0.0);
  var r1 = vec3<f32>(0.0);
  for (var s = 0u; s < 3u; s++) {
    let a = dof[s];
    if (a < 0) { continue; }
    let o = c * 9u + s * 3u;
    let w = vec3<f32>(cellCoef[o], cellCoef[o + 1u], cellCoef[o + 2u]);
    r0 = r0 + w * theta[tb + 2u * u32(a)];
    r1 = r1 + w * theta[tb + 2u * u32(a) + 1u];
  }
  let out = ((b + P.batchOffset) * P.nC + c) * 6u;
  As[out] = r0.x; As[out + 1u] = r0.y; As[out + 2u] = r0.z;
  As[out + 3u] = r1.x; As[out + 4u] = r1.y; As[out + 5u] = r1.z;
}
`;

// Port of cpab_cuda_kernel_expm3x3: Trels = expm(h * [A; 0 0 0]) via Pade(3,3)
// with scaling and squaring, exploiting the affine [0 0 1] last row.
const EXPM = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> As: array<f32>;
@group(0) @binding(2) var<storage, read_write> Trels: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= P.batch * P.nC) { return; }
  let o = (P.batchOffset * P.nC + idx) * 6u;
  let dT = P.h;
  let a00 = dT * As[o];      let a01 = dT * As[o + 1u]; let a02 = dT * As[o + 2u];
  let a10 = dT * As[o + 3u]; let a11 = dT * As[o + 4u]; let a12 = dT * As[o + 5u];

  let norm = sqrt(a00 * a00 + a01 * a01 + a02 * a02 + a10 * a10 + a11 * a11 + a12 * a12);
  var nSq = 0i;
  if (norm > 0.5) { nSq = i32(ceil(log2(norm * 2.0))); }
  let scale = ldexp(1.0, -nSq);
  let m00 = a00 * scale; let m01 = a01 * scale; let m02 = a02 * scale;
  let m10 = a10 * scale; let m11 = a11 * scale; let m12 = a12 * scale;

  let mm00 = m00 * m00 + m01 * m10;
  let mm01 = m00 * m01 + m01 * m11;
  let mm02 = m00 * m02 + m01 * m12;
  let mm10 = m10 * m00 + m11 * m10;
  let mm11 = m10 * m01 + m11 * m11;
  let mm12 = m10 * m02 + m11 * m12;

  let inv12 = 1.0 / 12.0;
  let p00 = 1.0 + 0.5 * m00 + inv12 * mm00;
  let p01 =       0.5 * m01 + inv12 * mm01;
  let p02 =       0.5 * m02 + inv12 * mm02;
  let p10 =       0.5 * m10 + inv12 * mm10;
  let p11 = 1.0 + 0.5 * m11 + inv12 * mm11;
  let p12 =       0.5 * m12 + inv12 * mm12;
  let q00 = 1.0 - 0.5 * m00 + inv12 * mm00;
  let q01 =     - 0.5 * m01 + inv12 * mm01;
  let q02 =     - 0.5 * m02 + inv12 * mm02;
  let q10 =     - 0.5 * m10 + inv12 * mm10;
  let q11 = 1.0 - 0.5 * m11 + inv12 * mm11;
  let q12 =     - 0.5 * m12 + inv12 * mm12;

  // R = Q^-1 P using the 2x2 block inverse of Q.
  let invDet = 1.0 / (q00 * q11 - q01 * q10);
  let qi00 =  q11 * invDet;
  let qi01 = -q01 * invDet;
  let qi10 = -q10 * invDet;
  let qi11 =  q00 * invDet;
  let qi02 = -(qi00 * q02 + qi01 * q12);
  let qi12 = -(qi10 * q02 + qi11 * q12);

  var r00 = qi00 * p00 + qi01 * p10;
  var r01 = qi00 * p01 + qi01 * p11;
  var r02 = qi00 * p02 + qi01 * p12 + qi02;
  var r10 = qi10 * p00 + qi11 * p10;
  var r11 = qi10 * p01 + qi11 * p11;
  var r12 = qi10 * p02 + qi11 * p12 + qi12;

  for (var i = 0i; i < nSq; i++) {
    let n00 = r00 * r00 + r01 * r10;
    let n01 = r00 * r01 + r01 * r11;
    let n02 = r00 * r02 + r01 * r12 + r02;
    let n10 = r10 * r00 + r11 * r10;
    let n11 = r10 * r01 + r11 * r11;
    let n12 = r10 * r02 + r11 * r12 + r12;
    r00 = n00; r01 = n01; r02 = n02;
    r10 = n10; r11 = n11; r12 = n12;
  }

  Trels[o] = r00;      Trels[o + 1u] = r01; Trels[o + 2u] = r02;
  Trels[o + 3u] = r10; Trels[o + 4u] = r11; Trels[o + 5u] = r12;
}
`;

// Port of cpab_cuda_kernel_forward_2D_optimized: closed-form per-cell steps
// p <- Trels_c p, with optional Heun correction when a step crosses a cell.
const FORWARD = PARAMS + AFFINE_HELPERS + FIND_CELL + /* wgsl */ `
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read> Trels: array<f32>;
@group(0) @binding(3) var<storage, read> As: array<f32>;
@group(0) @binding(4) var<storage, read_write> outPoints: array<f32>;

fn loadT(c: u32) -> Affine {
  let o = c * 6u;
  return Affine(vec3<f32>(Trels[o], Trels[o + 1u], Trels[o + 2u]),
                vec3<f32>(Trels[o + 3u], Trels[o + 4u], Trels[o + 5u]));
}

fn loadA(c: u32) -> Affine {
  let o = c * 6u;
  return Affine(vec3<f32>(As[o], As[o + 1u], As[o + 2u]),
                vec3<f32>(As[o + 3u], As[o + 4u], As[o + 5u]));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let b = gid.y;
  if (n >= P.nP || b >= P.batch) { return; }
  let inOff = P.pointsBatched * b * P.nP * 2u;
  var p = vec2<f32>(points[inOff + n], points[inOff + n + P.nP]);
  let cellBase = (b + P.batchOffset) * P.nC;
  let hHalf = vec2<f32>(0.5 * P.h);

  for (var t = 0u; t < P.nSteps; t++) {
    let c0 = find_cell(p);
    let q = apply(loadT(cellBase + c0), p);
    if (P.adaptive != 0u) {
      let c1 = find_cell(q);
      if (c1 != c0) {
        let v0 = apply(loadA(cellBase + c0), p);
        let v1 = apply(loadA(cellBase + c1), q);
        p = fma(hHalf, v0 + v1, p);
        continue;
      }
    }
    p = q;
  }

  let o = b * P.nP * 2u;
  outPoints[o + n] = p.x;
  outPoints[o + n + P.nP] = p.y;
}
`;

// Port of cpab_cuda_kernel_fused_backward: segmented-checkpoint replay of the
// midpoint trajectory, discrete adjoint sweep, and cell-wise moment accumulation
// (Eq. 8 and Eq. 11). As in CUDA, a thread accumulates locally while it stays in
// one cell and flushes when the cell changes. WGSL has no float atomics, so a
// flush adds f32 bits with integer atomics, in one of two ways (ACCUMULATE_*).
const BACKWARD_COMMON = PARAMS + AFFINE_HELPERS + FIND_CELL + /* wgsl */ `
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read> As: array<f32>;
@group(0) @binding(3) var<storage, read> gradOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> moments: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> gradPoints: array<f32>;

const SEG: u32 = ${CKPT_SEGMENT_SIZE}u;
const MAX_SEGS: u32 = (${MAX_STEPS}u + SEG - 1u) / SEG;
const NO_CELL: u32 = 0xffffffffu;

fn loadA(c: u32) -> Affine {
  let o = c * 6u;
  return Affine(vec3<f32>(As[o], As[o + 1u], As[o + 2u]),
                vec3<f32>(As[o + 3u], As[o + 4u], As[o + 5u]));
}

// cpab_forward_step_adaptive, with the start cell c0 already looked up.
fn forward_step(p: vec2<f32>, cellBase: u32, c0: u32) -> vec2<f32> {
  let h = vec2<f32>(P.h);
  let hHalf = vec2<f32>(0.5 * P.h);
  let A0 = loadA(cellBase + c0);
  let q = fma(h, apply(A0, fma(hHalf, apply(A0, p), p)), p);
  if (P.adaptive == 0u) { return q; }
  let c1 = find_cell(q);
  if (c1 == c0) { return q; }
  let v0 = apply(A0, p);
  let v1 = apply(loadA(cellBase + c1), q);
  return fma(hHalf, v0 + v1, p);
}

fn backward_point(n: u32, b: u32, momentBase: u32) {
  let h = P.h;
  let hHalf = 0.5 * h;
  let inOff = P.pointsBatched * b * P.nP * 2u;
  var p = vec2<f32>(points[inOff + n], points[inOff + n + P.nP]);
  let cellBase = (b + P.batchOffset) * P.nC;

  // Phase 1: forward pass, keeping only segment-boundary checkpoints.
  let nSeg = (P.nSteps + SEG - 1u) / SEG;
  var ckpt: array<vec2<f32>, MAX_SEGS + 1u>;
  ckpt[0] = p;
  var ci = 1u;
  for (var t = 0u; t < P.nSteps; t++) {
    p = forward_step(p, cellBase, find_cell(p));
    if ((t + 1u) % SEG == 0u) {
      ckpt[ci] = p;
      ci++;
    }
  }

  // Phase 2: replay each segment from its checkpoint, then sweep it backward.
  let gi = b * 2u * P.nP + n;
  var lam = vec2<f32>(gradOut[gi], gradOut[gi + P.nP]);
  var prev = NO_CELL;
  var m0 = vec3<f32>(0.0);
  var m1 = vec3<f32>(0.0);
  var segP: array<vec2<f32>, SEG + 1u>;
  var segC: array<u32, SEG>;

  for (var sg = i32(nSeg) - 1; sg >= 0; sg--) {
    let s0 = u32(sg) * SEG;
    let len = min(s0 + SEG, P.nSteps) - s0;
    p = ckpt[sg];
    segP[0] = p;
    for (var s = 0u; s < len; s++) {
      let c = find_cell(p);
      segC[s] = c;
      p = forward_step(p, cellBase, c);
      segP[s + 1u] = p;
    }

    for (var s = i32(len) - 1; s >= 0; s--) {
      p = segP[s];
      let c = segC[s];
      if (c != prev && prev != NO_CELL) {
        flush(momentBase, prev, m0, m1);
        m0 = vec3<f32>(0.0);
        m1 = vec3<f32>(0.0);
      }
      prev = c;

      // cpab_accumulate_moments_and_adjoint_2D
      let A = loadA(cellBase + c);
      let pMid = fma(vec2<f32>(hHalf), apply(A, p), p);
      var atl = vec2<f32>(A.r0.x * lam.x + A.r1.x * lam.y,
                          A.r0.y * lam.x + A.r1.y * lam.y);
      let wMid = h * lam;
      let wP = hHalf * h * atl;
      m0 = m0 + wMid.x * vec3<f32>(pMid, 1.0);
      m1 = m1 + wMid.y * vec3<f32>(pMid, 1.0);
      m0 = m0 + wP.x * vec3<f32>(p, 1.0);
      m1 = m1 + wP.y * vec3<f32>(p, 1.0);

      let lamMid = lam + hHalf * atl;
      atl = vec2<f32>(A.r0.x * lamMid.x + A.r1.x * lamMid.y,
                      A.r0.y * lamMid.x + A.r1.y * lamMid.y);
      lam = lam + h * atl;
    }
  }

  if (prev != NO_CELL) { flush(momentBase, prev, m0, m1); }
  if (P.writeGradPoints != 0u) {
    gradPoints[gi] = lam.x;
    gradPoints[gi + P.nP] = lam.y;
  }
}
`;

// WGSL has no __match_any_sync (the CUDA warp aggregation) either, so flushes go
// into a per-workgroup hash table in shared memory (CAS loops on f32 bits), which
// is added to global memory once per cell at the end. Neighbouring points (e.g.
// an image grid) hit the same few cells, and contended global CAS loops would
// otherwise serialise.
const ACCUMULATE_CAS = /* wgsl */ `
const WG_SLOTS: u32 = 128u;   // power of two
const SLOT_SHIFT: u32 = 25u;  // 32 - log2(WG_SLOTS)
const MAX_PROBES: u32 = 8u;

var<workgroup> slotKey: array<atomic<u32>, WG_SLOTS>;       // cell + 1, 0 = empty
var<workgroup> slotVal: array<atomic<u32>, WG_SLOTS * 6u>;  // f32 bits

fn global_add(i: u32, v: f32) {
  if (v == 0.0) { return; }
  var old = atomicLoad(&moments[i]);
  loop {
    let r = atomicCompareExchangeWeak(&moments[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn shared_add(i: u32, v: f32) {
  if (v == 0.0) { return; }
  var old = atomicLoad(&slotVal[i]);
  loop {
    let r = atomicCompareExchangeWeak(&slotVal[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

// Add one cell's partial moments to the workgroup table (open addressing), or
// directly to global memory if the probe sequence is full.
fn flush(momentBase: u32, cell: u32, m0: vec3<f32>, m1: vec3<f32>) {
  let key = cell + 1u;
  var slot = (cell * 2654435761u) >> SLOT_SHIFT;
  var probes = 0u;
  loop {
    let r = atomicCompareExchangeWeak(&slotKey[slot], 0u, key);
    if (r.exchanged || r.old_value == key) {
      let o = slot * 6u;
      shared_add(o, m0.x);
      shared_add(o + 1u, m0.y);
      shared_add(o + 2u, m0.z);
      shared_add(o + 3u, m1.x);
      shared_add(o + 4u, m1.y);
      shared_add(o + 5u, m1.z);
      return;
    }
    if (r.old_value != 0u) {  // taken by another cell (else: spurious weak failure, retry)
      probes++;
      if (probes == MAX_PROBES) { break; }
      slot = (slot + 1u) & (WG_SLOTS - 1u);
    }
  }
  let o = momentBase + cell * 6u;
  global_add(o, m0.x);
  global_add(o + 1u, m0.y);
  global_add(o + 2u, m0.z);
  global_add(o + 3u, m1.x);
  global_add(o + 4u, m1.y);
  global_add(o + 5u, m1.z);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let b = gid.y;  // uniform across the workgroup
  let momentBase = (b + P.batchOffset) * P.nC * 6u;
  if (gid.x < P.nP && b < P.batch) {
    backward_point(gid.x, b, momentBase);
  }
  workgroupBarrier();
  for (var s = lid; s < WG_SLOTS; s += ${WORKGROUP_SIZE}u) {
    let key = atomicLoad(&slotKey[s]);
    if (key == 0u) { continue; }
    let o = momentBase + (key - 1u) * 6u;
    for (var k = 0u; k < 6u; k++) {
      global_add(o + k, bitcast<f32>(atomicLoad(&slotVal[s * 6u + k])));
    }
  }
}
`;

// Fallback for WebKit on iOS 26, whose Metal translation of
// atomicCompareExchangeWeak does not compile. The float add uses atomicExchange
// only: a thread deposits its running sum and takes back whatever it displaced
// (another thread's deposit), until a deposit lands on zero. Every value is
// always either in memory or held by exactly one thread, so nothing is lost.
// Claiming a hash-table slot needs CAS, so flushes go straight to global memory.
const ACCUMULATE_EXCHANGE = /* wgsl */ `
fn global_add(i: u32, v: f32) {
  if (v == 0.0) { return; }
  var acc = v;
  loop {
    let prev = bitcast<f32>(atomicExchange(&moments[i], bitcast<u32>(acc)));
    if (prev == 0.0) { break; }
    acc = bitcast<f32>(atomicExchange(&moments[i], 0u)) + prev;
  }
}

fn flush(momentBase: u32, cell: u32, m0: vec3<f32>, m1: vec3<f32>) {
  let o = momentBase + cell * 6u;
  global_add(o, m0.x);
  global_add(o + 1u, m0.y);
  global_add(o + 2u, m0.z);
  global_add(o + 3u, m1.x);
  global_add(o + 4u, m1.y);
  global_add(o + 5u, m1.z);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.y;
  if (gid.x < P.nP && b < P.batch) {
    backward_point(gid.x, b, (b + P.batchOffset) * P.nC * 6u);
  }
}
`;

const BACKWARD = BACKWARD_COMMON + ACCUMULATE_CAS;
const BACKWARD_EXCHANGE = BACKWARD_COMMON + ACCUMULATE_EXCHANGE;

// Zero a buffer from a compute pass (cheaper than clearBuffer between compute
// passes on some backends, e.g. Metal, where it forces an encoder switch).
const CLEAR = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&buf)) { buf[gid.x] = 0u; }
}
`;

// grad_theta = B^T vec(M) as a per-vertex gather over incident cells (no atomics).
const PROJECT = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> moments: array<f32>;
@group(0) @binding(2) var<storage, read> cellCoef: array<f32>;
@group(0) @binding(3) var<storage, read> incPtr: array<u32>;
@group(0) @binding(4) var<storage, read> incList: array<u32>;
@group(0) @binding(5) var<storage, read_write> gradTheta: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  let b = gid.y + P.batchOffset;
  if (v >= P.nActive || gid.y >= P.batch) { return; }
  var gx = 0.0;
  var gy = 0.0;
  for (var e = incPtr[v]; e < incPtr[v + 1u]; e++) {
    let cs = incList[e];
    let c = cs >> 2u;
    let co = c * 9u + (cs & 3u) * 3u;
    let mo = (b * P.nC + c) * 6u;
    gx = gx + cellCoef[co] * moments[mo];
    gx = gx + cellCoef[co + 1u] * moments[mo + 1u];
    gx = gx + cellCoef[co + 2u] * moments[mo + 2u];
    gy = gy + cellCoef[co] * moments[mo + 3u];
    gy = gy + cellCoef[co + 1u] * moments[mo + 4u];
    gy = gy + cellCoef[co + 2u] * moments[mo + 5u];
  }
  gradTheta[b * P.d + 2u * v] = gx;
  gradTheta[b * P.d + 2u * v + 1u] = gy;
}
`;

export const SHADERS = { AFFINE, EXPM, FORWARD, BACKWARD, BACKWARD_EXCHANGE, PROJECT, CLEAR };
