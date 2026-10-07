/**
 * WebGPU engine for 2D CPAB transformations (forward + FG-CPAB gradients).
 *
 * Mirrors cpab.pytorch.transformer._CPABFunction:
 *   encodeAffine    theta -> As            (sparse basis, A = B theta)
 *   encodeExpm      As -> Trels            (expm(h A), Pade(3,3) + squaring)
 *   encodeForward   points -> out          (closed-form per-cell integration)
 *   encodeBackward  gradOut -> moments M   (adjoint sweep, Eq. 8 / Eq. 11)
 *   encodeProject   moments -> gradTheta   (grad = B^T vec(M), Eq. 12)
 *
 * All work is recorded into a caller-provided GPUCommandEncoder so a full
 * forward/backward/optimizer step can be submitted as one command buffer.
 * theta uses the same basis, ordering and scaling as
 * `Cpab(tess, zero_boundary=True)` in Python.
 */
import { buildTessellation2D } from './tessellation2d.js';
import { SHADERS, WORKGROUP_SIZE, MAX_STEPS } from './shaders2d.js';

export { MAX_STEPS };

const PARAMS_BYTES = 64;
const F32 = 4;

function divUp(a, b) {
  return Math.ceil(a / b);
}

async function makePipeline(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    const msg = errors.map((m) => `${label}:${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
    throw new Error(`WGSL compilation failed\n${msg}`);
  }
  return device.createComputePipelineAsync({
    label,
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
}

// Per device: does the compare-exchange backward pass compile? Safari on iOS 26
// translates atomicCompareExchangeWeak into Metal that the compiler rejects;
// there the engine uses the atomicExchange variant. The attempt runs in error
// scopes so that a failure is not also reported as an uncaptured error.
const compareExchangeSupport = new WeakMap();

function supportsCompareExchange(device) {
  if (!compareExchangeSupport.has(device)) {
    compareExchangeSupport.set(device, (async () => {
      device.pushErrorScope('validation');
      device.pushErrorScope('internal');
      const module = device.createShaderModule({ code: SHADERS.BACKWARD, label: 'cpab2d.backward.probe' });
      const compiled = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })
        .then(() => true, () => false);
      const errors = await Promise.all([device.popErrorScope(), device.popErrorScope()]);
      return compiled && errors.every((e) => e === null);
    })());
  }
  return compareExchangeSupport.get(device);
}

export class CPAB2D {
  /**
   * @param {GPUDevice} device
   * @param {object} opts
   * @param {[number, number]} opts.tess   cells per dimension, e.g. [15, 15]
   * @param {number} [opts.nSteps=50]      integration steps (Cpab.params.nstepsolver)
   * @param {boolean} [opts.adaptive=false] Heun correction on cell crossings
   * @param {number} [opts.batch=1]        number of theta vectors
   * @param {'auto'|'exchange'} [opts.atomics='auto']  moment accumulation in the backward
   *                                     pass: compare-exchange where the device compiles it,
   *                                     else atomicExchange; 'exchange' forces the latter.
   *                                     `engine.atomics` reports the one in use.
   */
  static async create(device, opts) {
    const engine = new CPAB2D(device, opts);
    await engine._compile();
    return engine;
  }

  constructor(device, { tess = [16, 16], nSteps = 50, adaptive = false, batch = 1, atomics = 'auto' } = {}) {
    if (!Number.isInteger(nSteps) || nSteps < 1 || nSteps > MAX_STEPS) {
      throw new Error(`nSteps must be an integer in [1, ${MAX_STEPS}], got ${nSteps}`);
    }
    if (!Number.isInteger(batch) || batch < 1) throw new Error(`batch must be >= 1, got ${batch}`);
    this.device = device;
    this.tess = buildTessellation2D(tess[0], tess[1]);
    this.nC = this.tess.nC;
    this.d = this.tess.d;
    this.batch = batch;
    this.nSteps = nSteps;
    this.adaptive = adaptive;
    this.atomics = atomics;
    this.h = Math.fround(1 / nSteps);

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const t = this.tess;
    this.buffers = {
      params: this._buffer(PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'params'),
      theta: this._buffer(batch * this.d * F32, S, 'theta'),
      affine: this._buffer(batch * this.nC * 6 * F32, S, 'As'),
      trels: this._buffer(batch * this.nC * 6 * F32, S, 'Trels'),
      moments: this._buffer(batch * this.nC * 6 * F32, S, 'moments'),
      gradTheta: this._buffer(batch * this.d * F32, S, 'gradTheta'),
      cellDof: this._upload(t.cellDof, 'cellDof'),
      cellCoef: this._upload(t.cellCoef, 'cellCoef'),
      incPtr: this._upload(t.incPtr, 'incPtr'),
      incList: this._upload(t.incList, 'incList'),
      dummy: this._buffer(16, GPUBufferUsage.STORAGE, 'dummy'),
    };
    this._writeParams(this.buffers.params, { nP: 0, batch, batchOffset: 0, pointsBatched: false, gradPoints: false });
  }

  _buffer(size, usage, label) {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label });
  }

  _upload(data, label) {
    const buf = this._buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label);
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  _writeParams(buffer, { nP, batch, batchOffset, pointsBatched, gradPoints }) {
    const u = new Uint32Array(PARAMS_BYTES / 4);
    const f = new Float32Array(u.buffer);
    u.set([
      nP, batch, batchOffset, pointsBatched ? 1 : 0,
      this.nC, this.tess.nx, this.tess.ny, this.nSteps,
      this.adaptive ? 1 : 0, gradPoints ? 1 : 0, this.tess.nActive, this.d,
    ]);
    f[12] = this.h;
    this.device.queue.writeBuffer(buffer, 0, u);
  }

  async _compile() {
    const d = this.device;
    const backwardPipeline = async () => {
      const cas = this.atomics !== 'exchange' && await supportsCompareExchange(d);
      this.atomics = cas ? 'compare-exchange' : 'exchange';
      return makePipeline(d, cas ? SHADERS.BACKWARD : SHADERS.BACKWARD_EXCHANGE, `cpab2d.backward.${this.atomics}`);
    };
    const [affine, expm, forward, backward, project, clear] = await Promise.all([
      makePipeline(d, SHADERS.AFFINE, 'cpab2d.affine'),
      makePipeline(d, SHADERS.EXPM, 'cpab2d.expm'),
      makePipeline(d, SHADERS.FORWARD, 'cpab2d.forward'),
      backwardPipeline(),
      makePipeline(d, SHADERS.PROJECT, 'cpab2d.project'),
      makePipeline(d, SHADERS.CLEAR, 'cpab2d.clear'),
    ]);
    this.pipelines = { affine, expm, forward, backward, project, clear };

    const B = this.buffers;
    this.bindGroups = {
      affine: this._bindGroup(affine, [B.params, B.theta, B.cellDof, B.cellCoef, B.affine]),
      expm: this._bindGroup(expm, [B.params, B.affine, B.trels]),
      project: this._bindGroup(project, [B.params, B.moments, B.cellCoef, B.incPtr, B.incList, B.gradTheta]),
      clearMoments: this._bindGroup(clear, [B.moments]),
    };
  }

  _bindGroup(pipeline, buffers) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  _dispatch(encoder, pipeline, bindGroup, x, y, timestampWrites) {
    const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }

  /** Upload theta values ([k, d] row-major) starting at batch entry `batchIndex`. */
  writeTheta(theta, batchIndex = 0) {
    if (theta.length % this.d !== 0 || batchIndex + theta.length / this.d > this.batch) {
      throw new Error(`theta has ${theta.length} values; expected a multiple of d=${this.d} fitting batch ${this.batch}`);
    }
    this.device.queue.writeBuffer(this.buffers.theta, batchIndex * this.d * F32, theta);
  }

  /**
   * Create the buffers and bind groups for transforming one set of points.
   *
   * @param {object} opts
   * @param {number} opts.nP                 number of points
   * @param {boolean} [opts.pointsBatched]   points are [batchCount, 2, nP] instead of shared [2, nP]
   * @param {number} [opts.batchOffset=0]    first theta entry this set is transformed by
   * @param {number} [opts.batchCount]       number of theta entries (default: all from batchOffset)
   * @param {boolean|GPUBuffer} [opts.gradPoints]  also write dL/dpoints in encodeBackward
   *                                     (into this buffer if one is given)
   * @param {GPUBuffer} [opts.points]        reuse an existing points buffer (e.g. another set's `out`)
   * @param {GPUBuffer} [opts.gradOut]       reuse an existing dL/dout buffer
   *
   * Chaining point sets (composition) needs no copies: pass the previous set's
   * `out` as `points` and its `gradOut` as `gradPoints`.
   */
  createPointSet({
    nP, pointsBatched = false, batchOffset = 0, batchCount, gradPoints = false, points, gradOut,
  } = {}) {
    const count = batchCount ?? this.batch - batchOffset;
    if (!Number.isInteger(nP) || nP < 1) throw new Error(`nP must be >= 1, got ${nP}`);
    if (batchOffset < 0 || count < 1 || batchOffset + count > this.batch) {
      throw new Error(`batch range [${batchOffset}, ${batchOffset + count}) outside engine batch ${this.batch}`);
    }
    const maxBinding = this.device.limits.maxStorageBufferBindingSize;
    const outBytes = count * 2 * nP * F32;
    if (outBytes > maxBinding) {
      throw new Error(`point set needs ${outBytes} bytes per buffer; device limit is ${maxBinding}`);
    }
    const maxPoints = this.device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
    if (nP > maxPoints) throw new Error(`nP=${nP} exceeds ${maxPoints} points per dispatch; split the point set`);
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const ps = {
      nP, batchOffset, batchCount: count, pointsBatched,
      params: this._buffer(PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'pointParams'),
      points: points ?? this._buffer((pointsBatched ? count : 1) * 2 * nP * F32, S, 'points'),
      out: this._buffer(outBytes, S, 'out'),
      gradOut: gradOut ?? this._buffer(outBytes, S, 'gradOut'),
      gradPoints: gradPoints === true ? this._buffer(outBytes, S, 'gradPoints') : (gradPoints || null),
      owned: ['params', 'out', ...(points ? [] : ['points']), ...(gradOut ? [] : ['gradOut']),
        ...(gradPoints === true ? ['gradPoints'] : [])],
    };
    this._writeParams(ps.params, { nP, batch: count, batchOffset, pointsBatched, gradPoints: !!ps.gradPoints });
    const B = this.buffers;
    ps.forwardBG = this._bindGroup(this.pipelines.forward, [ps.params, ps.points, B.trels, B.affine, ps.out]);
    ps.backwardBG = this._bindGroup(this.pipelines.backward,
      [ps.params, ps.points, B.affine, ps.gradOut, B.moments, ps.gradPoints ?? B.dummy]);
    return ps;
  }

  destroyPointSet(ps) {
    for (const key of ps.owned) ps[key].destroy();
  }

  encodeAffine(encoder, timestampWrites) {
    this._dispatch(encoder, this.pipelines.affine, this.bindGroups.affine,
      divUp(this.batch * this.nC, WORKGROUP_SIZE), 1, timestampWrites);
  }

  encodeExpm(encoder, timestampWrites) {
    this._dispatch(encoder, this.pipelines.expm, this.bindGroups.expm,
      divUp(this.batch * this.nC, WORKGROUP_SIZE), 1, timestampWrites);
  }

  encodeForward(encoder, ps, timestampWrites) {
    this._dispatch(encoder, this.pipelines.forward, ps.forwardBG,
      divUp(ps.nP, WORKGROUP_SIZE), ps.batchCount, timestampWrites);
  }

  /** Moments are zeroed first unless `accumulate` is set (e.g. for several point sets). */
  encodeBackward(encoder, ps, timestampWrites, { accumulate = false } = {}) {
    if (!accumulate) this.encodeClearMoments(encoder);
    this._dispatch(encoder, this.pipelines.backward, ps.backwardBG,
      divUp(ps.nP, WORKGROUP_SIZE), ps.batchCount, timestampWrites);
  }

  encodeClearMoments(encoder, timestampWrites) {
    this._dispatch(encoder, this.pipelines.clear, this.bindGroups.clearMoments,
      divUp(this.buffers.moments.size / 4, WORKGROUP_SIZE), 1, timestampWrites);
  }

  encodeProject(encoder, timestampWrites) {
    this._dispatch(encoder, this.pipelines.project, this.bindGroups.project,
      divUp(this.tess.nActive, WORKGROUP_SIZE), this.batch, timestampWrites);
  }

  destroy() {
    for (const buf of Object.values(this.buffers)) buf.destroy();
  }
}
