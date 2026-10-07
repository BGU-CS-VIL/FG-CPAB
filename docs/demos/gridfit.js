/**
 * Option 3 — fitting point correspondences: examples/rectangular_grid_diffeomorphism_2d.py.
 *
 * Same setup as the script: a grid of LINES horizontal and LINES vertical lines
 * (PER_LINE points each) is pushed through phi_theta (Cpab([24, 24])),
 * and Adam minimizes
 *   L = mean |phi_theta(x) - y|^2
 * against a swirl y, a rotation about the centre by 0.8 exp(-8 r^2). Every
 * iteration is recorded on the GPU: theta -> A -> expm -> forward -> dL/dphi ->
 * adjoint moment sweep (Eq. 11) -> sparse projection (Eq. 12) -> Adam.
 *
 * A second engine holds [t theta, -t theta] and chains the two flows, so the
 * panels show phi_{t theta}(x) and phi_{-t theta}(phi_{t theta}(x)): the inverse
 * of a CPAB map is the same flow with negated parameters.
 */
import { CPAB2D } from '../src/cpab2d.js';
import {
  getDevice, showFallback, configureCanvas, autoResize, el, mulberry32, gaussian, createAdam, adamParams,
  drawLogCurve,
} from './common.js';

const LINES = 17;            // grid lines per direction
const PER_LINE = 100;        // points per line
const N = 2 * LINES * PER_LINE;
const N_STEPS = 50;
const WG = 256;
// Run trains until the loss improves by less than MIN_GAIN over PATIENCE iterations:
// large swirls and fine tessellations need several times the script's 2,000.
const PATIENCE = 1000;
const MIN_GAIN = 0.02;       // relative improvement over the best loss that resets the patience
const MAX_ITERS = 20000;     // iterations per Run at most (~35 s)
const ITERS_PER_FRAME = 10;  // ~600 iterations/s at 60 fps
const READBACK_EVERY = 4;    // frames between metric readbacks while running
const INIT_STD = 1e-3;       // theta ~ N(0, INIT_STD^2), as in the script
const BACKGROUND = [0.051, 0.059, 0.086, 1];

// The script's line colors, lightened for the dark panels; widths in CSS px.
const STYLES = {
  target: { rgb: '#38bdf8', alpha: 1, width: 1.4 },
  warped: { rgb: '#2dd4bf', alpha: 1, width: 1.4 },
  source: { rgb: '#9ca3af', alpha: 0.5, width: 2.8 },
  recovered: { rgb: '#f472b6', alpha: 0.9, width: 1.2 },
};
// Layers drawn in each panel, bottom to top.
const PANELS = {
  target: ['target'],
  warped: ['warped'],
  inverse: ['source', 'recovered'],
};

// dL/dphi for L = mean over all 2N coordinates of (phi(x) - y)^2.
const GRAD_SHADER = /* wgsl */ `
struct GradParams {
  n: u32,
  scale: f32,
  _p0: u32,
  _p1: u32,
};
@group(0) @binding(0) var<uniform> P: GradParams;
@group(0) @binding(1) var<storage, read> warped: array<f32>;
@group(0) @binding(2) var<storage, read> goal: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradOut: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  gradOut[i] = (warped[i] - goal[i]) * P.scale;
}
`;

// dst = [f0 * src, f1 * src]: the display engine's two batch entries.
const SCALE_SHADER = /* wgsl */ `
struct ScaleParams {
  n: u32,
  d: u32,
  f0: f32,
  f1: f32,
};
@group(0) @binding(0) var<uniform> P: ScaleParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  dst[i] = src[i % P.d] * select(P.f1, P.f0, i < P.d);
}
`;

// Each polyline is one instance, drawn as a triangle strip read straight from
// a [2, nP] point buffer: vertex 2k + s sits at point k, pushed to side s along
// the miter. Points are in [0,1]^2 with y up, as in the script's plots.
const LINE_SHADER = /* wgsl */ `
struct LineParams {
  color: vec4<f32>,  // premultiplied
  canvas: vec2<f32>,
  halfWidth: f32,    // device px
  perLine: u32,
  nP: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};
@group(0) @binding(0) var<uniform> L: LineParams;
@group(0) @binding(1) var<storage, read> pts: array<f32>;

struct LOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) across: f32,
};

fn px(i: u32) -> vec2<f32> {
  return vec2<f32>(pts[i], pts[i + L.nP]) * L.canvas;
}

fn unit(v: vec2<f32>) -> vec2<f32> {
  let n = length(v);
  return select(vec2<f32>(1.0, 0.0), v / n, n > 1e-6);
}

@vertex
fn vs_line(@builtin(vertex_index) vi: u32, @builtin(instance_index) line: u32) -> LOut {
  let k = vi / 2u;
  let side = select(-1.0, 1.0, (vi & 1u) == 1u);
  let i = line * L.perLine + k;
  let p = px(i);
  var a = p - px(select(i, i - 1u, k > 0u));
  var b = px(select(i, i + 1u, k + 1u < L.perLine)) - p;
  if (k == 0u) { a = b; }
  if (k + 1u == L.perLine) { b = a; }
  let n0 = unit(a).yx * vec2<f32>(-1.0, 1.0);
  let n1 = unit(b).yx * vec2<f32>(-1.0, 1.0);
  let m = unit(n0 + n1);
  let ext = L.halfWidth + 1.0;  // one pixel of anti-aliasing
  let q = p + side * ext / max(dot(m, n1), 0.5) * m;
  var o: LOut;
  o.pos = vec4<f32>(q / L.canvas * 2.0 - 1.0, 0.0, 1.0);
  o.across = side * ext;
  return o;
}

@fragment
fn fs_line(in: LOut) -> @location(0) vec4<f32> {
  return L.color * clamp(L.halfWidth + 0.5 - abs(in.across), 0.0, 1.0);
}
`;

const key = (layer) => `<span class="key" style="--key: ${STYLES[layer].rgb}"></span>`;

const TEMPLATE = `
<div class="demo demo-register">
  <div class="panels three">
    <figure class="panel">
      <div class="demo-stage"><canvas class="demo-canvas" data-c="target"></canvas></div>
      <figcaption>${key('target')}Target <i>y</i></figcaption>
    </figure>
    <figure class="panel">
      <div class="demo-stage"><canvas class="demo-canvas" data-c="warped"></canvas></div>
      <figcaption>${key('warped')}Learned warp φ<sub>tθ</sub>(<i>x</i>)</figcaption>
    </figure>
    <figure class="panel wide-only">
      <div class="demo-stage"><canvas class="demo-canvas" data-c="inverse"></canvas></div>
      <figcaption>${key('recovered')}Inverse φ<sub>−tθ</sub>(φ<sub>tθ</sub>(<i>x</i>)) <span class="nowrap">on ${key('source')}<i>x</i></span></figcaption>
    </figure>
  </div>
  <div class="register-bottom">
    <div>
      <canvas class="loss-plot" data-c="loss" aria-label="Loss curve"></canvas>
      <div class="metrics">
        <div><strong data-o="iter">0</strong><span>iterations</span></div>
        <div><strong data-o="fit">–</strong><span>mean fit error <span class="formula">|φ(<i>x</i>) − <i>y</i>|</span></span></div>
        <div><strong data-o="cycle">–</strong><span>mean inverse error <span class="formula">|φ<sup>−1</sup>(φ(<i>x</i>)) − <i>x</i>|</span></span></div>
        <div><strong data-o="d">–</strong><span>parameters d</span></div>
      </div>
    </div>
    <div class="demo-panel">
      <div class="demo-controls">
        <label class="ctl">
          <span class="ctl-name">Tessellation</span>
          <input type="range" min="2" max="30" step="1" value="24" data-k="tess">
          <output data-o="tess"></output>
        </label>
        <label class="ctl">
          <span class="ctl-name">Swirl angle</span>
          <input type="range" min="-2.5" max="2.5" step="0.05" value="0.8" data-k="swirl">
          <output data-o="swirl"></output>
        </label>
        <label class="ctl">
          <span class="ctl-name">Learning rate</span>
          <input type="range" min="-2.5" max="0.5" step="0.1" value="0" data-k="lr">
          <output data-o="lr"></output>
        </label>
        <label class="ctl">
          <span class="ctl-name">Time t</span>
          <input type="range" min="0" max="1" step="0.001" value="1" data-k="t">
          <output data-o="t"></output>
        </label>
      </div>
      <div class="demo-buttons">
        <button type="button" data-k="run" aria-pressed="false">Run</button>
        <button type="button" data-k="play" aria-pressed="false">Play flow</button>
        <button type="button" data-k="restart">Reset</button>
      </div>
      <p class="demo-stats" data-o="status"></p>
    </div>
  </div>
</div>`;

/** The script's build_line_grid: LINES horizontal, then LINES vertical lines, as [2, N]. */
function lineGrid(lo = 0.05, hi = 0.95) {
  const at = (k, m) => lo + ((hi - lo) * k) / (m - 1);
  const pts = new Float32Array(2 * N);
  let i = 0;
  for (const vertical of [false, true]) {
    for (let l = 0; l < LINES; l++) {
      for (let k = 0; k < PER_LINE; k++, i++) {
        const along = at(k, PER_LINE);
        const across = at(l, LINES);
        pts[i] = vertical ? across : along;
        pts[N + i] = vertical ? along : across;
      }
    }
  }
  return pts;
}

/** The script's make_target: rotate about the centre by `angle` * exp(-8 r^2). */
function swirl(points, angle) {
  const out = new Float32Array(2 * N);
  for (let i = 0; i < N; i++) {
    const x = points[i] - 0.5;
    const y = points[N + i] - 0.5;
    const a = angle * Math.exp(-8 * (x * x + y * y));
    const c = Math.cos(a);
    const s = Math.sin(a);
    out[i] = Math.min(Math.max(0.5 + c * x - s * y, 0), 1);
    out[N + i] = Math.min(Math.max(0.5 + s * x + c * y, 0), 1);
  }
  return out;
}

function hexRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255].map((c) => c / 255);
}

export async function mountGridFit(root) {
  const ui = el(TEMPLATE);
  root.replaceChildren(ui);
  const q = (k) => ui.querySelector(`[data-k="${k}"]`);
  const out = (k) => ui.querySelector(`[data-o="${k}"]`);
  const canvases = Object.fromEntries([...ui.querySelectorAll('canvas[data-c]')].map((c) => [c.dataset.c, c]));

  let device;
  try {
    device = await getDevice();
  } catch (err) {
    showFallback(ui.querySelector('.panels'), err);
    return;
  }

  const U = GPUBufferUsage;
  const source = lineGrid();
  const sourceBuf = device.createBuffer({ size: source.byteLength, usage: U.STORAGE | U.COPY_DST });
  device.queue.writeBuffer(sourceBuf, 0, source);
  const targetBuf = device.createBuffer({ size: source.byteLength, usage: U.STORAGE | U.COPY_DST });
  const gradParams = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
  const adamParamBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
  const scaleParams = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
  const stepBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_DST });
  const layerParams = Object.fromEntries(Object.keys(STYLES).map((layer) =>
    [layer, device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST })]));
  {
    const f = new Float32Array(4);
    new Uint32Array(f.buffer)[0] = 2 * N;
    f[1] = 1 / N;  // d/dphi of sum r^2 / (2N)
    device.queue.writeBuffer(gradParams, 0, f);
  }

  const compute = (code, label) => device.createComputePipelineAsync({
    layout: 'auto', compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' },
  });
  const format = navigator.gpu.getPreferredCanvasFormat();
  const over = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' };
  const render = (code, label, vertex, fragment, topology) => {
    const module = device.createShaderModule({ code, label });
    return device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module, entryPoint: vertex },
      fragment: { module, entryPoint: fragment, targets: [{ format, blend: { color: over, alpha: over } }] },
      primitive: { topology },
    });
  };
  const [gradPipeline, scalePipeline, adam, linePipeline] = await Promise.all([
    compute(GRAD_SHADER, 'gridfit.grad'),
    compute(SCALE_SHADER, 'gridfit.scale'),
    createAdam(device),
    render(LINE_SHADER, 'gridfit.lines', 'vs_line', 'fs_line', 'triangle-strip'),
  ]);
  const entries = (buffers) => buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));

  const state = {
    n: 24,
    lr: 1,
    angle: 0.8,
    goal: null,           // target points y (CPU copy, for the metrics)
    running: false,
    stopAt: MAX_ITERS,
    best: Infinity,       // best loss of this fit, and the iteration it was reached at
    bestAt: 0,
    t: 1,
    playing: false,
    playClock: 0,
    displayDirty: true,   // the display engine's theta is stale
    redraw: true,         // the panels need a render
    needMetrics: true,    // read back the fit and inverse errors at the next t = 1 frame
    visible: false,
    iteration: 0,
    history: [],
    engine: null,
    display: null,        // batch [t theta, -t theta], for the panels and the metrics
    sets: null,
    bg: null,
    m1: null,
    m2: null,
    rateClock: { t: performance.now(), it: 0, value: 0 },
    readbackBusy: false,
    frame: 0,
    gen: 0,
    run: 0,
  };

  const contexts = {};
  for (const panel of Object.keys(PANELS)) {
    contexts[panel] = configureCanvas(device, canvases[panel]).context;
    autoResize(canvases[panel], () => { state.redraw = true; }, 1024);
  }

  // -------------------------------------------------------------------------
  // Engines and bind groups (rebuilt when the tessellation changes)
  // -------------------------------------------------------------------------

  async function build() {
    const gen = ++state.gen;
    const tess = [state.n, state.n];
    const [engine, display] = await Promise.all([
      CPAB2D.create(device, { tess, nSteps: N_STEPS }),
      CPAB2D.create(device, { tess, nSteps: N_STEPS, batch: 2 }),
    ]);
    if (gen !== state.gen) {
      engine.destroy();
      display.destroy();
      return;
    }
    const fit = engine.createPointSet({ nP: N, points: sourceBuf });
    const fwd = display.createPointSet({ nP: N, batchCount: 1, points: sourceBuf });
    const inv = display.createPointSet({ nP: N, batchOffset: 1, pointsBatched: true, points: fwd.out });
    const m1 = device.createBuffer({ size: engine.d * 4, usage: U.STORAGE | U.COPY_DST });
    const m2 = device.createBuffer({ size: engine.d * 4, usage: U.STORAGE | U.COPY_DST });
    const lineBG = (layer, buffer) => device.createBindGroup({
      layout: linePipeline.getBindGroupLayout(0), entries: entries([layerParams[layer], buffer]),
    });
    const bg = {
      grad: device.createBindGroup({
        layout: gradPipeline.getBindGroupLayout(0), entries: entries([gradParams, fit.out, targetBuf, fit.gradOut]),
      }),
      adam: device.createBindGroup({
        layout: adam.layout, entries: entries([adamParamBuf, engine.buffers.theta, engine.buffers.gradTheta, m1, m2, stepBuf]),
      }),
      scale: device.createBindGroup({
        layout: scalePipeline.getBindGroupLayout(0), entries: entries([scaleParams, engine.buffers.theta, display.buffers.theta]),
      }),
      layers: {
        target: lineBG('target', targetBuf),
        warped: lineBG('warped', fwd.out),
        source: lineBG('source', sourceBuf),
        recovered: lineBG('recovered', inv.out),
      },
    };
    if (state.engine) {
      state.engine.destroyPointSet(state.sets.fit);
      state.display.destroyPointSet(state.sets.fwd);
      state.display.destroyPointSet(state.sets.inv);
      state.engine.destroy();
      state.display.destroy();
      state.m1.destroy();
      state.m2.destroy();
    }
    Object.assign(state, { engine, display, sets: { fit, fwd, inv }, bg, m1, m2 });
    out('d').textContent = engine.d.toLocaleString();
    restart();
  }

  function restart() {
    const e = state.engine;
    if (!e) return;
    e.writeTheta(gaussian(e.d, INIT_STD, mulberry32(42 + state.run)));
    device.queue.writeBuffer(state.m1, 0, new Float32Array(e.d));
    device.queue.writeBuffer(state.m2, 0, new Float32Array(e.d));
    device.queue.writeBuffer(stepBuf, 0, new Uint32Array(4));
    state.iteration = 0;
    state.stopAt = MAX_ITERS;
    state.best = Infinity;
    state.bestAt = 0;
    state.history = [];
    state.run++;
    state.rateClock = { t: performance.now(), it: 0, value: 0 };
    state.needMetrics = true;
    setRunning(false);
    setPlaying(false);
    setT(1);
    out('iter').textContent = '0';
    out('fit').textContent = '–';
    out('cycle').textContent = '–';
    out('status').textContent = 'Press Run to fit the swirl. It runs until the fit stops improving.';
    lockSettings(false);
    drawLoss();
  }

  // -------------------------------------------------------------------------
  // Controls
  // -------------------------------------------------------------------------

  function setTess(n) {
    state.n = n;
    q('tess').value = String(n);
    out('tess').textContent = `${n}×${n}`;
  }
  function setAngle(angle) {
    state.angle = angle;
    q('swirl').value = String(angle);
    out('swirl').textContent = `${angle.toFixed(2)} rad`;
    state.goal = swirl(source, angle);
    device.queue.writeBuffer(targetBuf, 0, state.goal);
    state.needMetrics = true;
    state.redraw = true;
  }
  function setLr(log10) {
    state.lr = Math.pow(10, log10);
    q('lr').value = String(log10);
    out('lr').textContent = state.lr.toPrecision(2);
  }
  /** reason, when stopping: 'paused' (by the user), 'converged' or 'limit' (MAX_ITERS). */
  function setRunning(on, reason = 'paused') {
    if (state.running && !on) {
      const it = state.iteration.toLocaleString();
      const head = {
        paused: `Paused at ${it} iterations.`,
        converged: `Converged after ${it} iterations.`,
        limit: `Stopped at ${it} iterations, the limit for one run.`,
      }[reason];
      out('status').textContent = `${head} Drag Time t or press Play flow to watch the fitted transformation; `
        + `${reason === 'converged' ? '' : 'Run continues; '}Reset unlocks the settings.`;
    }
    state.running = on;
    q('run').setAttribute('aria-pressed', String(on));
    q('run').textContent = on ? 'Pause' : 'Run';
  }
  function setPlaying(on) {
    state.playing = on;
    q('play').setAttribute('aria-pressed', String(on));
    q('play').textContent = on ? 'Stop' : 'Play flow';
  }
  function setT(t) {
    state.t = t;
    q('t').value = String(t);
    out('t').textContent = t.toFixed(2);
    state.displayDirty = true;
    state.redraw = true;
  }
  /** The training settings are fixed from the first Run until Reset. */
  function lockSettings(locked) {
    for (const k of ['tess', 'swirl', 'lr']) q(k).disabled = locked;
  }
  function runningStatus() {
    const rate = state.rateClock.value ? ` · ${Math.round(state.rateClock.value).toLocaleString()} iterations/s` : '';
    out('status').textContent = `Optimizing until the fit stops improving…${rate}`;
  }

  q('tess').addEventListener('input', (ev) => { out('tess').textContent = `${ev.target.value}×${ev.target.value}`; });
  q('tess').addEventListener('change', (ev) => { setTess(Number(ev.target.value)); build(); });
  q('swirl').addEventListener('input', (ev) => {
    setPlaying(false);
    setT(1);
    setAngle(Number(ev.target.value));
  });
  q('lr').addEventListener('input', (ev) => setLr(Number(ev.target.value)));
  q('run').addEventListener('click', () => {
    if (!state.running) {
      // Train until the loss plateaus (see readback), showing the fit at t = 1.
      state.stopAt = state.iteration + MAX_ITERS;
      state.bestAt = state.iteration;
      lockSettings(true);
      setPlaying(false);
      setT(1);
      state.rateClock = { t: performance.now(), it: state.iteration, value: 0 };
      runningStatus();
    }
    setRunning(!state.running);
  });
  q('play').addEventListener('click', () => {
    setRunning(false);
    setPlaying(!state.playing);
    state.playClock = Math.acos(Math.min(1, Math.max(-1, 1 - 2 * state.t)));
  });
  q('t').addEventListener('input', (ev) => {
    setRunning(false);
    setPlaying(false);
    setT(Number(ev.target.value));
  });
  q('restart').addEventListener('click', () => restart());

  // Only spend GPU time while the demo is on screen.
  new IntersectionObserver((list) => {
    state.visible = list.some((e) => e.isIntersecting);
    if (state.visible) state.redraw = true;
  }, { threshold: 0.05 }).observe(ui);

  const lossCanvas = canvases.loss;
  autoResize(lossCanvas, () => drawLoss(), 1600);
  function drawLoss() {
    drawLogCurve(lossCanvas, state.history);
  }

  // -------------------------------------------------------------------------
  // Frame loop
  // -------------------------------------------------------------------------

  function writeParams() {
    const e = state.engine;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    device.queue.writeBuffer(adamParamBuf, 0, adamParams({ n: e.d, lr: state.lr }));

    const sf = new Float32Array(4);
    const su = new Uint32Array(sf.buffer);
    su[0] = 2 * e.d;
    su[1] = e.d;
    sf[2] = state.t;
    sf[3] = -state.t;
    device.queue.writeBuffer(scaleParams, 0, sf);

    for (const [panel, layers] of Object.entries(PANELS)) {
      const c = canvases[panel];
      for (const layer of layers) {
        const s = STYLES[layer];
        const f = new Float32Array(12);
        const u = new Uint32Array(f.buffer);
        f.set([...hexRgb(s.rgb).map((v) => v * s.alpha), s.alpha, c.width, c.height, 0.5 * s.width * dpr]);
        u[7] = PER_LINE;
        u[8] = N;
        device.queue.writeBuffer(layerParams[layer], 0, f);
      }
    }
  }

  function encodeIteration(enc) {
    const e = state.engine;
    const { fit } = state.sets;
    e.encodeAffine(enc);
    e.encodeExpm(enc);
    e.encodeForward(enc, fit);
    const pass = enc.beginComputePass();
    pass.setPipeline(gradPipeline);
    pass.setBindGroup(0, state.bg.grad);
    pass.dispatchWorkgroups(Math.ceil((2 * N) / WG));
    pass.end();
    e.encodeBackward(enc, fit);
    e.encodeProject(enc);
    adam.encode(enc, state.bg.adam, e.d);
  }

  /** phi_{t theta}(x) and phi_{-t theta}(phi_{t theta}(x)) from the display engine. */
  function encodeDisplay(enc) {
    const pass = enc.beginComputePass();
    pass.setPipeline(scalePipeline);
    pass.setBindGroup(0, state.bg.scale);
    pass.dispatchWorkgroups(Math.ceil((2 * state.engine.d) / WG));
    pass.end();
    const d = state.display;
    d.encodeAffine(enc);
    d.encodeExpm(enc);
    d.encodeForward(enc, state.sets.fwd);
    d.encodeForward(enc, state.sets.inv);
  }

  function renderPanels(enc) {
    for (const [panel, layers] of Object.entries(PANELS)) {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: contexts[panel].getCurrentTexture().createView(),
          loadOp: 'clear', storeOp: 'store', clearValue: BACKGROUND,
        }],
      });
      pass.setPipeline(linePipeline);
      for (const layer of layers) {
        pass.setBindGroup(0, state.bg.layers[layer]);
        pass.draw(2 * PER_LINE, 2 * LINES);
      }
      pass.end();
    }
  }

  /** Fit and inverse errors (mean absolute coordinate error, as the script prints) and the MSE loss;
   *  while training, also the stopping rule (PATIENCE). */
  async function readback(staging) {
    state.readbackBusy = true;
    const { iteration, run, goal } = state;
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange());
    let sq = 0;
    let fit = 0;
    let cycle = 0;
    for (let i = 0; i < 2 * N; i++) {
      const r = data[i] - goal[i];
      sq += r * r;
      fit += Math.abs(r);
      cycle += Math.abs(data[2 * N + i] - source[i]);
    }
    staging.destroy();
    state.readbackBusy = false;
    if (run !== state.run) return;
    out('fit').textContent = (fit / (2 * N)).toExponential(2);
    out('cycle').textContent = (cycle / (2 * N)).toExponential(2);
    const h = state.history;
    const loss = sq / (2 * N);
    if (h.length && h[h.length - 1][0] === iteration) h[h.length - 1][1] = loss;
    else h.push([iteration, loss]);
    if (h.length > 2000) state.history = h.filter((_, i) => i % 2 === 0);
    drawLoss();
    if (!state.running) return;
    if (loss < state.best * (1 - MIN_GAIN)) {
      state.best = loss;
      state.bestAt = iteration;
    } else if (iteration - state.bestAt >= PATIENCE) {
      setRunning(false, 'converged');
    }
  }

  let prevNow = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - prevNow) / 1000);
    prevNow = now;
    if (!state.engine || !state.visible) return;
    state.frame++;
    if (state.playing) {
      state.playClock += dt * 1.4;
      setT(0.5 - 0.5 * Math.cos(state.playClock));
    }
    const iterated = state.running;
    const wantMetrics = state.needMetrics && state.t === 1;
    if (!iterated && !state.displayDirty && !state.redraw && !wantMetrics) return;

    writeParams();
    const enc = device.createCommandEncoder();
    let reachedStop = false;
    if (iterated) {
      const k = Math.min(ITERS_PER_FRAME, state.stopAt - state.iteration);
      for (let i = 0; i < k; i++) encodeIteration(enc);
      state.iteration += k;
      reachedStop = state.iteration >= state.stopAt;
    }
    if (iterated || state.displayDirty) encodeDisplay(enc);
    state.displayDirty = false;
    renderPanels(enc);
    state.redraw = false;
    let staging = null;
    const periodic = iterated && state.frame % READBACK_EVERY === 0;
    if (state.t === 1 && (reachedStop || (!state.readbackBusy && (wantMetrics || periodic)))) {
      const bytes = 2 * N * 4;
      staging = device.createBuffer({ size: 2 * bytes, usage: U.COPY_DST | U.MAP_READ });
      enc.copyBufferToBuffer(state.sets.fwd.out, 0, staging, 0, bytes);
      enc.copyBufferToBuffer(state.sets.inv.out, 0, staging, bytes, bytes);
      state.needMetrics = false;
    }
    device.queue.submit([enc.finish()]);
    if (staging) readback(staging);

    if (iterated) {
      out('iter').textContent = state.iteration.toLocaleString();
      const rc = state.rateClock;
      if (now - rc.t > 500) {
        rc.value = ((state.iteration - rc.it) * 1000) / (now - rc.t);
        rc.t = now;
        rc.it = state.iteration;
        runningStatus();
      }
    }
    if (reachedStop) setRunning(false, 'limit');
  }

  setTess(state.n);
  setLr(Math.log10(state.lr));
  setAngle(state.angle);
  await build();
  requestAnimationFrame(frame);
  return { state };
}
