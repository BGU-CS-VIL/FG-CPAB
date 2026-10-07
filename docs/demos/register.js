/**
 * Option 2 — live registration with FG-CPAB gradients.
 *
 * Source and target are RGB densities. The source is a Gaussian mixture,
 * evaluated analytically (value and gradient, in float32); the target is given
 * by its values at the GRID^2 grid points x_n. The loss is pointwise:
 *   L = mean_n |rho_s(phi_theta(x_n)) - rho_t(x_n)|^2.
 * Every iteration is recorded on the GPU: theta -> A -> expm -> forward (per
 * composed stage) -> pointwise loss -> adjoint moment sweep (Eq. 11, stages in
 * reverse) -> sparse projection (Eq. 12) -> Adam.
 *
 * The optimization starts on Run and stops once the loss plateaus (PATIENCE). The
 * warped panel is drawn at a time t in [0, 1] of the fitted flow, from a second
 * engine whose theta is the fitted one scaled per stage (see stageFactors).
 */
import { CPAB2D } from '../src/cpab2d.js';
import { thetaFromVelocities } from '../src/tessellation2d.js';
import {
  getDevice, showFallback, configureCanvas, autoResize, el, makeCanvas, mulberry32, randomVelocities, gridPoints,
  gaussian, createAdam, adamParams, drawLogCurve, WGSL_FULLSCREEN, WGSL_COLORMAPS,
} from './common.js';

const GRID = 128;           // the loss is evaluated at GRID^2 points
const N_STEPS = 50;
const WG = 256;
const N_PARTIALS = Math.ceil((GRID * GRID) / WG);
const MAX_COMPONENTS = 96;  // Gaussians in the source density
const PAINT = 256;          // resolution of the canvases that targets are sampled from
const READBACK_EVERY = 4;   // frames between loss readbacks
// Run trains until the loss improves by less than MIN_GAIN over PATIENCE iterations.
const PATIENCE = 1000;
const MIN_GAIN = 0.02;      // relative improvement over the best loss that resets the patience
const MAX_ITERS = 2000;     // iterations per Run at most (~11 s); Run again adds another batch
const ITERS_PER_FRAME = 3;  // ~180 iterations/s at 60 fps
const FLAG_DIFF = 1;
const ORANGE = [0.95, 0.65, 0.25];
// theta ~ N(0, INIT_STD^2) at restart, as in optimized_circle_to_two_circles_2d.py. With
// a zero start every composed stage gets the same gradient and the stages stay equal,
// so the chain collapses to one stationary flow (phi_theta^m = phi_{m theta}).
const INIT_STD = 1e-3;
const DEFAULT_LR = Math.log10(0.5);  // the slider is log10(lr)

const PRESETS = {
  recover: {
    tess: 12, stages: 1, lr: DEFAULT_LR,
    label: 'Recover a hidden warp',
    note: 'A density made of random Gaussian blobs, and a target '
      + 'made by warping it with a hidden CPAB transformation on the same tessellation. Adam on the '
      + 'factorized gradients recovers it.',
  },
  shapes: {
    tess: 12, stages: 1, lr: DEFAULT_LR,
    label: 'Disk → heart',
    note: 'The disk is deformed into the heart. The map stays a diffeomorphism throughout, so the shape bends '
      + 'and stretches but never tears or folds.',
  },
};

// Gaussian-mixture density: component k contributes w_k * exp(-a_k |p - mu_k|^2),
// where w_k is its RGB weight and a_k = 1 / (2 sigma_k^2).
const WGSL_DENSITY = /* wgsl */ `
struct Component {
  mu_a: vec4<f32>,   // mu.x, mu.y, a, unused
  w: vec4<f32>,      // RGB weight, unused
};

struct Density {
  v: vec3<f32>,
  dx: vec3<f32>,
  dy: vec3<f32>,
};

fn density_value(p: vec2<f32>, n: u32) -> vec3<f32> {
  var v = vec3<f32>(0.0);
  for (var k = 0u; k < n; k++) {
    let c = comps[k];
    let o = p - c.mu_a.xy;
    v = v + c.w.xyz * exp(-dot(o, o) * c.mu_a.z);
  }
  return v;
}

fn density_grad(p: vec2<f32>, n: u32) -> Density {
  var d: Density;
  d.v = vec3<f32>(0.0);
  d.dx = vec3<f32>(0.0);
  d.dy = vec3<f32>(0.0);
  for (var k = 0u; k < n; k++) {
    let c = comps[k];
    let o = p - c.mu_a.xy;
    let g = c.w.xyz * exp(-dot(o, o) * c.mu_a.z);
    let s = -2.0 * c.mu_a.z;
    d.v = d.v + g;
    d.dx = d.dx + g * (s * o.x);
    d.dy = d.dy + g * (s * o.y);
  }
  return d;
}
`;

const LOSS_SHADER = /* wgsl */ `
struct LossParams {
  nP: u32,
  nComp: u32,
  invN: f32,
  _p0: f32,
};
@group(0) @binding(0) var<uniform> P: LossParams;
@group(0) @binding(1) var<storage, read> warped: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;
@group(0) @binding(4) var<storage, read> comps: array<Component>;
@group(0) @binding(5) var<storage, read> tgtVals: array<vec4<f32>>;
` + WGSL_DENSITY + /* wgsl */ `
var<workgroup> red: array<f32, ${WG}>;

// Pointwise loss at x_n: |rho_s(phi(x_n)) - rho_t(x_n)|^2 / N, and dL/dphi(x_n)
// from the analytic gradient of the source density.
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let n = gid.x;
  var l = 0.0;
  if (n < P.nP) {
    let p = vec2<f32>(warped[n], warped[n + P.nP]);
    let d = density_grad(p, P.nComp);
    let r = d.v - tgtVals[n].xyz;
    l = dot(r, r) * P.invN;
    gradOut[n] = 2.0 * dot(r, d.dx) * P.invN;
    gradOut[n + P.nP] = 2.0 * dot(r, d.dy) * P.invN;
  }
  red[lid] = l;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k = k >> 1u) {
    if (lid < k) { red[lid] = red[lid] + red[lid + k]; }
    workgroupBarrier();
  }
  if (lid == 0u) { partial[wid.x] = red[0]; }
}
`;

// dst = src with stage k's parameters scaled by f[k]: the flow at time t of the
// composed chain (see stageFactors).
const SCALE_SHADER = /* wgsl */ `
struct ScaleParams {
  n: u32,
  d: u32,
  _p0: u32,
  _p1: u32,
  f: array<vec4<f32>, 3>,
};
@group(0) @binding(0) var<uniform> P: ScaleParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let k = i / P.d;
  dst[i] = src[i] * P.f[k / 4u][k % 4u];
}
`;

const PANEL_SHADER = WGSL_FULLSCREEN + WGSL_COLORMAPS + /* wgsl */ `
struct PanelParams {
  mapSize: u32,
  flags: u32,
  nComp: u32,
  _p0: u32,
};
@group(0) @binding(0) var<uniform> U: PanelParams;
@group(0) @binding(1) var<storage, read> mapBuf: array<f32>;
@group(0) @binding(2) var<storage, read> comps: array<Component>;
@group(0) @binding(3) var<storage, read> tgtVals: array<vec4<f32>>;
` + WGSL_DENSITY + /* wgsl */ `
const BACKGROUND = vec3<f32>(0.051, 0.059, 0.086);

fn shade(v: vec3<f32>) -> vec3<f32> { return min(BACKGROUND + v, vec3<f32>(1.0)); }

// Bilinear interpolation on the GRID^2 point grid (x fastest).
struct Cell { i: u32, f: vec2<f32> };
fn grid_cell(uv: vec2<f32>) -> Cell {
  let n = f32(U.mapSize - 1u);
  let q = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * n;
  let q0 = min(floor(q), vec2<f32>(n - 1.0));
  return Cell(u32(q0.x) + u32(q0.y) * U.mapSize, q - q0);
}

fn map_point(i: u32) -> vec2<f32> {
  return vec2<f32>(mapBuf[i], mapBuf[i + U.mapSize * U.mapSize]);
}

fn map_at(uv: vec2<f32>) -> vec2<f32> {
  let c = grid_cell(uv);
  let w = U.mapSize;
  return mix(mix(map_point(c.i), map_point(c.i + 1u), c.f.x),
             mix(map_point(c.i + w), map_point(c.i + w + 1u), c.f.x), c.f.y);
}

fn target_at(uv: vec2<f32>) -> vec3<f32> {
  let c = grid_cell(uv);
  let w = U.mapSize;
  return mix(mix(tgtVals[c.i].xyz, tgtVals[c.i + 1u].xyz, c.f.x),
             mix(tgtVals[c.i + w].xyz, tgtVals[c.i + w + 1u].xyz, c.f.x), c.f.y);
}

@fragment
fn fs_source(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(shade(density_value(in.uv, U.nComp)), 1.0);
}

@fragment
fn fs_warped(in: VSOut) -> @location(0) vec4<f32> {
  let v = density_value(map_at(in.uv), U.nComp);
  var col = shade(v);
  if ((U.flags & ${FLAG_DIFF}u) != 0u) {
    col = ramp(pow(length(v - target_at(in.uv)) / sqrt(3.0), 0.6));
  }
  return vec4<f32>(col, 1.0);
}

@fragment
fn fs_target(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(shade(target_at(in.uv)), 1.0);
}
`;

const TEMPLATE = `
<div class="demo demo-register">
  <div class="panels three">
    <figure class="panel wide-only">
      <div class="demo-stage"><canvas class="demo-canvas" data-c="source"></canvas></div>
      <figcaption>Source</figcaption>
    </figure>
    <figure class="panel">
      <div class="demo-stage"><canvas class="demo-canvas" data-c="warped"></canvas></div>
      <figcaption>Warped source at time t</figcaption>
    </figure>
    <figure class="panel">
      <div class="demo-stage"><canvas class="demo-canvas" data-c="target"></canvas></div>
      <figcaption>Target</figcaption>
    </figure>
  </div>
  <div class="register-bottom">
    <div>
      <canvas class="loss-plot" data-c="loss" aria-label="Loss curve"></canvas>
      <div class="metrics">
        <div><strong data-o="iter">0</strong><span>iterations</span></div>
        <div><strong data-o="rate">–</strong><span>iterations / s</span></div>
        <div><strong data-o="loss">–</strong><span>pointwise loss (MSE)</span></div>
        <div><strong data-o="d">–</strong><span>parameters d</span></div>
      </div>
    </div>
    <div class="demo-panel">
      <div class="demo-controls">
        <label class="ctl">
          <span class="ctl-name">Task</span>
          <select data-k="preset">${Object.entries(PRESETS).map(([k, p]) => `<option value="${k}">${p.label}</option>`).join('')}</select>
        </label>
        <label class="ctl">
          <span class="ctl-name">Tessellation</span>
          <input type="range" min="2" max="30" step="1" value="12" data-k="tess">
          <output data-o="tess"></output>
        </label>
        <label class="ctl">
          <span class="ctl-name">Learning rate</span>
          <input type="range" min="-2.5" max="0.5" step="0.1" value="-0.3" data-k="lr">
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
        <label class="toggle"><input type="checkbox" data-k="diff"> Show error</label>
      </div>
      <p class="demo-stats" data-o="status"></p>
      <p class="demo-stats" data-o="note"></p>
    </div>
  </div>
</div>`;

// ---------------------------------------------------------------------------
// Densities
// ---------------------------------------------------------------------------

/** Pack a mixture [{x, y, sigma, w}] (w = weight per channel) into the GPU layout (Component). */
function mixtureData(mix) {
  if (mix.length > MAX_COMPONENTS) throw new Error(`at most ${MAX_COMPONENTS} components`);
  const out = new Float32Array(MAX_COMPONENTS * 8);
  mix.forEach((c, k) => out.set([c.x, c.y, 1 / (2 * c.sigma * c.sigma), 0, ...c.w, 0], 8 * k));
  return out;
}

function evalMixture(mix, x, y) {
  const v = [0, 0, 0];
  for (const c of mix) {
    const g = Math.exp(-((x - c.x) ** 2 + (y - c.y) ** 2) / (2 * c.sigma * c.sigma));
    for (let k = 0; k < 3; k++) v[k] += c.w[k] * g;
  }
  return v;
}

/** Target values rho_t(x_n) at the grid points, as vec4 rows. */
function gridValues(fn) {
  const out = new Float32Array(GRID * GRID * 4);
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      out.set(fn(j / (GRID - 1), i / (GRID - 1)), 4 * (i * GRID + j));
    }
  }
  return out;
}

/** Target values sampled (bilinearly) from a canvas drawn on black. */
function canvasValues(canvas) {
  const px = canvas.getContext('2d').getImageData(0, 0, PAINT, PAINT).data;
  return gridValues((x, y) => {
    const fx = Math.min(Math.max(x * PAINT - 0.5, 0), PAINT - 1.001);
    const fy = Math.min(Math.max(y * PAINT - 0.5, 0), PAINT - 1.001);
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const ax = fx - x0;
    const ay = fy - y0;
    const at = (xx, yy, ch) => px[4 * (yy * PAINT + xx) + ch] / 255;
    return [0, 1, 2].map((ch) => (at(x0, y0, ch) * (1 - ax) + at(x0 + 1, y0, ch) * ax) * (1 - ay)
      + (at(x0, y0 + 1, ch) * (1 - ax) + at(x0 + 1, y0 + 1, ch) * ax) * ay);
  });
}

function hsl(h, s, l) {
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

/** Nine random Gaussian blobs, sigma in [0.03, 0.12], as in the synthetic benchmark (App. C.1). */
function randomBlobs(seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: 9 }, () => ({
    x: 0.15 + 0.7 * rng(),
    y: 0.15 + 0.7 * rng(),
    sigma: 0.03 + 0.09 * rng(),
    w: hsl(360 * rng(), 0.75, 0.55).map((v) => 0.8 * v),
  }));
}

/** A soft round disk: small Gaussians on concentric rings, with a flat interior. */
function diskMixture(radius = 0.2, spacing = 0.04) {
  const sigma = 0.6 * spacing;
  // About one component per spacing^2 of area, so the interior value is
  // amp * 2 pi sigma^2 / spacing^2; make it 1.
  const w = ORANGE.map((v) => (v * spacing * spacing) / (2 * Math.PI * sigma * sigma));
  const out = [{ x: 0.5, y: 0.5, sigma, w }];
  for (let r = spacing; r <= radius + 1e-9; r += spacing) {
    const n = Math.round((2 * Math.PI * r) / spacing);
    for (let k = 0; k < n; k++) {
      const a = (2 * Math.PI * k) / n + (r / spacing) * 0.5;
      out.push({ x: 0.5 + r * Math.cos(a), y: 0.5 + r * Math.sin(a), sigma, w });
    }
  }
  return out;
}

function blurredCanvas(draw, blurPx = 6) {
  const c = makeCanvas(PAINT);
  const g = c.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, PAINT, PAINT);
  g.filter = `blur(${blurPx}px)`;
  draw(g);
  return c;
}

function heartCanvas() {
  return blurredCanvas((g) => {
    g.fillStyle = `rgb(${ORANGE.map((v) => Math.round(255 * v)).join(',')})`;
    g.beginPath();
    const s = PAINT / 256;
    g.moveTo(128 * s, 214 * s);
    g.bezierCurveTo(56 * s, 160 * s, 26 * s, 118 * s, 42 * s, 80 * s);
    g.bezierCurveTo(58 * s, 42 * s, 108 * s, 40 * s, 128 * s, 82 * s);
    g.bezierCurveTo(148 * s, 40 * s, 198 * s, 42 * s, 214 * s, 80 * s);
    g.bezierCurveTo(230 * s, 118 * s, 200 * s, 160 * s, 128 * s, 214 * s);
    g.fill();
  });
}

export async function mountRegister(root) {
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
  const lossParams = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
  const adamParamBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
  const panelParams = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
  const scaleParams = device.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST });
  const partial = device.createBuffer({ size: N_PARTIALS * 4, usage: U.STORAGE | U.COPY_SRC });
  const stepBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_DST });
  const sourceMix = device.createBuffer({ size: MAX_COMPONENTS * 32, usage: U.STORAGE | U.COPY_DST });
  const targetVals = device.createBuffer({ size: GRID * GRID * 16, usage: U.STORAGE | U.COPY_DST });
  const grid = gridPoints(GRID);

  const compute = (code, entryPoint, label) => device.createComputePipelineAsync({
    layout: 'auto', compute: { module: device.createShaderModule({ code, label }), entryPoint },
  });
  const [lossPipeline, scalePipeline, adam] = await Promise.all([
    compute(LOSS_SHADER, 'main', 'register.loss'),
    compute(SCALE_SHADER, 'main', 'register.scale'),
    createAdam(device),
  ]);

  const storage = (binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } });
  const panelBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      storage(1), storage(2), storage(3),
    ],
  });
  const format = navigator.gpu.getPreferredCanvasFormat();
  const panelModule = device.createShaderModule({ code: PANEL_SHADER, label: 'register.panels' });
  const panelLayout = device.createPipelineLayout({ bindGroupLayouts: [panelBGL] });
  const fragments = ['source', 'warped', 'target'];
  const panelPipelines = Object.fromEntries(await Promise.all(fragments.map(async (name) => [name,
    await device.createRenderPipelineAsync({
      layout: panelLayout,
      vertex: { module: panelModule, entryPoint: 'vs_fullscreen' },
      fragment: { module: panelModule, entryPoint: `fs_${name}`, targets: [{ format }] },
    })])));
  const contexts = {};
  for (const name of fragments) {
    contexts[name] = configureCanvas(device, canvases[name]).context;
    autoResize(canvases[name], null, 1024);
  }

  const state = {
    preset: 'recover',
    n: 12,
    stages: 1,
    lr: 0.5,
    running: false,
    stopAt: MAX_ITERS,
    best: Infinity,       // best loss of this fit, and the iteration it was reached at
    bestAt: 0,
    t: 1,
    playing: false,
    playClock: 0,
    displayDirty: true,
    visible: false,
    iteration: 0,
    history: [],
    engine: null,
    sets: [],
    display: null,        // second engine: the fitted chain at time t, for the warped panel
    displaySets: [],
    bg: null,
    m1: null,
    m2: null,
    nComp: 0,
    rateClock: { t: performance.now(), it: 0, value: 0 },
    readbackBusy: false,
    frame: 0,
    gen: 0,
    run: 0,
    gpu: { device, lossPipeline, partial, lossParams },
  };

  function setDensities(sourceMixture, targetValues) {
    device.queue.writeBuffer(sourceMix, 0, mixtureData(sourceMixture));
    device.queue.writeBuffer(targetVals, 0, targetValues);
    state.nComp = sourceMixture.length;
    state.displayDirty = true;
  }

  // -------------------------------------------------------------------------
  // Engine and bind groups (rebuilt when the tessellation or stage count changes)
  // -------------------------------------------------------------------------

  /** Chain of point sets: set s transforms set s-1's output and writes its
   *  dL/dpoints straight into set s-1's dL/dout, so the composed backward needs no copies. */
  function chain(engine, nP, first) {
    const sets = [];
    for (let s = 0; s < engine.batch; s++) {
      const prev = sets[s - 1];
      sets.push(engine.createPointSet({
        nP, batchOffset: s, batchCount: 1, pointsBatched: s > 0,
        points: prev?.out, gradPoints: prev ? prev.gradOut : false,
      }));
    }
    device.queue.writeBuffer(sets[0].points, 0, first);
    return sets;
  }

  async function build() {
    const gen = ++state.gen;
    const preset = state.preset;
    const options = { tess: [state.n, state.n], nSteps: N_STEPS, batch: state.stages };
    const [engine, display] = await Promise.all([CPAB2D.create(device, options), CPAB2D.create(device, options)]);
    if (gen !== state.gen) {
      engine.destroy();
      display.destroy();
      return;
    }
    const sets = chain(engine, GRID * GRID, grid);
    const last = sets[sets.length - 1];
    const displaySets = chain(display, GRID * GRID, grid);
    const nTheta = engine.batch * engine.d;
    const m1 = device.createBuffer({ size: nTheta * 4, usage: U.STORAGE | U.COPY_DST });
    const m2 = device.createBuffer({ size: nTheta * 4, usage: U.STORAGE | U.COPY_DST });
    const entries = (buffers) => buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));
    const bg = {
      loss: device.createBindGroup({
        layout: lossPipeline.getBindGroupLayout(0),
        entries: entries([lossParams, last.out, last.gradOut, partial, sourceMix, targetVals]),
      }),
      adam: device.createBindGroup({
        layout: adam.layout, entries: entries([adamParamBuf, engine.buffers.theta, engine.buffers.gradTheta, m1, m2, stepBuf]),
      }),
      scale: device.createBindGroup({
        layout: scalePipeline.getBindGroupLayout(0),
        entries: entries([scaleParams, engine.buffers.theta, display.buffers.theta]),
      }),
      panel: device.createBindGroup({
        layout: panelBGL,
        entries: entries([panelParams, displaySets[displaySets.length - 1].out, sourceMix, targetVals]),
      }),
    };
    if (state.engine) {
      for (const ps of state.sets) state.engine.destroyPointSet(ps);
      for (const ps of state.displaySets) state.display.destroyPointSet(ps);
      state.engine.destroy();
      state.display.destroy();
      state.m1.destroy();
      state.m2.destroy();
    }
    Object.assign(state, { engine, sets, display, displaySets, bg, m1, m2, builtPreset: preset });
    out('d').textContent = nTheta.toLocaleString();
    restart();
  }

  function restart() {
    const e = state.engine;
    if (!e) return;
    const nTheta = e.batch * e.d;
    e.writeTheta(gaussian(nTheta, INIT_STD, mulberry32(42 + state.run)));
    device.queue.writeBuffer(state.m1, 0, new Float32Array(nTheta));
    device.queue.writeBuffer(state.m2, 0, new Float32Array(nTheta));
    device.queue.writeBuffer(stepBuf, 0, new Uint32Array(4));
    const enc = device.createCommandEncoder();
    e.encodeClearMoments(enc);
    device.queue.submit([enc.finish()]);
    state.iteration = 0;
    state.stopAt = MAX_ITERS;
    state.best = Infinity;
    state.bestAt = 0;
    state.history = [];
    state.run++;
    state.rateClock = { t: performance.now(), it: 0, value: 0 };
    setRunning(false);
    setPlaying(false);
    setT(1);
    out('iter').textContent = '0';
    out('loss').textContent = '–';
    out('status').textContent = `Press Run to fit for up to ${MAX_ITERS.toLocaleString()} iterations. `
      + 'It stops early once the fit stops improving.';
    lockSettings(false);
    drawLoss();
  }

  // -------------------------------------------------------------------------
  // Presets
  // -------------------------------------------------------------------------

  /** rho_s(phi*(x_n)) for a random CPAB map phi* on the current tessellation. */
  async function hiddenWarpValues(mixture) {
    const eng = await CPAB2D.create(device, { tess: [state.n, state.n], nSteps: N_STEPS });
    const u = randomVelocities(eng.tess, mulberry32(1234 + state.n), { amplitude: 0.09, waves: 4, frequency: 1.2 });
    eng.writeTheta(thetaFromVelocities(eng.tess, u));
    const ps = eng.createPointSet({ nP: GRID * GRID });
    device.queue.writeBuffer(ps.points, 0, grid);
    const bytes = 2 * GRID * GRID * 4;
    const staging = device.createBuffer({ size: bytes, usage: U.COPY_DST | U.MAP_READ });
    const enc = device.createCommandEncoder();
    eng.encodeAffine(enc);
    eng.encodeExpm(enc);
    eng.encodeForward(enc, ps);
    enc.copyBufferToBuffer(ps.out, 0, staging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const map = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    eng.destroyPointSet(ps);
    eng.destroy();
    const N = GRID * GRID;
    const values = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) values.set(evalMixture(mixture, map[i], map[N + i]), 4 * i);
    return values;
  }

  async function loadDensities(name) {
    if (name === 'recover') {
      const mix = randomBlobs(7);
      setDensities(mix, await hiddenWarpValues(mix));
    } else if (name === 'shapes') {
      setDensities(diskMixture(), canvasValues(heartCanvas()));
    }
  }

  async function loadPreset(name) {
    const p = PRESETS[name];
    state.preset = name;
    out('note').textContent = p.note;
    setLr(p.lr);
    setTess(p.tess);
    state.stages = p.stages;
    await loadDensities(name);
    await build();
  }

  // -------------------------------------------------------------------------
  // Controls
  // -------------------------------------------------------------------------

  function setLr(log10) {
    state.lr = Math.pow(10, log10);
    q('lr').value = String(log10);
    out('lr').textContent = state.lr.toPrecision(2);
  }
  function setTess(n) {
    state.n = n;
    q('tess').value = String(n);
    out('tess').textContent = `${n}×${n}`;
  }
  /** reason, when stopping: 'paused' (by the user), 'converged' or 'limit' (MAX_ITERS). */
  function setRunning(on, reason = 'paused') {
    if (state.running && !on) {
      const it = state.iteration.toLocaleString();
      const head = {
        paused: `Paused at ${it} iterations.`,
        converged: `Converged after ${it} iterations.`,
        limit: `Stopped at ${it} iterations.`,
      }[reason];
      const tail = {
        paused: 'Run continues; ',
        converged: '',
        limit: `Run adds ${MAX_ITERS.toLocaleString()} more iterations; `,
      }[reason];
      out('status').textContent = `${head} Drag Time t or press Play flow to watch the fitted transformation; `
        + `${tail}Reset unlocks the settings.`;
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
  }
  /** The training settings are fixed from the first Run until Reset. */
  function lockSettings(locked) {
    for (const k of ['preset', 'tess', 'lr']) q(k).disabled = locked;
  }
  q('preset').addEventListener('change', (ev) => loadPreset(ev.target.value));
  q('tess').addEventListener('input', (ev) => { out('tess').textContent = `${ev.target.value}×${ev.target.value}`; });
  q('tess').addEventListener('change', async (ev) => {
    setTess(Number(ev.target.value));
    // The hidden warp lives on the chosen tessellation, so regenerate it.
    if (state.preset === 'recover') await loadDensities('recover');
    await build();
  });
  q('lr').addEventListener('input', (ev) => setLr(Number(ev.target.value)));
  q('run').addEventListener('click', () => {
    if (!state.running) {
      // Train until the loss plateaus (see readback) or state.stopAt, showing the fit at t = 1.
      // Resuming after a pause finishes the current batch; after the limit, start another.
      if (state.iteration >= state.stopAt) state.stopAt += MAX_ITERS;
      state.bestAt = state.iteration;
      lockSettings(true);
      setPlaying(false);
      setT(1);
      out('status').textContent = 'Optimizing until the fit stops improving…';
    }
    setRunning(!state.running);
    state.rateClock = { t: performance.now(), it: state.iteration, value: state.rateClock.value };
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
  new IntersectionObserver((entries) => {
    state.visible = entries.some((e) => e.isIntersecting);
  }, { threshold: 0.05 }).observe(ui);

  // -------------------------------------------------------------------------
  // Loss plot
  // -------------------------------------------------------------------------

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
    const p = PRESETS[state.builtPreset];
    const lf = new Float32Array(4);
    const lu = new Uint32Array(lf.buffer);
    lu[0] = GRID * GRID;
    lu[1] = state.nComp;
    lf[2] = 1 / (GRID * GRID);
    device.queue.writeBuffer(lossParams, 0, lf);

    device.queue.writeBuffer(adamParamBuf, 0, adamParams({
      n: e.batch * e.d,
      // Optional learning-rate decay (as the StepLR in the example scripts): halves every halfLife iterations.
      lr: state.lr * (p.halfLife ? Math.pow(0.5, state.iteration / p.halfLife) : 1),
      decay: p.decay ?? 0,
    }));

    device.queue.writeBuffer(panelParams, 0, new Uint32Array([
      GRID, q('diff').checked ? FLAG_DIFF : 0, state.nComp, 0,
    ]));

    const sf = new Float32Array(16);
    const su = new Uint32Array(sf.buffer);
    su[0] = e.batch * e.d;
    su[1] = e.d;
    sf.set(stageFactors(state.t, e.batch), 4);
    device.queue.writeBuffer(scaleParams, 0, sf);
  }

  /** Time t of the composed chain phi_m o ... o phi_1: stages before s = t * m run in
   *  full, stage floor(s) for the fraction s - floor(s), and later ones not at all.
   *  With one stage this is the stationary flow phi_{t theta}. */
  function stageFactors(t, m) {
    return Array.from({ length: m }, (_, k) => Math.min(1, Math.max(0, t * m - k)));
  }

  /** The warped panel's map: the display engine runs the chain with the scaled theta. */
  function encodeDisplay(enc) {
    const e = state.engine;
    const pass = enc.beginComputePass();
    pass.setPipeline(scalePipeline);
    pass.setBindGroup(0, state.bg.scale);
    pass.dispatchWorkgroups(Math.ceil((e.batch * e.d) / WG));
    pass.end();
    state.display.encodeAffine(enc);
    state.display.encodeExpm(enc);
    for (const ps of state.displaySets) state.display.encodeForward(enc, ps);
  }

  function encodeIteration(enc) {
    const e = state.engine;
    e.encodeAffine(enc);
    e.encodeExpm(enc);
    for (const ps of state.sets) e.encodeForward(enc, ps);
    const pass = enc.beginComputePass();
    pass.setPipeline(lossPipeline);
    pass.setBindGroup(0, state.bg.loss);
    pass.dispatchWorkgroups(N_PARTIALS);
    pass.end();
    e.encodeClearMoments(enc);
    for (let s = state.sets.length - 1; s >= 0; s--) e.encodeBackward(enc, state.sets[s], undefined, { accumulate: true });
    e.encodeProject(enc);
    adam.encode(enc, state.bg.adam, e.batch * e.d);
  }

  function renderPanels(enc) {
    for (const panel of fragments) {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: contexts[panel].getCurrentTexture().createView(),
          loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1],
        }],
      });
      pass.setPipeline(panelPipelines[panel]);
      pass.setBindGroup(0, state.bg.panel);
      pass.draw(3);
      pass.end();
    }
  }

  /** The loss, and while training the stopping rule (PATIENCE). */
  async function readback(lossCopy) {
    state.readbackBusy = true;
    const iteration = state.iteration;
    const run = state.run;
    await lossCopy.mapAsync(GPUMapMode.READ);
    const loss = new Float32Array(lossCopy.getMappedRange()).reduce((a, b) => a + b, 0);
    lossCopy.destroy();
    if (run === state.run && Number.isFinite(loss)) {
      state.history.push([iteration, loss]);
      if (state.history.length > 2000) state.history = state.history.filter((_, i) => i % 2 === 0);
      out('loss').textContent = loss.toExponential(2);
      drawLoss();
      if (state.running) {
        if (loss < state.best * (1 - MIN_GAIN)) {
          state.best = loss;
          state.bestAt = iteration;
        } else if (iteration - state.bestAt >= PATIENCE) {
          setRunning(false, 'converged');
        }
      }
    }
    state.readbackBusy = false;
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
    writeParams();
    const enc = device.createCommandEncoder();
    const iterated = state.running;
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
    let lossCopy = null;
    if (iterated && (reachedStop || (!state.readbackBusy && state.frame % READBACK_EVERY === 0))) {
      lossCopy = device.createBuffer({ size: N_PARTIALS * 4, usage: U.COPY_DST | U.MAP_READ });
      enc.copyBufferToBuffer(partial, 0, lossCopy, 0, N_PARTIALS * 4);
    }
    device.queue.submit([enc.finish()]);
    if (lossCopy) readback(lossCopy);

    if (reachedStop) setRunning(false, 'limit');
    if (iterated) {
      out('iter').textContent = state.iteration.toLocaleString();
      const rc = state.rateClock;
      if (now - rc.t > 500) {
        rc.value = ((state.iteration - rc.it) * 1000) / (now - rc.t);
        rc.t = now;
        rc.it = state.iteration;
        out('rate').textContent = Math.round(rc.value).toLocaleString();
      }
    }
  }

  await loadPreset('recover');
  requestAnimationFrame(frame);
  return { state };
}
