/**
 * Shared helpers for the FG-CPAB demos: the WebGPU device, canvases, images,
 * CPU-side evaluation of the CPA velocity field, and small WGSL snippets.
 */

let devicePromise = null;

/** Why navigator.gpu is missing, with a fix where there is one. */
function missingWebGPUReason() {
  if (!window.isSecureContext) {
    // Browsers expose WebGPU only to https:// pages and localhost. A local server
    // opened through http://[::]:8000 or a LAN address is not a secure context.
    const local = `http://localhost${location.port ? `:${location.port}` : ''}${location.pathname}`;
    return `Browsers only enable WebGPU on https:// pages and on localhost, and ${location.host} is neither. `
      + `Open ${local} instead (for a server on another machine, forward the port over SSH or serve over https).`;
  }
  const ua = navigator.userAgent;
  if (/Firefox\//.test(ua)) {
    return 'This Firefox build does not expose WebGPU (it ships on Windows and macOS first). '
      + 'Use Chrome, Edge or Safari, or enable dom.webgpu.enabled in about:config.';
  }
  if (/Linux/.test(ua) && /Chrome\//.test(ua)) {
    return 'Chrome on Linux may need WebGPU switched on: enable chrome://flags/#enable-unsafe-webgpu '
      + 'and chrome://flags/#enable-vulkan, then restart the browser.';
  }
  if (/Safari\//.test(ua) && !/Chrome\//.test(ua)) return 'Safari supports WebGPU from version 26.';
  return 'Use a recent version of Chrome, Edge or Safari.';
}

/** One GPUDevice for the whole page; all demos share it. */
export function getDevice() {
  devicePromise ??= (async () => {
    if (!navigator.gpu) throw new Error(missingWebGPUReason());
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('WebGPU is enabled, but the browser offered no GPU adapter. Check that hardware '
        + 'acceleration is on (in Chrome, chrome://gpu shows the WebGPU status).');
    }
    const device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (ev) => reportError(ev.error));
    device.lost.then((info) => {
      if (info.reason !== 'destroyed') reportError(new Error(`WebGPU device lost: ${info.message || info.reason}`));
    });
    return device;
  })();
  return devicePromise;
}

/**
 * Show a demo or GPU error on the page. A demo that fails after WebGPU starts
 * otherwise stays black, with the reason only in the console (which a phone
 * does not show). Repeated messages are shown once; a click dismisses the box.
 */
export function reportError(err) {
  console.error(err);
  const message = String(err?.message ?? err);
  let box = document.querySelector('.demo-error');
  if (!box) {
    box = el('<div class="demo-error" role="alert"><strong>A demo stopped with an error on this device.</strong></div>');
    box.addEventListener('click', () => box.remove());
    document.body.append(box);
  }
  const lines = [...box.querySelectorAll('p')];
  if (lines.length >= 4 || lines.some((p) => p.textContent === message)) return;
  const line = document.createElement('p');
  line.textContent = message;
  box.append(line);
}

export function showFallback(container, err) {
  container.classList.add('demo-fallback');
  const note = document.createElement('div');
  note.className = 'fallback-note';
  note.innerHTML = '<strong>The interactive demo needs WebGPU.</strong> ';
  note.append(document.createTextNode(err?.message ?? 'Use a recent version of Chrome, Edge or Safari.'));
  container.replaceChildren(note);
}

export function configureCanvas(device, canvas) {
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  return { context, format };
}

/** Keep the canvas backing store matched to its CSS size (capped). */
export function autoResize(canvas, onResize, maxPx = 2048) {
  const observer = new ResizeObserver(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.min(maxPx, Math.round(canvas.clientWidth * dpr)));
    const h = Math.max(1, Math.min(maxPx, Math.round(canvas.clientHeight * dpr)));
    if (w !== canvas.width || h !== canvas.height) {
      canvas.width = w;
      canvas.height = h;
      onResize?.();
    }
  });
  observer.observe(canvas);
  return observer;
}

export function el(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

export function makeCanvas(size) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  return c;
}

/** Procedural test card: checkerboard, rings and a title, so warps are easy to read. */
export function makeTestCard(size = 512) {
  const c = makeCanvas(size);
  const g = c.getContext('2d');
  const n = 16;
  const s = size / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const t = (i + j) / (2 * n - 2);
      const light = (i + j) % 2 === 0;
      const hue = 200 + 120 * t;
      g.fillStyle = `hsl(${hue} ${light ? 70 : 55}% ${light ? 88 : 62}%)`;
      g.fillRect(j * s, i * s, s + 1, s + 1);
    }
  }
  g.lineWidth = size / 64;
  for (let r = 1; r <= 4; r++) {
    g.strokeStyle = r % 2 ? 'rgba(20, 24, 40, 0.85)' : 'rgba(255, 255, 255, 0.9)';
    g.beginPath();
    g.arc(size / 2, size / 2, (r * size) / 11, 0, 2 * Math.PI);
    g.stroke();
  }
  g.fillStyle = 'rgba(20, 24, 40, 0.92)';
  g.fillRect(size * 0.18, size * 0.43, size * 0.64, size * 0.14);
  g.fillStyle = '#fff';
  g.font = `700 ${Math.round(size * 0.085)}px system-ui, -apple-system, Segoe UI, sans-serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('FG-CPAB', size / 2, size / 2 + size * 0.004);
  return c;
}

/** Draw `source` into a square canvas, cropping to cover (optionally mirrored, for webcams). */
export function drawCover(canvas, source, { mirror = false } = {}) {
  const g = canvas.getContext('2d');
  const sw = source.videoWidth || source.width;
  const sh = source.videoHeight || source.height;
  const side = Math.min(sw, sh);
  const sx = (sw - side) / 2;
  const sy = (sh - side) / 2;
  g.save();
  g.imageSmoothingQuality = 'high';   // the default bilinear filter aliases on large downscales
  if (mirror) {
    g.translate(canvas.width, 0);
    g.scale(-1, 1);
  }
  g.drawImage(source, sx, sy, side, side, 0, 0, canvas.width, canvas.height);
  g.restore();
  return canvas;
}

export function createImageTexture(device, size, mipLevelCount = 1) {
  return device.createTexture({
    size: [size, size],
    format: 'rgba8unorm',
    mipLevelCount,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

/** Upload a square canvas to every mip level of `texture` (levels are downscaled with drawImage). */
export function uploadCanvas(device, texture, canvas) {
  let level = canvas;
  for (let m = 0; m < texture.mipLevelCount; m++) {
    const size = texture.width >> m;
    if (level.width !== size) {
      const next = makeCanvas(size);
      next.getContext('2d').drawImage(level, 0, 0, size, size);
      level = next;
    }
    device.queue.copyExternalImageToTexture({ source: level }, { texture, mipLevel: m }, [size, size]);
  }
}

// ---------------------------------------------------------------------------
// CPU-side CPA field (for resampling theta between tessellations and seeding)
// ---------------------------------------------------------------------------

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Triangle containing (x, y) in [0,1]^2; same cell order as the kernels. */
export function findCell(tess, x, y) {
  const qx = Math.min(Math.max(x, 0), 1 - 1e-7) * tess.nx;
  const qy = Math.min(Math.max(y, 0), 1 - 1e-7) * tess.ny;
  const cx = Math.min(Math.floor(qx), tess.nx - 1);
  const cy = Math.min(Math.floor(qy), tess.ny - 1);
  const lx = qx - cx;
  const ly = qy - cy;
  let tri = ((lx < ly) << 1) | ((1 - lx) < ly);
  tri ^= (tri >> 1) & 1;
  return (cx + cy * tess.nx) * 4 + tri;
}

/** Normal samples (Box-Muller) with standard deviation `std`. */
export function gaussian(count, std, rng) {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const u = Math.max(rng(), 1e-12);
    out[i] = std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  return out;
}

/** Velocity v(x, y) of the CPA field with parameters theta (Python basis). */
export function velocityAt(tess, theta, x, y) {
  const c = findCell(tess, x, y);
  let vx = 0;
  let vy = 0;
  for (let s = 0; s < 3; s++) {
    const a = tess.cellDof[4 * c + s];
    if (a < 0) continue;
    const o = 9 * c + 3 * s;
    const w = tess.cellCoef[o] * x + tess.cellCoef[o + 1] * y + tess.cellCoef[o + 2];
    vx += w * theta[2 * a];
    vy += w * theta[2 * a + 1];
  }
  return [vx, vy];
}

/** Nodal velocities u [nActive * 2] of the field `theta` defined on another tessellation. */
export function resampleVelocities(fromTess, fromTheta, toTess) {
  const u = new Float32Array(toTess.d);
  for (let a = 0; a < toTess.nActive; a++) {
    const v = toTess.activeVerts[a];
    const [vx, vy] = velocityAt(fromTess, fromTheta, toTess.xy[2 * v], toTess.xy[2 * v + 1]);
    u[2 * a] = vx;
    u[2 * a + 1] = vy;
  }
  return u;
}

/** Smooth random nodal velocities: a few random plane waves evaluated at the vertices. */
export function randomVelocities(tess, rng, { amplitude = 0.1, waves = 5, frequency = 1.6 } = {}) {
  const ws = Array.from({ length: waves }, () => {
    const ang = rng() * 2 * Math.PI;
    const f = frequency * (0.5 + rng());
    return {
      kx: Math.cos(ang) * f, ky: Math.sin(ang) * f, phase: rng() * 2 * Math.PI,
      ax: (rng() - 0.5) * 2, ay: (rng() - 0.5) * 2,
    };
  });
  const u = new Float32Array(tess.d);
  for (let a = 0; a < tess.nActive; a++) {
    const v = tess.activeVerts[a];
    const x = tess.xy[2 * v];
    const y = tess.xy[2 * v + 1];
    for (const w of ws) {
      const s = Math.sin(2 * Math.PI * (w.kx * x + w.ky * y) + w.phase);
      u[2 * a] += (amplitude / Math.sqrt(waves)) * w.ax * s;
      u[2 * a + 1] += (amplitude / Math.sqrt(waves)) * w.ay * s;
    }
  }
  return u;
}

// ---------------------------------------------------------------------------
// WGSL snippets
// ---------------------------------------------------------------------------

/** Fullscreen triangle with uv in [0,1]^2, origin at the top-left (image convention). */
export const WGSL_FULLSCREEN = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VSOut {
  let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VSOut;
  o.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2<f32>(xy.x, 1.0 - xy.y);
  return o;
}
`;

/**
 * Anti-aliased triangle edges of an nx x ny CPAB tessellation, evaluated at a
 * point p given in source (undeformed) coordinates. The four line families
 * x*nx, y*ny, x*nx - y*ny, x*nx + y*ny take integer values exactly on the cell
 * borders and the two diagonals, so drawing them at p = phi(uv) shows the
 * tessellation carried by the deformation, at constant screen width.
 */
export const WGSL_MESH_LINES = /* wgsl */ `
fn mesh_lines(p: vec2<f32>, n: vec2<f32>, widthPx: f32) -> f32 {
  let f = vec4<f32>(p.x * n.x, p.y * n.y, p.x * n.x - p.y * n.y, p.x * n.x + p.y * n.y);
  let d = abs(fract(f + 0.5) - 0.5);
  let w = max(fwidth(f), vec4<f32>(1e-6));
  let a = 1.0 - smoothstep(vec4<f32>(0.0), w * widthPx, d);
  return max(max(a.x, a.y), 0.75 * max(a.z, a.w));
}
`;

/** Colormaps: a sequential ramp (dark blue -> magenta -> amber) and a diverging blue-white-red. */
export const WGSL_COLORMAPS = /* wgsl */ `
fn ramp(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0) * 4.0;
  let c0 = vec3<f32>(0.05, 0.03, 0.18);
  let c1 = vec3<f32>(0.32, 0.07, 0.48);
  let c2 = vec3<f32>(0.72, 0.18, 0.44);
  let c3 = vec3<f32>(0.97, 0.50, 0.20);
  let c4 = vec3<f32>(0.99, 0.92, 0.62);
  if (x < 1.0) { return mix(c0, c1, x); }
  if (x < 2.0) { return mix(c1, c2, x - 1.0); }
  if (x < 3.0) { return mix(c2, c3, x - 2.0); }
  return mix(c3, c4, x - 3.0);
}

fn diverging(t: f32) -> vec3<f32> {
  let blue = vec3<f32>(0.19, 0.42, 0.85);
  let mid = vec3<f32>(0.97, 0.97, 0.96);
  let red = vec3<f32>(0.85, 0.26, 0.22);
  let s = clamp(t, -1.0, 1.0);
  return select(mix(mid, blue, -s), mix(mid, red, s), s > 0.0);
}
`;

/** Cell lookup for points inside [0,1]^2 (cuda_findcellidx_2D_fast without the outside branches). */
export const WGSL_FIND_CELL = /* wgsl */ `
fn find_cell_in(p: vec2<f32>, ncx: u32, ncy: u32) -> u32 {
  let n = vec2<f32>(f32(ncx), f32(ncy));
  let q = clamp(p, vec2<f32>(0.0), vec2<f32>(1.0 - 1e-7)) * n;
  let c = min(vec2<u32>(floor(q)), vec2<u32>(ncx - 1u, ncy - 1u));
  let l = q - vec2<f32>(c);
  let bit1 = select(0u, 1u, l.x < l.y);
  let bit0 = select(0u, 1u, (1.0 - l.x) < l.y);
  var tri = (bit1 << 1u) | bit0;
  tri = tri ^ ((tri >> 1u) & 1u);
  return (c.x + c.y * ncx) * 4u + tri;
}
`;

// ---------------------------------------------------------------------------
// Adam on the GPU
// ---------------------------------------------------------------------------

const ADAM_WG = 256;

const WGSL_ADAM = /* wgsl */ `
struct AdamParams {
  n: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  decay: f32,
  _p1: f32,
  _p2: f32,
};
@group(0) @binding(0) var<uniform> P: AdamParams;
@group(0) @binding(1) var<storage, read_write> theta: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m1: array<f32>;
@group(0) @binding(4) var<storage, read_write> m2: array<f32>;
@group(0) @binding(5) var<storage, read_write> step: array<u32>;

@compute @workgroup_size(1)
fn tick() { step[0] = step[0] + 1u; }

// Adam with optional decoupled weight decay (AdamW); decay = 0 is plain Adam.
@compute @workgroup_size(${ADAM_WG})
fn adam(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let g = grad[i];
  let a = P.beta1 * m1[i] + (1.0 - P.beta1) * g;
  let b = P.beta2 * m2[i] + (1.0 - P.beta2) * g * g;
  m1[i] = a;
  m2[i] = b;
  let t = f32(step[0]);
  let ah = a / (1.0 - pow(P.beta1, t));
  let bh = b / (1.0 - pow(P.beta2, t));
  theta[i] = theta[i] - P.lr * (ah / (sqrt(bh) + P.eps) + P.decay * theta[i]);
}
`;

/**
 * Adam as torch.optim.Adam. Bind groups use `layout`, with bindings
 * [params (adamParams), theta, grad, m1, m2, step counter]; `encode` records
 * one step (advance the counter, then update all n parameters).
 */
export async function createAdam(device) {
  const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  const layout = device.createBindGroupLayout({
    entries: [entry(0, 'uniform'), entry(1, 'storage'), entry(2, 'read-only-storage'),
      entry(3, 'storage'), entry(4, 'storage'), entry(5, 'storage')],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const module = device.createShaderModule({ code: WGSL_ADAM, label: 'adam' });
  const [adam, tick] = await Promise.all(['adam', 'tick'].map((entryPoint) =>
    device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } })));
  return {
    layout,
    encode(enc, bindGroup, n) {
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, bindGroup);
      pass.setPipeline(tick);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(adam);
      pass.dispatchWorkgroups(Math.ceil(n / ADAM_WG));
      pass.end();
    },
  };
}

/** The 32-byte uniform block for createAdam. */
export function adamParams({ n, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, decay = 0 }) {
  const f = new Float32Array(8);
  new Uint32Array(f.buffer)[0] = n;
  f.set([lr, beta1, beta2, eps, decay], 1);
  return f;
}

// ---------------------------------------------------------------------------
// Loss plot
// ---------------------------------------------------------------------------

/** Log-scale curve of [iteration, value] pairs, in the page's theme colors. */
export function drawLogCurve(canvas, history) {
  const g = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const css = getComputedStyle(canvas);
  g.clearRect(0, 0, W, H);
  if (history.length < 2) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pad = 10 * dpr;
  const logs = history.map(([, l]) => Math.log10(Math.max(l, 1e-12)));
  const lo = Math.min(...logs);
  const hi = Math.max(...logs);
  const maxIt = Math.max(history[history.length - 1][0], 1);
  g.strokeStyle = (css.getPropertyValue('--line').trim() || '#ccc');
  g.lineWidth = dpr;
  for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) {
    const y = pad + (H - 2 * pad) * (1 - (k - lo) / Math.max(hi - lo, 1e-6));
    g.beginPath();
    g.moveTo(pad, y);
    g.lineTo(W - pad, y);
    g.stroke();
    g.fillStyle = (css.getPropertyValue('--ink-3').trim() || '#888');
    g.font = `${11 * dpr}px Inter, system-ui, sans-serif`;
    g.fillText(`1e${k}`, pad + 2 * dpr, y - 3 * dpr);
  }
  g.strokeStyle = (css.getPropertyValue('--accent').trim() || '#2f5bd3');
  g.lineWidth = 2 * dpr;
  g.beginPath();
  history.forEach(([it], i) => {
    const x = pad + (W - 2 * pad) * (it / maxIt);
    const y = pad + (H - 2 * pad) * (1 - (logs[i] - lo) / Math.max(hi - lo, 1e-6));
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  });
  g.stroke();
}

/** [2, n*n] point grid over [0,1]^2 (x fastest), as Cpab.uniform_meshgrid. */
export function gridPoints(n) {
  const pts = new Float32Array(2 * n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      pts[i * n + j] = j / (n - 1);
      pts[n * n + i * n + j] = i / (n - 1);
    }
  }
  return pts;
}

/** Centres of a w x h pixel grid in [0, 1]^2, row-major from the top left (all x, then all y). */
export function pixelCenters(w, h) {
  const pts = new Float32Array(2 * w * h);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      pts[i * w + j] = (j + 0.5) / w;
      pts[w * h + i * w + j] = (i + 0.5) / h;
    }
  }
  return pts;
}
