/**
 * Option 1 — "Sculpt a diffeomorphism".
 *
 * Dragging on the image adds velocity to nearby tessellation vertices (the
 * nodal parameterisation of the CPA basis, Appendix E). Every frame the image
 * is pulled back through phi_{-t theta} = phi_{t theta}^{-1}: every canvas pixel
 * is integrated on the GPU with the WebGPU port of the CUDA kernels. The
 * tessellation and velocity field are drawn on top.
 */
import { CPAB2D } from '../src/cpab2d.js';
import { thetaFromVelocities } from '../src/tessellation2d.js';
import {
  getDevice, showFallback, configureCanvas, autoResize, el, makeCanvas, makeTestCard, drawCover,
  createImageTexture, uploadCanvas, mulberry32, resampleVelocities, randomVelocities, pixelCenters,
  WGSL_FULLSCREEN, WGSL_MESH_LINES, WGSL_FIND_CELL,
} from './common.js';

const IMG = 1024;     // source image resolution (the stage is up to 560 CSS px, ~1120 device px)
const N_STEPS = 50;   // Cpab default nstepsolver
const ARROWS = 18;    // velocity arrows per side
const FLAG_MESH = 1;
const TEST_CARD = new URL('../data/test-card.jpg', import.meta.url);

const SHADER = WGSL_FULLSCREEN + WGSL_MESH_LINES + /* wgsl */ `
struct Uniforms {
  canvas: vec2<f32>,
  _pad0: u32,
  flags: u32,
  ncx: u32,
  ncy: u32,
  arrowGrid: u32,
  arrowScale: f32,
  dpr: f32,
  _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> mapBuf: array<f32>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

// Source position of the output pixel at framebuffer position pos: the map holds
// one integrated point per canvas pixel (all x, then all y), so no interpolation.
fn map_at(pos: vec2<f32>) -> vec2<f32> {
  let w = u32(U.canvas.x);
  let i = u32(pos.x) + u32(pos.y) * w;
  return vec2<f32>(mapBuf[i], mapBuf[i + w * u32(U.canvas.y)]);
}

@fragment
fn fs_image(in: VSOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let p = map_at(in.pos.xy);
  var col = textureSampleLevel(src, samp, p, 0.0).rgb;

  // The tessellation lives on the fixed domain, like the velocity field: draw it
  // at the output pixel, not at its warped source position.
  let lines = mesh_lines(uv, vec2<f32>(f32(U.ncx), f32(U.ncy)), 0.9 * U.dpr);
  if ((U.flags & ${FLAG_MESH}u) != 0u) {
    col = mix(col, vec3<f32>(0.04, 0.05, 0.10), 0.55 * lines);
  }
  return vec4<f32>(col, 1.0);
}
`;

const ARROW_SHADER = WGSL_FIND_CELL + /* wgsl */ `
struct Uniforms {
  canvas: vec2<f32>,
  _pad0: u32,
  flags: u32,
  ncx: u32,
  ncy: u32,
  arrowGrid: u32,
  arrowScale: f32,
  dpr: f32,
  _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> As: array<f32>;

// One arrow per instance: the CPA velocity v(g) = A_c [g; 1] at a grid point g,
// drawn as a quad around the arrow and shaded from its signed distance, so the
// edges are antialiased and a soft shadow keeps it readable on any image.
// The length grows with the speed but levels off below the grid spacing, so
// arrows never run into each other; the colour carries the speed instead.
struct AOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,   // px: along the arrow from its tail, and across it
  @location(1) shape: vec4<f32>,   // length, head length, head half width, shaft half width
  @location(2) color: vec4<f32>,
};

const SHADOW = 3.0;   // soft shadow width, CSS px

// Slow to fast: indigo, magenta, gold.
fn speed_color(speed: f32) -> vec3<f32> {
  let k = smoothstep(-6.0, -2.3, log2(max(speed, 1e-6)));
  let slow = vec3<f32>(0.42, 0.42, 1.0);
  let mid = vec3<f32>(0.96, 0.34, 0.64);
  let fast = vec3<f32>(1.0, 0.82, 0.28);
  return select(mix(mid, fast, 2.0 * k - 1.0), mix(slow, mid, 2.0 * k), k < 0.5);
}

@vertex
fn vs_arrow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> AOut {
  let k = U.arrowGrid;
  let g = (vec2<f32>(f32(ii % k), f32(ii / k)) + 0.5) / f32(k);
  let o = find_cell_in(g, U.ncx, U.ncy) * 6u;
  let v = vec2<f32>(As[o] * g.x + As[o + 1u] * g.y + As[o + 2u],
                    As[o + 3u] * g.x + As[o + 4u] * g.y + As[o + 5u]);
  let speed = length(v);
  let dir = select(vec2<f32>(1.0, 0.0), v / max(speed, 1e-9), speed > 1e-9);
  let nrm = vec2<f32>(-dir.y, dir.x);

  // Displacement per unit time, in px, eased to at most 0.95 grid spacings.
  let maxLen = 0.95 * U.canvas.x / f32(k);
  let len = maxLen * (1.0 - exp(-speed * U.arrowScale * U.canvas.x / (1.2 * maxLen)));
  let head = min(7.5 * U.dpr, 0.5 * len);
  let headHalf = 0.62 * head;
  let shaft = min(1.3 * U.dpr, 0.4 * headHalf + 0.3 * U.dpr);
  let pad = SHADOW * U.dpr + 1.0;

  // The quad spans the arrow plus its shadow, centred on the grid point.
  var along = array<f32, 6>(-pad, len + pad, len + pad, -pad, len + pad, -pad);
  var across = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
  let w = headHalf + pad;
  let local = vec2<f32>(along[vi], across[vi] * w);
  let p = g * U.canvas + dir * (local.x - 0.5 * len) + nrm * local.y;

  var out: AOut;
  out.pos = vec4<f32>(p.x / U.canvas.x * 2.0 - 1.0, 1.0 - p.y / U.canvas.y * 2.0, 0.0, 1.0);
  out.local = local;
  out.shape = vec4<f32>(len, head, headHalf, shaft);
  out.color = vec4<f32>(speed_color(speed * U.arrowScale), smoothstep(3.0 * U.dpr, 9.0 * U.dpr, len));
  return out;
}

// Isosceles triangle with its apex at the origin, opening along +y to height q.y
// and half width q.x (after Inigo Quilez).
fn sd_triangle(p0: vec2<f32>, q: vec2<f32>) -> f32 {
  let p = vec2<f32>(abs(p0.x), p0.y);
  let a = p - q * clamp(dot(p, q) / dot(q, q), 0.0, 1.0);
  let b = p - q * vec2<f32>(clamp(p.x / q.x, 0.0, 1.0), 1.0);
  let s = -sign(q.y);
  let d = min(vec2<f32>(dot(a, a), s * (p.x * q.y - p.y * q.x)), vec2<f32>(dot(b, b), s * (p.y - q.y)));
  return -sqrt(d.x) * sign(d.y);
}

@fragment
fn fs_arrow(in: AOut) -> @location(0) vec4<f32> {
  let len = in.shape.x;
  let head = in.shape.y;
  let x = in.local.x;
  let y = in.local.y;
  let shaftEnd = len - 0.6 * head;
  let dShaft = length(vec2<f32>(x - clamp(x, 0.0, shaftEnd), y)) - in.shape.w;
  let dHead = sd_triangle(vec2<f32>(y, len - x), vec2<f32>(in.shape.z, head));
  let d = min(dShaft, dHead);
  let fill = clamp(0.5 - d, 0.0, 1.0);
  let shadow = 0.55 * (1.0 - smoothstep(-0.5, SHADOW * U.dpr, d));
  let a = fill + (1.0 - fill) * shadow;
  let rgb = in.color.rgb * fill + vec3<f32>(0.03, 0.04, 0.08) * (1.0 - fill) * shadow;
  return vec4<f32>(rgb, a) * in.color.a;
}
`;

const TEMPLATE = `
<div class="demo demo-sculpt">
  <div class="demo-stage square">
    <canvas class="demo-canvas" aria-label="Interactive CPAB warp. Drag to deform the image."></canvas>
    <div class="demo-hint">Drag on the image to sculpt it</div>
    <div class="demo-toast" role="status" hidden>
      <button type="button" class="demo-toast-close" data-k="toast-close" aria-label="Dismiss">×</button>
      <p><strong>Hey! That's animal cruelty </strong><br>Leave the corgi alone and warp your own picture.</p>
      <div class="demo-toast-actions">
        <button type="button" data-k="toast-upload">Upload a photo</button>
        <button type="button" data-k="toast-webcam">Use webcam</button>
      </div>
    </div>
  </div>
  <div class="demo-panel">
    <div class="demo-controls">
      <label class="ctl">
        <span class="ctl-name">Tessellation</span>
        <input type="range" min="2" max="40" step="1" value="12" data-k="tess">
        <output data-o="tess"></output>
      </label>
      <label class="ctl">
        <span class="ctl-name">Brush radius</span>
        <input type="range" min="0.03" max="0.3" step="0.005" value="0.12" data-k="radius">
        <output data-o="radius"></output>
      </label>
      <label class="ctl">
        <span class="ctl-name">Time t</span>
        <input type="range" min="-1" max="1" step="0.001" value="1" data-k="t">
        <output data-o="t"></output>
      </label>
    </div>
    <div class="demo-buttons">
      <button type="button" data-k="play" aria-pressed="false">Play flow</button>
      <button type="button" data-k="random">Random warp</button>
      <button type="button" data-k="reset">Reset</button>
      <span class="sep"></span>
      <label class="toggle"><input type="checkbox" data-k="mesh"> Show tessellation</label>
      <label class="toggle"><input type="checkbox" data-k="arrows"> Show velocity field</label>
      <span class="sep"></span>
      <button type="button" data-k="card">Test card</button>
      <button type="button" data-k="upload">Upload image</button>
      <button type="button" data-k="webcam" aria-pressed="false">Webcam</button>
      <input type="file" accept="image/*" data-k="file" hidden>
    </div>
    <p class="demo-stats" data-o="stats"></p>
  </div>
</div>`;

export async function mountSculpt(root) {
  const ui = el(TEMPLATE);
  root.replaceChildren(ui);
  const q = (k) => ui.querySelector(`[data-k="${k}"]`);
  const out = (k) => ui.querySelector(`[data-o="${k}"]`);
  const stage = ui.querySelector('.demo-stage');
  const canvas = ui.querySelector('canvas');

  let device;
  try {
    device = await getDevice();
  } catch (err) {
    showFallback(stage, err);
    return;
  }

  // The test card image, or a drawn card if the file cannot be loaded.
  const testCard = fetch(TEST_CARD)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`test card: HTTP ${r.status}`))))
    .then((blob) => createImageBitmap(blob))
    .catch(() => makeTestCard(IMG));

  const { context, format } = configureCanvas(device, canvas);
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  const srcTexture = createImageTexture(device, IMG);
  const srcCanvas = makeCanvas(IMG);
  const uniforms = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const imageModule = device.createShaderModule({ code: SHADER, label: 'sculpt.image' });
  const arrowModule = device.createShaderModule({ code: ARROW_SHADER, label: 'sculpt.arrows' });
  const imagePipeline = await device.createRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: imageModule, entryPoint: 'vs_fullscreen' },
    fragment: { module: imageModule, entryPoint: 'fs_image', targets: [{ format }] },
  });
  const arrowPipeline = await device.createRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: arrowModule, entryPoint: 'vs_arrow' },
    fragment: {
      module: arrowModule, entryPoint: 'fs_arrow',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      }],
    },
  });

  const state = {
    n: 12,
    engine: null,
    pixels: null,
    imageBG: null,
    arrowBG: null,
    u: null,              // nodal velocities [nActive * 2]
    theta: null,
    t: 1,
    playing: false,
    playClock: 0,
    radius: 0.12,
    hint: true,           // "Drag on the image" hint, until the first interaction
    onCard: true,         // the test card is the current image
    teased: false,        // the "animal cruelty" note was shown (once per visit)
    dirty: true,
    webcam: null,
    rng: mulberry32(7),
    tessGen: 0,
  };

  // One point at the centre of every canvas pixel, mapped by batch 1 (see buildEngine).
  function makePixels(engine) {
    const w = canvas.width;
    const h = canvas.height;
    const pixels = Object.assign(engine.createPointSet({ nP: w * h, batchOffset: 1, batchCount: 1 }), { w, h });
    device.queue.writeBuffer(pixels.points, 0, pixelCenters(w, h));
    const imageBG = device.createBindGroup({
      layout: imagePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: pixels.out } },
        { binding: 2, resource: srcTexture.createView() },
        { binding: 3, resource: sampler },
      ],
    });
    return { pixels, imageBG };
  }

  async function buildEngine(n) {
    const engine = await CPAB2D.create(device, { tess: [n, n], nSteps: N_STEPS, adaptive: true, batch: 2 });
    // Batch 0 holds theta, for the velocity arrows: v_theta is stationary, so they
    // do not change with t. Batch 1 holds -t*theta: the pull-back map
    // phi^-1 = phi_{-t theta} evaluated at every output pixel.
    const arrowBG = device.createBindGroup({
      layout: arrowPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: engine.buffers.affine } },
      ],
    });
    return { engine, ...makePixels(engine), arrowBG };
  }

  async function setTessellation(n) {
    const gen = ++state.tessGen;
    const next = await buildEngine(n);
    if (gen !== state.tessGen) {  // a newer request superseded this one
      next.engine.destroyPointSet(next.pixels);
      next.engine.destroy();
      return;
    }
    if (state.engine) {
      const u = resampleVelocities(state.engine.tess, state.theta, next.engine.tess);
      state.engine.destroyPointSet(state.pixels);
      state.engine.destroy();
      state.u = u;
    } else {
      state.u = new Float32Array(next.engine.d);
    }
    Object.assign(state, next, { n });
    state.theta = thetaFromVelocities(state.engine.tess, state.u);
    updateStats();
    state.dirty = true;
  }

  function updateStats() {
    const e = state.engine;
    const { w, h } = state.pixels;
    out('tess').textContent = `${state.n}×${state.n}`;
    out('stats').textContent = `${e.d.toLocaleString()} parameters (d) · ${e.nC.toLocaleString()} triangles · `
      + `${(w * h).toLocaleString()} pixels (${w}×${h}) × ${N_STEPS} integration steps per frame, on your GPU`;
  }

  function setImage(source, opts) {
    drawCover(srcCanvas, source, opts);
    uploadCanvas(device, srcTexture, srcCanvas);
    state.dirty = true;
  }

  /** After the first warp of the test card, ask the user to warp their own picture instead. */
  function tease() {
    if (state.teased || !state.onCard) return;
    state.teased = true;
    setTimeout(() => {
      if (!state.onCard) return;
      const toast = ui.querySelector('.demo-toast');
      toast.hidden = false;
      setTimeout(() => { toast.hidden = true; }, 10000);
    }, 400);
  }

  function hideToast() {
    ui.querySelector('.demo-toast').hidden = true;
  }

  function hideHint() {
    if (!state.hint) return;
    state.hint = false;
    ui.querySelector('.demo-hint').classList.add('hidden');
  }

  function brush(x, y, dx, dy) {
    const tess = state.engine.tess;
    const r2 = 2 * state.radius * state.radius;
    for (let a = 0; a < tess.nActive; a++) {
      const v = tess.activeVerts[a];
      const ox = tess.xy[2 * v] - x;
      const oy = tess.xy[2 * v + 1] - y;
      const w = Math.exp(-(ox * ox + oy * oy) / r2);
      if (w < 1e-3) continue;
      state.u[2 * a] += dx * w;
      state.u[2 * a + 1] += dy * w;
    }
    thetaFromVelocities(tess, state.u, state.theta);
    state.dirty = true;
  }

  // Pointer sculpting. Each pointer (finger) keeps its own last position, so a
  // multi-touch drag never takes a stroke from one finger to another.
  const strokes = new Map();   // pointerId -> last uv
  let dragged = false;
  const toUV = (ev) => {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
  };
  canvas.addEventListener('pointerdown', (ev) => {
    hideHint();
    setPlaying(false);
    setT(1);
    canvas.setPointerCapture(ev.pointerId);
    if (strokes.size === 0) dragged = false;
    strokes.set(ev.pointerId, toUV(ev));
  });
  canvas.addEventListener('pointermove', (ev) => {
    const last = strokes.get(ev.pointerId);
    if (!last) return;
    const p = toUV(ev);
    brush(last[0], last[1], p[0] - last[0], p[1] - last[1]);
    dragged ||= p[0] !== last[0] || p[1] !== last[1];
    strokes.set(ev.pointerId, p);
  });
  const endDrag = (ev) => {
    if (!strokes.delete(ev.pointerId)) return;
    if (strokes.size === 0 && dragged) tease();
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  // Controls.
  function setT(t) {
    state.t = t;
    q('t').value = String(t);
    out('t').textContent = t.toFixed(2);
    state.dirty = true;
  }
  function setPlaying(on) {
    state.playing = on;
    q('play').setAttribute('aria-pressed', String(on));
    q('play').textContent = on ? 'Pause' : 'Play flow';
  }

  q('t').addEventListener('input', (ev) => { hideHint(); setPlaying(false); setT(Number(ev.target.value)); });
  q('radius').addEventListener('input', (ev) => {
    state.radius = Number(ev.target.value);
    out('radius').textContent = state.radius.toFixed(2);
  });
  q('tess').addEventListener('input', (ev) => { out('tess').textContent = `${ev.target.value}×${ev.target.value}`; });
  q('tess').addEventListener('change', (ev) => setTessellation(Number(ev.target.value)));
  q('play').addEventListener('click', () => {
    hideHint();
    setPlaying(!state.playing);
    state.playClock = Math.acos(Math.min(1, Math.max(-1, 1 - 2 * state.t)));
  });
  q('random').addEventListener('click', () => {
    hideHint();
    state.u = randomVelocities(state.engine.tess, state.rng, { amplitude: 0.14 });
    thetaFromVelocities(state.engine.tess, state.u, state.theta);
    setT(1);
    state.dirty = true;
    tease();
  });
  q('reset').addEventListener('click', () => {
    hideHint();
    setPlaying(false);
    state.u.fill(0);
    state.theta.fill(0);
    setT(1);
  });
  for (const k of ['mesh', 'arrows']) q(k).addEventListener('change', () => { state.dirty = true; });
  q('card').addEventListener('click', async () => {
    stopWebcam();
    setImage(await testCard);
    state.onCard = true;
  });
  q('toast-close').addEventListener('click', hideToast);
  q('toast-upload').addEventListener('click', () => { hideToast(); q('file').click(); });
  q('toast-webcam').addEventListener('click', () => { hideToast(); if (!state.webcam) q('webcam').click(); });
  q('upload').addEventListener('click', () => q('file').click());
  q('file').addEventListener('change', async (ev) => {
    const file = ev.target.files?.[0];
    if (file) await loadFile(file);
    ev.target.value = '';
  });
  stage.addEventListener('dragover', (ev) => ev.preventDefault());
  stage.addEventListener('drop', async (ev) => {
    ev.preventDefault();
    const file = [...(ev.dataTransfer?.files ?? [])].find((f) => f.type.startsWith('image/'));
    if (file) await loadFile(file);
  });
  async function loadFile(file) {
    stopWebcam();
    const bitmap = await createImageBitmap(file);
    setImage(bitmap);
    bitmap.close();
    state.onCard = false;
    hideToast();
  }

  q('webcam').addEventListener('click', async () => {
    if (state.webcam) {
      stopWebcam();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();
      state.webcam = { stream, video };
      state.onCard = false;
      hideToast();
      q('webcam').setAttribute('aria-pressed', 'true');
    } catch (err) {
      out('stats').textContent = `Webcam unavailable: ${err.message}`;
    }
  });
  function stopWebcam() {
    if (!state.webcam) return;
    state.webcam.stream.getTracks().forEach((t) => t.stop());
    state.webcam = null;
    q('webcam').setAttribute('aria-pressed', 'false');
  }

  autoResize(canvas, () => { state.dirty = true; });

  // Frame loop.
  const thetaPair = { buf: null };
  function writeUniforms() {
    const f = new Float32Array(12);
    const u = new Uint32Array(f.buffer);
    f[0] = canvas.width;
    f[1] = canvas.height;
    u[3] = q('mesh').checked ? FLAG_MESH : 0;
    u[4] = state.n;
    u[5] = state.n;
    u[6] = ARROWS;
    f[7] = 1;      // arrow length follows the displacement per unit time (eased, see vs_arrow)
    f[8] = Math.min(window.devicePixelRatio || 1, 2);
    device.queue.writeBuffer(uniforms, 0, f);
  }

  let prev = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - prev) / 1000);
    prev = now;
    if (!state.engine) {
      requestAnimationFrame(frame);
      return;
    }

    // The map holds one point per canvas pixel: rebuild it when the canvas is resized.
    if (state.pixels.w !== canvas.width || state.pixels.h !== canvas.height) {
      state.engine.destroyPointSet(state.pixels);
      Object.assign(state, makePixels(state.engine));
      updateStats();
      state.dirty = true;
    }

    if (state.playing) {
      state.playClock += dt * 1.4;
      setT(0.5 - 0.5 * Math.cos(state.playClock));
    }
    if (state.webcam && state.webcam.video.readyState >= 2) {
      drawCover(srcCanvas, state.webcam.video, { mirror: true });
      uploadCanvas(device, srcTexture, srcCanvas);
      state.dirty = true;
    }

    if (state.dirty) {
      state.dirty = false;
      const d = state.engine.d;
      if (!thetaPair.buf || thetaPair.buf.length !== 2 * d) thetaPair.buf = new Float32Array(2 * d);
      for (let i = 0; i < d; i++) {
        thetaPair.buf[i] = state.theta[i];
        thetaPair.buf[d + i] = -state.t * state.theta[i];
      }
      state.engine.writeTheta(thetaPair.buf);
      writeUniforms();

      const enc = device.createCommandEncoder();
      state.engine.encodeAffine(enc);
      state.engine.encodeExpm(enc);
      state.engine.encodeForward(enc, state.pixels);
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(imagePipeline);
      pass.setBindGroup(0, state.imageBG);
      pass.draw(3);
      if (q('arrows').checked) {
        pass.setPipeline(arrowPipeline);
        pass.setBindGroup(0, state.arrowBG);
        pass.draw(6, ARROWS * ARROWS);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    requestAnimationFrame(frame);
  }

  setImage(await testCard);
  out('radius').textContent = state.radius.toFixed(2);
  setT(1);
  await setTessellation(state.n);
  requestAnimationFrame(frame);
  return { state };
}
