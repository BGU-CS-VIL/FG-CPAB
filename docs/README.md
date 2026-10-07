# FG-CPAB project page

`index.html` is the project page. It hosts three interactive WebGPU demos built on
`src/`, the WebGPU port of the CUDA kernels.

| Path | Contents |
|---|---|
| `src/` | CPAB engine: basis, matrix exponential, forward, FG-CPAB backward, projection |
| `demos/sculpt.js` | Hero demo: drag to sculpt a diffeomorphism of an image, webcam or upload |
| `demos/register.js` | Live registration: Adam on the factorized gradients (1–10 composed stages); Run trains until the loss stops improving, time slider to replay the fitted flow |
| `demos/gridfit.js` | Point correspondences: `examples/rectangular_grid_diffeomorphism_2d.py` (line grid fitted to a swirl, then inverted with −θ) |
| `demos/common.js` | Shared WebGPU, image, Adam and WGSL helpers |

## Run locally

ES modules need a web server, and WebGPU needs a secure page (https or localhost):

```bash
cd docs && python -m http.server 8000 --bind 127.0.0.1    # then open http://127.0.0.1:8000
```

Without `--bind`, Python announces the server as `http://[::]:8000`. Browsers hide
WebGPU on that address, so the demos show a "needs WebGPU" message; use
`http://localhost:8000` instead. For a server on another machine, forward the
port (`ssh -L 8000:localhost:8000 host`) and open it via localhost.

Add `?debug` to the URL to expose the demo state as `window.__demos`.
