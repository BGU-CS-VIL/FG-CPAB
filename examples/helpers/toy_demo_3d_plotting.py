import json

import numpy as np


def _generate_html(meshes, output_path, lighting=None):
    if lighting is None:
        lighting = {
            "ambient": {"color": 0x404060, "intensity": 0.6},
            "directional": [
                {"color": 0xFFFFFF, "intensity": 0.8, "position": [2.0, 3.0, 2.0]},
            ],
            "point": [
                {"color": 0xE94560, "intensity": 0.5, "distance": 5.0, "position": [0.5, 1.5, 0.5]},
            ],
        }

    mesh_data = []
    for mesh in meshes:
        mesh_data.append(
            {
                "name": mesh["name"],
                "vertices": mesh["vertices"].tolist(),
                "faces": mesh["faces"],
                "uv": mesh["uv"].tolist(),
            }
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D CPAB Mesh Deformation</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #1a1a2e;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e0e0e0;
    overflow: hidden;
  }}
  #header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    background: linear-gradient(135deg, #16213e, #0f3460);
    padding: 12px 24px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 20px rgba(0,0,0,0.5);
  }}
  #header h1 {{
    font-size: 18px; font-weight: 600;
    background: linear-gradient(90deg, #e94560, #f5a623);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  #header .controls {{
    display: flex; gap: 8px; align-items: center;
  }}
  .btn {{
    padding: 6px 16px; border: 1px solid #e94560; border-radius: 20px;
    background: transparent; color: #e94560; cursor: pointer;
    font-size: 13px; transition: all 0.2s;
  }}
  .btn:hover, .btn.active {{
    background: #e94560; color: white;
  }}
  #info {{
    position: fixed; bottom: 12px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 52, 96, 0.85); padding: 8px 20px; border-radius: 20px;
    font-size: 12px; color: #aaa; z-index: 100;
  }}
  #container {{ width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<div id="header">
  <h1>3D CPAB Mesh Deformation</h1>
  <div class="controls" id="buttons"></div>
</div>
<div id="container"></div>
<div id="info">Drag to rotate · Scroll to zoom · Right-drag to pan</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const MESHES = {json.dumps(mesh_data)};
const LIGHTING = {json.dumps(lighting)};

const container = document.getElementById('container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.FogExp2(0x1a1a2e, 0.3);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(1.8, 1.2, 1.8);
camera.lookAt(0.5, 0.5, 0.5);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0.5, 0.5, 0.5);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const ambientCfg = LIGHTING.ambient || {{}};
scene.add(new THREE.AmbientLight(
  ambientCfg.color !== undefined ? ambientCfg.color : 0x404060,
  ambientCfg.intensity !== undefined ? ambientCfg.intensity : 0.6
));

(LIGHTING.directional || []).forEach((cfg) => {{
  const light = new THREE.DirectionalLight(
    cfg.color !== undefined ? cfg.color : 0xffffff,
    cfg.intensity !== undefined ? cfg.intensity : 0.8
  );
  const pos = cfg.position || [2, 3, 2];
  light.position.set(pos[0], pos[1], pos[2]);
  scene.add(light);
}});

(LIGHTING.point || []).forEach((cfg) => {{
  const light = new THREE.PointLight(
    cfg.color !== undefined ? cfg.color : 0xe94560,
    cfg.intensity !== undefined ? cfg.intensity : 0.5,
    cfg.distance !== undefined ? cfg.distance : 5
  );
  const pos = cfg.position || [0.5, 1.5, 0.5];
  light.position.set(pos[0], pos[1], pos[2]);
  scene.add(light);
}});

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const boxEdges = new THREE.EdgesGeometry(boxGeo);
const boxLine = new THREE.LineSegments(
  boxEdges,
  new THREE.LineBasicMaterial({{ color: 0x445566, linewidth: 1 }})
);
boxLine.position.set(0.5, 0.5, 0.5);
scene.add(boxLine);

const meshObjects = [];
MESHES.forEach((mdata, idx) => {{
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(mdata.vertices.flat()), 3)
  );

  const colors = new Float32Array(mdata.vertices.length * 3);
  for (let i = 0; i < mdata.uv.length; i++) {{
    const color = new THREE.Color();
    color.setHSL(mdata.uv[i][0] * 0.85 + 0.05, 0.7, 0.35 + 0.35 * mdata.uv[i][1]);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }}
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(mdata.faces.flat()), 1));
  geometry.computeVertexNormals();

  const material = new THREE.MeshPhongMaterial({{
    vertexColors: true,
    side: THREE.DoubleSide,
    shininess: 60,
    specular: 0x222222,
  }});

  const mesh = new THREE.Mesh(geometry, material);
  mesh.visible = idx === 0;
  scene.add(mesh);
  meshObjects.push(mesh);
}});

const buttons = document.getElementById('buttons');
MESHES.forEach((mdata, idx) => {{
  const btn = document.createElement('button');
  btn.className = 'btn' + (idx === 0 ? ' active' : '');
  btn.textContent = mdata.name;
  btn.onclick = () => {{
    meshObjects.forEach((mesh, j) => mesh.visible = j === idx);
    document.querySelectorAll('.btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
  }};
  buttons.appendChild(btn);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


def save_mesh_viewer(verts, faces, uv, points_target, points_recovered, output_path):
    verts_np = verts.T
    target_np = points_target[0].detach().cpu().numpy().T
    recovered_np = points_recovered[0].detach().cpu().numpy().T

    n_v = verts_np.shape[0]
    offset = np.array([1.2, 0.0, 0.0], dtype=np.float32)
    v_combined = np.concatenate([target_np, recovered_np + offset], axis=0)
    f_combined = np.concatenate([np.array(faces), np.array(faces) + n_v], axis=0)
    uv_combined = np.concatenate([uv, uv], axis=0)

    meshes = [
        {"name": "Original", "vertices": verts_np, "faces": faces, "uv": uv},
        {"name": "Target (theta*)", "vertices": target_np, "faces": faces, "uv": uv},
        {"name": "Recovered (theta_hat)", "vertices": recovered_np, "faces": faces, "uv": uv},
        {
            "name": "Target vs Recovered",
            "vertices": v_combined,
            "faces": f_combined.tolist(),
            "uv": uv_combined,
        },
    ]

    _generate_html(meshes, output_path)
