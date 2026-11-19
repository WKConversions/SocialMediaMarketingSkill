// Building blocks for the v4 showcase teaser (three.js r170). Everything is driven by explicit time values (frame-pure).
import * as THREE from 'three';
import { MeshSurfaceSampler } from 'three/addons/math/MeshSurfaceSampler.js';

export const COL = { navy: 0x040b19, navy2: 0x07142b, ink: 0xf6f8ff, blue: 0x0c56ff, blue2: 0x2f73ff, blue3: 0x5b8fff, blue4: 0x8fb0ff, mars: 0xc8643c };

// a deterministic random stream
export function rng(seed = 1) { let s = seed | 0; return () => { s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
export const seg = (t, a, b) => clamp((t - a) / (b - a));
export const lerp = (a, b, u) => a + (b - a) * u;
export function bez(x1, y1, x2, y2) { return x => { if (x <= 0) return 0; if (x >= 1) return 1; let t = x;
  for (let i = 0; i < 7; i++) { const cx = 3 * x1 * t * (1 - t) ** 2 + 3 * x2 * t * t * (1 - t) + t ** 3 - x;
    const d = 3 * x1 * (1 - t) ** 2 + 6 * (x2 - x1) * t * (1 - t) + 3 * (1 - x2) * t * t; if (Math.abs(d) < 1e-6) break; t -= cx / d; }
  return 3 * y1 * t * (1 - t) ** 2 + 3 * y2 * t * t * (1 - t) + t ** 3; }; }
export const SIG = bez(.22, 1, .36, 1), INOUT = bez(.65, 0, .35, 1), EIN = bez(.55, 0, .9, .4);

// the sky: ESO's 360-degree Milky Way on an inside-out sphere, dimmed toward the site's navy
export function makeSky(tex, dim = 0.55) {
  tex.colorSpace = THREE.SRGBColorSpace; tex.mapping = THREE.EquirectangularReflectionMapping;
  const m = new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, color: new THREE.Color(dim, dim, dim * 1.08), depthWrite: false, toneMapped: false });
  const s = new THREE.Mesh(new THREE.SphereGeometry(60, 64, 32), m); s.renderOrder = -10; return s;
}

// a thin atmosphere: a fresnel shell, additive
export function atmosphere(radius, color = new THREE.Color(1.0, 0.45, 0.25), power = 3.0, strength = 0.9) {
  const m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.BackSide,
    uniforms: { uColor: { value: color }, uPow: { value: power }, uStr: { value: strength }, uOpacity: { value: 1 } },
    vertexShader: `varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uPow, uStr, uOpacity; varying vec3 vN; varying vec3 vV;
      void main(){ float f = pow(clamp(1.0 + dot(vN, vV), 0.0, 1.0), uPow); gl_FragColor = vec4(uColor * f * uStr * uOpacity, 1.0); }` });
  return new THREE.Mesh(new THREE.SphereGeometry(radius, 96, 48), m);
}

// the hologram look: fresnel rim, scanlines that travel, a slow flicker, a reveal that sweeps along an axis
export function holoMaterial(opts = {}) {
  return new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color(opts.color ?? 0x5b8fff) }, uOpacity: { value: 1 },
      uReveal: { value: 1 }, uY0: { value: opts.y0 ?? -1 }, uY1: { value: opts.y1 ?? 1 }, uScan: { value: opts.scan ?? 90 }, uFill: { value: opts.fill ?? 0.10 } },
    vertexShader: `varying vec3 vN; varying vec3 vV; varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.0); vW = w.xyz;
      vec4 mv = viewMatrix*w; vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform float uTime, uOpacity, uReveal, uY0, uY1, uScan, uFill; uniform vec3 uColor; varying vec3 vN; varying vec3 vV; varying vec3 vW;
      float h(float n){ return fract(sin(n)*43758.5453); }
      void main(){
        float y = (vW.y - uY0) / (uY1 - uY0);
        float edge = uReveal * 1.08 - 0.04;
        if (y > edge) discard;
        float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
        float scan = 0.55 + 0.45 * sin((vW.y * uScan - uTime * 3.0) * 6.2831);
        float band = smoothstep(0.035, 0.0, abs(y - edge)) * 1.6;                 // the bright reveal line
        float flick = 0.92 + 0.08 * h(floor(uTime * 24.0));
        float a = (uFill + fres * 0.9) * scan * flick + band;
        gl_FragColor = vec4(uColor * a * uOpacity, 1.0);
      }` });
}

// the shield: a sphere with a hexagonal cell pattern, a fresnel rim, and ripples where it is hit (up to 12 hits)
export function shieldMaterial() {
  const hits = []; for (let i = 0; i < 12; i++) hits.push(new THREE.Vector4(0, 0, 1, -99));
  return new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color(0x2f73ff) }, uHits: { value: hits }, uOpacity: { value: 1 }, uReveal: { value: 1 }, uBase: { value: 0.05 } },
    vertexShader: `varying vec3 vN; varying vec3 vV; varying vec3 vP; void main(){ vP = normalize(position); vec4 mv = modelViewMatrix*vec4(position,1.0);
      vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform float uTime, uOpacity, uReveal, uBase; uniform vec3 uColor; uniform vec4 uHits[12]; varying vec3 vN; varying vec3 vV; varying vec3 vP;
      vec2 hexC(vec2 uv){ const vec2 S = vec2(1.0, 1.7320508); vec2 a = mod(uv, S) - S * 0.5; vec2 b = mod(uv - S * 0.5, S) - S * 0.5; return dot(a, a) < dot(b, b) ? a : b; }
      float hexD(vec2 p){ p = abs(p); return max(dot(p, vec2(0.5, 0.8660254)), p.x); }
      void main(){
        float lat = asin(clamp(vP.y, -1.0, 1.0)), lon = atan(vP.z, vP.x);
        vec2 uv = vec2(lon * 7.0, lat * 7.0 * 1.1);
        float cell = smoothstep(0.43, 0.49, hexD(hexC(uv)));                       // cell borders
        float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.5);
        float hit = 0.0;
        for (int i = 0; i < 12; i++) { float age = uTime - uHits[i].w; if (age < 0.0 || age > 1.2) continue;
          float ang = acos(clamp(dot(vP, normalize(uHits[i].xyz)), -1.0, 1.0));
          float ring = smoothstep(0.10, 0.0, abs(ang - age * 0.9)) * (1.0 - age / 1.2);
          float core = smoothstep(0.35, 0.0, ang) * exp(-age * 5.0);
          hit += ring * (0.6 + cell * 1.4) + core * 1.2; }
        float sweep = smoothstep(uReveal * 2.4 - 1.2 + 0.08, uReveal * 2.4 - 1.2, vP.y);   // the bubble builds from the bottom
        float a = (uBase * cell + fres * (0.10 + cell * 0.40) + hit * 0.8) * sweep;
        gl_FragColor = vec4(uColor * a * uOpacity + vec3(0.6, 0.8, 1.0) * hit * 0.25 * uOpacity, 1.0);
      }` });
}

// points on the surface of a mesh (or meshes), as a Float32Array of xyz
export function sampleMesh(object, n, seed = 3) {
  const r = rng(seed), out = new Float32Array(n * 3), meshes = [];
  object.updateMatrixWorld(true); object.traverse(o => { if (o.isMesh) meshes.push(o); });
  const areas = meshes.map(m => { const g = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry; let a = 0; const p = g.attributes.position, v = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    for (let i = 0; i < p.count; i += 3) { v[0].fromBufferAttribute(p, i); v[1].fromBufferAttribute(p, i + 1); v[2].fromBufferAttribute(p, i + 2); a += v[1].sub(v[0]).cross(v[2].sub(v[0])).length() / 2; } return a; });
  const tot = areas.reduce((a, b) => a + b, 0), v = new THREE.Vector3(); let k = 0;
  meshes.forEach((m, i) => { const cnt = i === meshes.length - 1 ? n - k : Math.round(n * areas[i] / tot); if (cnt <= 0) return;
    const s = new MeshSurfaceSampler(m).setRandomGenerator(r).build();
    for (let j = 0; j < cnt && k < n; j++, k++) { s.sample(v); v.applyMatrix4(m.matrixWorld); out[k * 3] = v.x; out[k * 3 + 1] = v.y; out[k * 3 + 2] = v.z; } });
  return out;
}
// points inside the letters of a text (centred, width w in scene units, in the z = 0 plane)
export function textPoints(text, font, n, w, seed = 5) {
  const c = document.createElement('canvas'); c.width = 1400; c.height = 300; const g = c.getContext('2d');
  g.fillStyle = '#fff'; g.font = font; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.letterSpacing = '0.1em'; g.fillText(text, 700 + 0.05 * parseFloat(font.match(/(\d+)px/)[1]), 150);
  const d = g.getImageData(0, 0, 1400, 300).data, ins = [];
  let x0 = 1400, x1 = 0; for (let y = 0; y < 300; y += 1) for (let x = 0; x < 1400; x += 1) if (d[(y * 1400 + x) * 4 + 3] > 128) { ins.push([x, y]); x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
  const r = rng(seed), out = new Float32Array(n * 3), s = w / (x1 - x0), cx = (x0 + x1) / 2;
  for (let i = 0; i < n; i++) { const p = ins[Math.floor(r() * ins.length)]; out[i * 3] = (p[0] - cx + r() - 0.5) * s; out[i * 3 + 1] = -(p[1] - 150 + r() - 0.5) * s; out[i * 3 + 2] = 0; }
  return out;
}
// a point cloud that morphs between up to four shapes; per-point delay, curl-ish drift while in flight
export function morphPoints(shapes, n, color = 0x8fb0ff, size = 3.0) {
  const g = new THREE.BufferGeometry(), r = rng(9);
  g.setAttribute('position', new THREE.BufferAttribute(shapes[0].slice(), 3));
  shapes.forEach((s, i) => g.setAttribute('aS' + i, new THREE.BufferAttribute(s, 3)));
  const rnd = new Float32Array(n * 4); for (let i = 0; i < n * 4; i++) rnd[i] = r(); g.setAttribute('aRnd', new THREE.BufferAttribute(rnd, 4));
  const m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uA: { value: 0 }, uB: { value: 1 }, uU: { value: 0 }, uTime: { value: 0 }, uColor: { value: new THREE.Color(color) }, uSize: { value: size }, uOpacity: { value: 1 }, uSpread: { value: 0.6 }, uDrift: { value: 0.02 } },
    vertexShader: `attribute vec3 aS0; attribute vec3 aS1; attribute vec3 aS2; attribute vec3 aS3; attribute vec4 aRnd; uniform float uA, uB, uU, uTime, uSize, uSpread, uDrift; varying float vA;
      vec3 shp(float k){ return k < 0.5 ? aS0 : k < 1.5 ? aS1 : k < 2.5 ? aS2 : aS3; }
      void main(){ float d = aRnd.x * 0.45; float u = clamp((uU - d) / 0.55, 0.0, 1.0); u = u * u * (3.0 - 2.0 * u);
        vec3 a = shp(uA), b = shp(uB); vec3 p = mix(a, b, u);
        float lift = sin(3.14159 * u); p += (aRnd.yzw - 0.5) * uSpread * lift * 2.0 + vec3(sin(uTime * 1.3 + aRnd.y * 6.28), cos(uTime * 1.1 + aRnd.z * 6.28), sin(uTime * 0.9 + aRnd.w * 6.28)) * uDrift;
        vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = uSize * (0.6 + aRnd.w * 0.8) * (6.0 / -mv.z); vA = 0.55 + 0.45 * aRnd.y; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; varying float vA; void main(){ vec2 c = gl_PointCoord - 0.5; float r = length(c); if (r > 0.5) discard;
      float a = smoothstep(0.5, 0.0, r); gl_FragColor = vec4(uColor * a * vA * uOpacity, 1.0); }` });
  return new THREE.Points(g, m);
}

// solar wind: streaks flowing along +x through a box; near a protected sphere they are pushed around it
export function solarWind(n, box, seed = 11) {
  const r = rng(seed), seeds = []; for (let i = 0; i < n; i++) seeds.push({ y: lerp(box.y0, box.y1, r()), z: lerp(box.z0, box.z1, r()), ph: r(), v: 0.7 + r() * 0.6, len: 0.25 + r() * 0.5 });
  const g = new THREE.BufferGeometry(); const pos = new Float32Array(n * 6), col = new Float32Array(n * 6);
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const m = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  const lines = new THREE.LineSegments(g, m); lines.frustumCulled = false;
  lines.userData.update = (t, speed, shield /* {c: Vector3, r, on} */, warm = [1.0, 0.86, 0.72], opacity = 1) => {
    const L = box.x1 - box.x0, P = new THREE.Vector3();
    for (let i = 0; i < n; i++) { const s = seeds[i]; let x = box.x0 + ((s.ph * L + t * speed * s.v) % L);
      const pts = [[x - s.len, s.y, s.z], [x, s.y, s.z]];
      for (let k = 0; k < 2; k++) { P.set(...pts[k]);
        if (shield && shield.on > 0) { const d = P.clone().sub(shield.c), dl = d.length(), R = shield.r * 1.04;
          if (dl < R * 1.6) { const push = clamp((R * 1.6 - dl) / (R * 0.6)) ** 2 * shield.on; const out = d.normalize().multiplyScalar(lerp(dl, Math.max(dl, R), push) - dl); P.add(out); } }
        pos[i * 6 + k * 3] = P.x; pos[i * 6 + k * 3 + 1] = P.y; pos[i * 6 + k * 3 + 2] = P.z;
        const a = (k === 0 ? 0.0 : 1.0) * opacity * clamp((x - box.x0) / 0.6) * clamp((box.x1 - x) / 0.6);
        col[i * 6 + k * 3] = warm[0] * a; col[i * 6 + k * 3 + 1] = warm[1] * a; col[i * 6 + k * 3 + 2] = warm[2] * a; } }
    g.attributes.position.needsUpdate = true; g.attributes.color.needsUpdate = true; };
  return lines;
}

// a station in the spirit of the site's render: modules, a node, solar wings; as white clay or as a hologram
export function station(mat, edgeMat) {
  const grp = new THREE.Group(), add = (geo, x, y, z, rx = 0, ry = 0, rz = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.rotation.set(rx, ry, rz); grp.add(m);
    if (edgeMat) { const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 25), edgeMat); e.position.copy(m.position); e.rotation.copy(m.rotation); grp.add(e); } return m; };
  const mod = new THREE.CylinderGeometry(0.32, 0.32, 1.5, 32, 4), node = new THREE.SphereGeometry(0.42, 32, 16), ring = new THREE.TorusGeometry(0.34, 0.035, 8, 32);
  add(node, 0, 0, 0); add(mod, 0, 1.15, 0); add(mod, 0, -1.15, 0); add(mod, 1.15, 0, 0, 0, 0, Math.PI / 2); add(mod, -1.15, 0, 0, 0, 0, Math.PI / 2);
  [[0, 1.9, 0, Math.PI / 2, 0, 0], [0, -1.9, 0, Math.PI / 2, 0, 0], [1.9, 0, 0, 0, Math.PI / 2, 0], [-1.9, 0, 0, 0, Math.PI / 2, 0]].forEach(a => add(ring, ...a));
  const panel = new THREE.BoxGeometry(1.5, 0.02, 0.62, 6, 1, 3), truss = new THREE.CylinderGeometry(0.03, 0.03, 1.1, 6);
  [[1, 1], [-1, 1], [1, -1], [-1, -1]].forEach(([sx, sy]) => { add(truss, sx * 0.55, sy * 1.15, 0, 0, 0, Math.PI / 2); add(panel, sx * 1.75, sy * 1.15, 0, 0.35 * sx, 0, 0); });
  add(new THREE.CylinderGeometry(0.02, 0.02, 0.6, 6), 0, 2.6, 0); add(new THREE.SphereGeometry(0.06, 12, 8), 0, 2.92, 0);
  return grp;
}

// a procedural astronaut (fallback when no model is available): capsules and spheres, standing, ~1.8 units tall
export function astronaut(mat, edgeMat) {
  const g = new THREE.Group(), parts = [];
  const add = (geo, x, y, z, rx = 0, ry = 0, rz = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.rotation.set(rx, ry, rz); g.add(m); parts.push(m);
    if (edgeMat) { const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), edgeMat); e.position.copy(m.position); e.rotation.copy(m.rotation); g.add(e); } return m; };
  add(new THREE.SphereGeometry(0.17, 32, 20), 0, 1.62, 0);                                   // helmet
  add(new THREE.SphereGeometry(0.12, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.45), 0, 1.63, 0.07, Math.PI / 2.4, 0, 0);   // visor
  add(new THREE.CapsuleGeometry(0.23, 0.38, 8, 20), 0, 1.18, 0);                             // torso
  add(new THREE.BoxGeometry(0.42, 0.52, 0.2, 2, 2, 1), 0, 1.24, -0.24);                       // backpack
  add(new THREE.CapsuleGeometry(0.075, 0.42, 6, 12), 0.33, 1.12, 0.02, 0, 0, 0.22);           // arms
  add(new THREE.CapsuleGeometry(0.075, 0.42, 6, 12), -0.33, 1.12, 0.02, 0, 0, -0.22);
  add(new THREE.SphereGeometry(0.075, 12, 8), 0.43, 0.82, 0.03); add(new THREE.SphereGeometry(0.075, 12, 8), -0.43, 0.82, 0.03);   // gloves
  add(new THREE.CapsuleGeometry(0.21, 0.12, 6, 16), 0, 0.82, 0);                              // hips
  add(new THREE.CapsuleGeometry(0.09, 0.5, 6, 12), 0.11, 0.42, 0); add(new THREE.CapsuleGeometry(0.09, 0.5, 6, 12), -0.11, 0.42, 0);   // legs
  add(new THREE.BoxGeometry(0.15, 0.09, 0.26, 1, 1, 2), 0.11, 0.05, 0.04); add(new THREE.BoxGeometry(0.15, 0.09, 0.26, 1, 1, 2), -0.11, 0.05, 0.04);   // boots
  g.userData.parts = parts; return g;
}

// camera keys: [{t, p:[x,y,z], q:[x,y,z] (look-at), fov}], eased between keys with a per-key curve
export function cameraAt(keys, t) {
  let i = 0; while (i < keys.length - 2 && t > keys[i + 1].t) i++;
  const a = keys[i], b = keys[i + 1], u = (b.ease || INOUT)(seg(t, a.t, b.t));
  const L3 = (x, y) => [lerp(x[0], y[0], u), lerp(x[1], y[1], u), lerp(x[2], y[2], u)];
  return { p: L3(a.p, b.p), q: L3(a.q, b.q), fov: lerp(a.fov ?? 32, b.fov ?? 32, u), roll: lerp(a.roll ?? 0, b.roll ?? 0, u) };
}

// points that start as a photo (each point carries its pixel's colour, on a plane) and morph into a shape (blue)
export function photoPoints(img, n, plane /* {o: Vector3, right: Vector3, up: Vector3, w, h} */, target, seed = 13) {
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height; const g = c.getContext('2d'); g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data, r = rng(seed);
  const pos0 = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const x = Math.floor(r() * c.width), y = Math.floor(r() * c.height), k = (y * c.width + x) * 4;
    const u = x / c.width - 0.5, v = 0.5 - y / c.height, lum = (d[k] * 0.3 + d[k + 1] * 0.59 + d[k + 2] * 0.11) / 255;
    const p = plane.o.clone().add(plane.right.clone().multiplyScalar(u * plane.w)).add(plane.up.clone().multiplyScalar(v * plane.h));
    pos0[i * 3] = p.x; pos0[i * 3 + 1] = p.y; pos0[i * 3 + 2] = p.z + (lum - 0.5) * 0.04;
    col[i * 3] = Math.pow(d[k] / 255, 2.2); col[i * 3 + 1] = Math.pow(d[k + 1] / 255, 2.2); col[i * 3 + 2] = Math.pow(d[k + 2] / 255, 2.2); }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos0.slice(), 3));
  geo.setAttribute('aS0', new THREE.BufferAttribute(pos0, 3)); geo.setAttribute('aS1', new THREE.BufferAttribute(target, 3)); geo.setAttribute('aCol', new THREE.BufferAttribute(col, 3));
  const rnd = new Float32Array(n * 4); for (let i = 0; i < n * 4; i++) rnd[i] = r(); geo.setAttribute('aRnd', new THREE.BufferAttribute(rnd, 4));
  const m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uU: { value: 0 }, uTime: { value: 0 }, uColor: { value: new THREE.Color(0x8fb0ff) }, uSize: { value: 3.0 }, uOpacity: { value: 1 }, uSpread: { value: 0.5 }, uLift: { value: 0 } },
    vertexShader: `attribute vec3 aS0; attribute vec3 aS1; attribute vec3 aCol; attribute vec4 aRnd; uniform float uU, uTime, uSize, uSpread, uLift; varying vec3 vC; varying float vU;
      void main(){ float d = (1.0 - aRnd.x) * 0.35 + aRnd.y * 0.15; float u = clamp((uU - d) / 0.5, 0.0, 1.0); u = u * u * (3.0 - 2.0 * u); vU = u;
        vec3 p = mix(aS0, aS1, u); p += (aRnd.yzw - 0.5) * uSpread * sin(3.14159 * u) * 2.0; p.z += uLift * (aRnd.w - 0.3) * 0.25 * (1.0 - u);
        vC = aCol; vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = uSize * (0.7 + aRnd.w * 0.6) * (6.0 / -mv.z); }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; varying vec3 vC; varying float vU; void main(){ vec2 c = gl_PointCoord - 0.5; float r = length(c); if (r > 0.5) discard;
      float a = smoothstep(0.5, 0.05, r); vec3 col = mix(vC * 1.2, uColor, smoothstep(0.15, 0.7, vU)); gl_FragColor = vec4(col * a * uOpacity, 1.0); }` });
  const pts = new THREE.Points(geo, m); pts.frustumCulled = false; return pts;
}

// make a standard material disappear above a world-space height (with a glowing cut line): for "real -> hologram" scans
export function scanCut(material, uniforms) {
  material.onBeforeCompile = sh => {
    sh.uniforms.uCutY = uniforms.uCutY; sh.uniforms.uCutW = uniforms.uCutW;
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vWPos;').replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vWPos; uniform float uCutY; uniform float uCutW;')
      .replace('#include <dithering_fragment>', '#include <dithering_fragment>\nif (vWPos.y > uCutY) discard;\nfloat e = smoothstep(uCutW, 0.0, uCutY - vWPos.y);\ngl_FragColor.rgb += vec3(0.35, 0.6, 1.0) * e * 3.0;');
  };
  material.needsUpdate = true;
}

// a holographic globe: latitude/longitude graticule lines + fresnel shell, revealed above a world-space height
export function holoGlobe(radius, color = 0x5b8fff) {
  const g = new THREE.Group(), mat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uColor: { value: new THREE.Color(color) }, uCutY: { value: -99 }, uOpacity: { value: 1 } },
    vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uCutY, uOpacity; varying vec3 vW; void main(){ if (vW.y < uCutY) discard; float e = smoothstep(0.06, 0.0, vW.y - uCutY);
      gl_FragColor = vec4(uColor * (0.55 + e * 2.0) * uOpacity, 1.0); }` });
  const lines = [];
  for (let la = -75; la <= 75; la += 15) { const y = Math.sin(la * Math.PI / 180) * radius, rr = Math.cos(la * Math.PI / 180) * radius, p = [];
    for (let k = 0; k <= 96; k++) { const a = k / 96 * Math.PI * 2; p.push(rr * Math.cos(a), y, rr * Math.sin(a)); } lines.push(p); }
  for (let lo = 0; lo < 180; lo += 15) { const p = []; for (let k = 0; k <= 96; k++) { const a = k / 96 * Math.PI * 2; const x = Math.cos(a) * radius, y = Math.sin(a) * radius;
    p.push(x * Math.cos(lo * Math.PI / 180), y, x * Math.sin(lo * Math.PI / 180)); } lines.push(p); }
  for (const p of lines) { const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(p, 3)); g.add(new THREE.Line(geo, mat)); }
  const shell = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.995, 64, 32), holoMaterial({ y0: -radius, y1: radius, scan: 40, fill: 0.04 }));
  g.add(shell); g.userData = { mat, shell }; return g;
}
