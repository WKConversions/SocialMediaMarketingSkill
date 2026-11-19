// ------------------------------------------------------------------ helpers
let seed = 20261119;
function rnd() { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
function bez(x1, y1, x2, y2) {                  // cubic-bezier easing, solved by Newton
  return x => { if (x <= 0) return 0; if (x >= 1) return 1; let t = x;
    for (let i = 0; i < 6; i++) { const cx = 3 * x1 * t * (1 - t) ** 2 + 3 * x2 * t * t * (1 - t) + t ** 3 - x;
      const d = 3 * x1 * (1 - t) ** 2 + 6 * (x2 - x1) * t * (1 - t) + 3 * (1 - x2) * t * t; if (Math.abs(d) < 1e-6) break; t -= cx / d; }
    return 3 * y1 * t * (1 - t) ** 2 + 3 * y2 * t * t * (1 - t) + t ** 3; };
}
const SIG = bez(.22, 1, .36, 1);                 // the film's signature curve (arrive)
const PEN = bez(.45, .05, .35, 1);               // a hand: accelerates, then slows into the end of the line
const FLAT = bez(.6, 0, .25, 1);                 // the dome's shield flattening into the line: accelerates into the 12.0 s hit
const lerp = (a, b, u) => a + (b - a) * u;
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || CAM).appendChild(e); return e; }
const CAM = document.getElementById('cam');
function grp(parent, attrs = {}) { return el('g', attrs, parent); }

function resample(pts, step = 5) {               // even spacing along the polyline
  const out = [pts[0]]; let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; const L = Math.hypot(x1 - x0, y1 - y0); let d = step - carry;
    while (d <= L) { const u = d / L; out.push([x0 + (x1 - x0) * u, y0 + (y1 - y0) * u]); d += step; }
    carry = L - (d - step);
  }
  out.push(pts[pts.length - 1]); return out;
}
function wobble(pts, amp = 1.1) {                // a hand's low-frequency wobble along the line
  const p = resample(pts, 5), ph = [rnd() * 6.3, rnd() * 6.3], fr = [0.011 + rnd() * 0.01, 0.03 + rnd() * 0.02];
  let s = 0;
  return p.map((q, i) => { if (i) s += Math.hypot(q[0] - p[i - 1][0], q[1] - p[i - 1][1]);
    const a = i ? Math.atan2(q[1] - p[i - 1][1], q[0] - p[i - 1][0]) : 0;
    const off = amp * (Math.sin(s * fr[0] + ph[0]) + 0.5 * Math.sin(s * fr[1] + ph[1]));
    return [q[0] - Math.sin(a) * off, q[1] + Math.cos(a) * off]; });
}
function overshoot(pts, e0 = 6, e1 = 9) {        // sketch lines run a little past their ends
  if (pts.length < 2) return pts; const [a, b] = [pts[0], pts[1]], [c, d] = [pts[pts.length - 2], pts[pts.length - 1]];
  const la = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, lc = Math.hypot(d[0] - c[0], d[1] - c[1]) || 1;
  return [[a[0] - (b[0] - a[0]) / la * e0, a[1] - (b[1] - a[1]) / la * e0], ...pts, [d[0] + (d[0] - c[0]) / lc * e1, d[1] + (d[1] - c[1]) / lc * e1]];
}
const dstr = pts => 'M' + pts.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('L');
function arc(cx, cy, rx, ry, a0, a1, n = 90, rot = 0) {
  const out = []; for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n;
    const x = rx * Math.cos(a), y = ry * Math.sin(a);
    out.push([cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]); } return out;
}
const line = (x0, y0, x1, y1) => [[x0, y0], [x1, y1]];
function catmull(P, closed = true, n = 10) {
  const out = [], L = P.length, g = i => P[closed ? (i + L) % L : Math.max(0, Math.min(L - 1, i))];
  for (let i = 0; i < (closed ? L : L - 1); i++) { const p0 = g(i - 1), p1 = g(i), p2 = g(i + 1), p3 = g(i + 2);
    for (let k = 0; k < n; k++) { const t = k / n, t2 = t * t, t3 = t2 * t;
      out.push([0, 1].map(j => 0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3))); } }
  if (closed) out.push(out[0]); else out.push(P[L - 1]); return out;
}

// ------------------------------------------------------------------ strokes (drawn by dash offset)
const STROKES = [];
function stroke(parent, pts, t0, dur, o = {}) {
  const w = (o.w ?? 2.2) * 1.42 / K, op = o.op ?? 0.9, col = o.col ?? INK;   // page widths (the camera scales by ~K)
  const base = o.raw ? pts : wobble(o.noOver ? pts : overshoot(pts, o.e0 ?? 5, o.e1 ?? 8), o.amp ?? 1.1);
  const p = el('path', { d: dstr(base), fill: 'none', stroke: col, 'stroke-width': w, 'stroke-linecap': 'round',
    'stroke-linejoin': 'round', opacity: op }, parent);
  STROKES.push({ p, t0, dur, ease: o.ease || PEN });
  if (o.pass !== false && !o.raw) {               // the second, lighter pass a draughtsman makes
    const q = el('path', { d: dstr(wobble(overshoot(pts, o.e1 ?? 8, o.e0 ?? 5), (o.amp ?? 1.1) * 1.4)), fill: 'none', stroke: col,
      'stroke-width': w * 0.6, 'stroke-linecap': 'round', opacity: op * 0.45 }, parent);
    STROKES.push({ p: q, t0: t0 + dur * 0.35, dur: dur * 0.9, ease: PEN });
  }
  return p;
}
function hatch(parent, inside, ang, gap, box, t0, dur, o = {}) {   // hatching revealed by a hand's sweep (a mask)
  const d = [], ca = Math.cos(ang), sa = Math.sin(ang), [x0, y0, x1, y1] = box, R = Math.hypot(x1 - x0, y1 - y0);
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  for (let k = -R; k <= R; k += gap) {
    let run = null;
    for (let s = -R; s <= R; s += 3) {
      const x = mx + ca * s - sa * k + (rnd() - 0.5) * 0.6, y = my + sa * s + ca * k;
      const ok = x > x0 && x < x1 && y > y0 && y < y1 && inside(x, y);
      if (ok && !run) run = [[x, y]]; else if (ok) run[1] = [x, y];
      else if (run) { if (run[1] && Math.hypot(run[1][0] - run[0][0], run[1][1] - run[0][1]) > 4) d.push(run); run = null; }
    }
    if (run && run[1]) d.push(run);
  }
  // cut each run into short strokes with small gaps (a hand hatching in bundles)
  const cut = [];
  for (const [a, b] of d) { const L = Math.hypot(b[0] - a[0], b[1] - a[1]); let u = 0;
    while (u < L) { const len = 24 + rnd() * 46, gap = 3 + rnd() * 9, u1 = Math.min(L, u + len);
      const j = (rnd() - 0.5) * 1.6;
      cut.push([[a[0] + (b[0] - a[0]) * u / L + j, a[1] + (b[1] - a[1]) * u / L], [a[0] + (b[0] - a[0]) * u1 / L - j, a[1] + (b[1] - a[1]) * u1 / L]]);
      u = u1 + gap; } }
  d.length = 0; d.push(...cut);
  const id = 'hm' + STROKES.length + '_' + Math.floor(rnd() * 1e6);
  const m = el('mask', { id, maskUnits: 'userSpaceOnUse', x: -200, y: -200, width: W + 400, height: H + 400 }, parent);
  const r = el('rect', { x: x1, y: -200, width: 0, height: H + 400, fill: '#fff' }, m);
  const g = grp(parent, { mask: `url(#${id})` });
  el('path', { d: d.map(s => 'M' + s[0][0].toFixed(1) + ' ' + s[0][1].toFixed(1) + 'L' + s[1][0].toFixed(1) + ' ' + s[1][1].toFixed(1)).join(''),
    stroke: o.col ?? INK, 'stroke-width': (o.w ?? 1.2) / K, opacity: o.op ?? 0.5, 'stroke-linecap': 'round', fill: 'none' }, g);
  STROKES.push({ sweep: r, x0, x1, t0, dur, dir: o.dir ?? -1 });
}
const NOTES = [];
function note(parent, txt, x, y, t0, dur, size = 34, rot = 0, op = 0.6) {   // mirror script, written right to left
  const id = 'nm' + NOTES.length;
  const m = el('mask', { id, maskUnits: 'userSpaceOnUse', x: -200, y: -200, width: W + 400, height: H + 400 }, parent);
  const r = el('rect', { x: x, y: y - size * 1.4, width: 0, height: size * 2.2, fill: '#fff' }, m);
  const g = grp(parent, { mask: `url(#${id})` });
  const t = el('text', { x: 0, y: 0, 'font-family': 'Leo', 'font-size': size, fill: INK, opacity: op,
    transform: `translate(${x},${y}) rotate(${rot}) scale(-1,1)` }, g);
  t.textContent = txt;
  NOTES.push({ r, t, x, t0, dur, w: 0 });
}
const pre = (px, py) => [CX + (px - CX) / K, CY + (py - CY) / K];   // page position -> drawing coordinates
function noteP(parent, txt, px, py, t0, dur, size, rot, op) { const [x, y] = pre(px, py); note(parent, txt, x, y, t0, dur, size / K, rot, op); }
const ENV = [];                                    // group opacity envelopes: [g, in0, in1, out0, out1]
function env(g, a, b, c = 99, d = 100) { ENV.push([g, a, b, c, d]); return g; }

