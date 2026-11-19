// Frame renderer for a render page (see render.html and production/coded-render.md).
// node render.mjs test 0,45,120      -> test/f0000.png ... (chosen frames, PNG)
// node render.mjs full 0 899 4 8     -> frames/f0000.jpg ... (4 parallel pages, 8 motion-blur samples; 1 = no blur)
// Env: PAGE (default render.html), SIZE (default 1080x1920). Needs playwright and sharp in this folder.
import { chromium } from 'playwright';
import sharp from 'sharp';
import fs from 'fs';
const [, , mode, a, b, workers = '4', samples = '8'] = process.argv;
const [W, H] = (process.env.SIZE || '1080x1920').split('x').map(Number);
const S = +samples, SHUTTER = 0.5;                  // sub-frames per frame, 180° shutter
const url = 'file://' + process.cwd() + '/' + (process.env.PAGE || 'render.html');
const browser = await chromium.launch({ args: ['--allow-file-access-from-files'] });
async function open() {
  const p = await browser.newPage({ viewport: { width: W, height: H } });
  p.on('pageerror', e => console.log('PAGEERR', e.message));
  await p.goto(url); await p.evaluate(() => window.__ready); return p;
}
async function shot(p, f, dir, jpg) {                // averages S renders spread across the shutter: motion blur
  const acc = new Float32Array(W * H * 3);
  for (let k = 0; k < S; k++) {
    await p.evaluate(t => window.render(t), S === 1 ? f : f + ((k + 0.5) / S - 0.5) * SHUTTER);
    const img = await p.screenshot({ type: 'jpeg', quality: 100, clip: { x: 0, y: 0, width: W, height: H } });
    const { data } = await sharp(img).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    for (let i = 0; i < data.length; i++) acc[i] += data[i];
  }
  const out = Buffer.alloc(W * H * 3); for (let i = 0; i < out.length; i++) out[i] = Math.round(acc[i] / S);
  const img = sharp(out, { raw: { width: W, height: H, channels: 3 } });
  await (jpg ? img.jpeg({ quality: 94 }) : img.png()).toFile(`${dir}/f${String(f).padStart(4, '0')}.${jpg ? 'jpg' : 'png'}`);
}
if (mode === 'test') {
  fs.mkdirSync('test', { recursive: true }); const p = await open();
  for (const f of a.split(',').map(Number)) await shot(p, f, 'test', false);
} else {
  fs.mkdirSync('frames', { recursive: true }); let next = +a;
  const pages = await Promise.all(Array.from({ length: +workers }, open));
  await Promise.all(pages.map(async p => { for (let f = next++; f <= +b; f = next++) await shot(p, f, 'frames', true); }));
}
await browser.close();
