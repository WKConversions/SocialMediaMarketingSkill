import { chromium } from 'playwright'; import fs from 'fs';
const b = await chromium.launch({ args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
fs.mkdirSync('demo', { recursive: true }); let next = 0; const N = 460;
const pages = await Promise.all([0,1,2,3].map(async () => { const p = await b.newPage({ viewport: { width: 1920, height: 1080 } }); await p.goto('http://127.0.0.1:8766/render5.html'); await p.evaluate(() => window.__ready); return p; }));
await Promise.all(pages.map(async p => { for (let k = next++; k < N; k = next++) { await p.evaluate(f => window.render(f), k * 3); await p.screenshot({ path: `demo/d${String(k).padStart(4,'0')}.jpg`, type: 'jpeg', quality: 85 }); } }));
await b.close();
