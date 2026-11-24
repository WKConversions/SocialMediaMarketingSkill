import { chromium } from 'playwright'; import fs from 'fs';
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1440, height: 900 } }); const reqs = [];
p.on('request', r => reqs.push(r.url()));
await p.goto(process.argv[2], { waitUntil: 'networkidle', timeout: 60000 }).catch(e => console.log(e.message)); await p.waitForTimeout(2000);
const h = await p.evaluate(() => document.documentElement.scrollHeight);
for (let y = 0, k = 0; y < h && k < 12; y += 900, k++) { await p.evaluate(v => scrollTo(0, v), y); await p.waitForTimeout(700); await p.screenshot({ path: `brand/s${String(k).padStart(2, '0')}.png` }); }
const info = await p.evaluate(() => { const c = {}, f = {}, bg = {}; for (const e of document.querySelectorAll('*')) { const s = getComputedStyle(e); c[s.color] = (c[s.color] || 0) + 1; f[s.fontFamily + '|' + s.fontWeight] = (f[s.fontFamily + '|' + s.fontWeight] || 0) + 1; if (s.backgroundColor !== 'rgba(0, 0, 0, 0)') bg[s.backgroundColor] = (bg[s.backgroundColor] || 0) + 1; }
  const vars = {}; for (const sh of document.styleSheets) { try { for (const r of sh.cssRules) if (r.selectorText === ':root') for (const n of r.style) if (n.startsWith('--')) vars[n] = r.style.getPropertyValue(n).trim(); } catch (e) {} }
  return { title: document.title, text: document.body.innerText.slice(0, 5000), c, f, bg, vars, media: [...document.querySelectorAll('img,svg,video')].map(e => (e.currentSrc || e.src || e.tagName) + ' ' + e.clientWidth + 'x' + e.clientHeight).slice(0, 40) }; });
fs.writeFileSync('brand/info.json', JSON.stringify({ h, info, reqs: [...new Set(reqs)].filter(u => !u.startsWith('data:')).slice(0, 80) }, null, 1)); await b.close();
