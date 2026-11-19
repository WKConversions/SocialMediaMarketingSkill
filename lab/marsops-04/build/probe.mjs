import { chromium } from 'playwright';
const b = await chromium.launch({ args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
p.on('console', m => console.log(m.text())); p.on('pageerror', e => console.log('ERR', e.message));
await p.goto('http://127.0.0.1:8766/render5.html'); await p.waitForTimeout(240000); await b.close();
