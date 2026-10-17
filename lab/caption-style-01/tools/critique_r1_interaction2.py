"""Per caption: background luma range behind the letters and frame-to-frame change inside the letters (ours)."""
import sys, json
import numpy as np
from PIL import Image
S = sys.argv[1]; LAB = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01'
rep = json.load(open(sys.argv[2] if len(sys.argv) > 2 else LAB + '/out/user_v1.render.json'))
def zoomed_src(n, z):
    im = Image.open(f'{S}/src/s{n:03d}.png').convert('RGB')
    if abs(z - 1) < 1e-4: return np.asarray(im).astype(np.float32) / 255
    W, H = im.size
    # out(x) = in(c + (x - c)/z)
    a = 1 / z
    im2 = im.transform((W, H), Image.AFFINE, (a, 0, W/2 - a*W/2, 0, a, H/2 - a*H/2), resample=Image.BICUBIC)
    return np.asarray(im2).astype(np.float32) / 255
res = {}
prev = None
for fr in rep['frames']:
    n = fr['frame']
    if not fr['words']: continue
    o = np.asarray(Image.open(f'{S}/ours/o{n:03d}.png').convert('RGB')).astype(np.float32) / 255
    s = zoomed_src(fr['source_frame'], fr['zoom'])
    u = fr['words'][0]['id']
    blend = [x for x in rep['units'] if x['text'] == fr['words'][0]['text']][0]['blend']
    C = np.array([226, 139, 139]) / 255 if blend == 'difference_pink' else np.ones(3)
    model = np.abs(C - s)
    m = (np.abs(o - model).mean(2) < 0.05) & (np.abs(o - s).mean(2) > 0.2)
    if m.sum() < 500: continue
    L = (0.2126 * o[..., 0] + 0.7152 * o[..., 1] + 0.0722 * o[..., 2])[m]
    bgL = (0.2126 * s[..., 0] + 0.7152 * s[..., 1] + 0.0722 * s[..., 2])[m]
    rec = res.setdefault(fr['words'][0]['text'], {'frames': 0, 'bg_p5': [], 'bg_p95': [], 'L_mean': [], 'dL': []})
    rec['frames'] += 1; rec['bg_p5'].append(np.percentile(bgL, 5)); rec['bg_p95'].append(np.percentile(bgL, 95)); rec['L_mean'].append(L.mean())
    if prev is not None and prev[0] == u:
        mm = m & prev[2]
        if mm.sum() > 200: rec['dL'].append(np.abs(o - prev[1]).mean(2)[mm].mean() * 255)
    prev = (u, o, m)
for k, r in res.items():
    print(f'{k:10s} frames={r["frames"]:2d} bg luma p5..p95 = {np.mean(r["bg_p5"]):.3f}..{np.mean(r["bg_p95"]):.3f}  letter luma mean={np.mean(r["L_mean"]):.2f}  mean |frame-to-frame change| inside letters = {np.mean(r["dL"]) if r["dL"] else float("nan"):.1f} levels')
