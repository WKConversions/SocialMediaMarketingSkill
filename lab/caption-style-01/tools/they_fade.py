"""Opacity + blur entrance of the stacked 'They / don't / wanna' block.
Plate recovered from the sharp final frame f098 (inside text: B = 1-I, outside: B = I),
warped to each earlier frame with the measured zoom chain; then fit A = alpha * G_sigma(mask)."""
import numpy as np, cv2, json
from scipy import ndimage as ndi
from prime import sim_matrix, F
from common import W, H
C = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
ref = 98
I98 = F(ref)
masks = {
 'They': np.load(f'{C}/mask_They_f098.npy').astype(np.float32),
 "don't": (np.load(f'{C}/rend_dont.npy').astype(np.float32) > 0.5).astype(np.float32),
 'wanna': (np.load(f'{C}/rend_wanna.npy').astype(np.float32) > 0.5).astype(np.float32),
}
Mall = np.clip(sum(masks.values()), 0, 1)
Ms = cv2.GaussianBlur(Mall, (0, 0), 0.6)
B98 = np.where(Ms[..., None] > 0.5, 1 - I98, I98)
edge98 = (Ms > 0.1) & (Ms < 0.9)
def warp(img, A, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, A[:2].astype(np.float32), (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT)
res = {}
for name, M in masks.items():
    res[name] = {}
    region = ndi.binary_dilation(M > 0.5, iterations=14)
    if name == 'They': region[560:] = False      # keep to the neon area (no pedestrians)
    for t in range(86, 100):
        A = sim_matrix(ref, t)
        Bt = warp(B98, A); Mt = warp(M, A); Rt = warp(region.astype(np.float32), A) > 0.5
        Et = warp(edge98.astype(np.float32), A) > 0.05
        It = F(t)
        lev = np.abs(1 - 2*Bt).mean(-1)
        sel = Rt & (lev > 0.2) & ~Et
        best = None
        for sg in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
            G = cv2.GaussianBlur(Mt, (0, 0), sg) if sg > 0 else Mt
            d = (1 - 2*Bt)[sel]; g = G[sel][:, None]; y = (It - Bt)[sel]
            x = (g*d).ravel(); yy = y.ravel()
            al = float((x*yy).sum()/max((x*x).sum(), 1e-9))
            al = min(max(al, 0), 1.2)
            err = float(np.sqrt(((yy - al*x)**2).mean()))
            if best is None or err < best[0]: best = (err, sg, al)
        err0 = float(np.sqrt((((It - Bt)[sel])**2).mean()))
        res[name][t] = dict(alpha=round(best[2], 3), blur_sigma_px=best[1], rmse=round(best[0], 4), rmse_no_text=round(err0, 4), n=int(sel.sum()))
        print(f'{name:6s} f{t:03d} alpha={best[2]:.2f} sigma={best[1]:>2}px rmse={best[0]:.3f} (no-text rmse {err0:.3f}) n={int(sel.sum())}')
json.dump(res, open(f'{C}/they_fade.json', 'w'), indent=1)
