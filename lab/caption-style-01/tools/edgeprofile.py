"""Alpha profile across glyph edges -> opacity and blur (sigma) per frame, for difference-blend text.
A(d) = (I(d)-B).(1-2B)/|1-2B|^2 with B = I sampled 10-12 px outside the edge."""
import numpy as np, cv2, json, sys
from scipy import ndimage as ndi
from scipy.special import erf
from scipy.optimize import least_squares
from prime import sim_matrix, F
from common import W, H
C = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
D = np.arange(-12, 13, 1.0)
def profile(I, M, minwidth=12, ylim=None):
    Ms = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 1.0)
    gy, gx = np.gradient(Ms)
    edge = (M > 0.5) & ~ndi.binary_erosion(M > 0.5)
    din = ndi.distance_transform_edt(M > 0.5)
    ys, xs = np.nonzero(edge)
    if ylim: k = (ys >= ylim[0]) & (ys < ylim[1]); ys, xs = ys[k], xs[k]
    nx, ny = -gx[ys, xs], -gy[ys, xs]; L = np.hypot(nx, ny) + 1e-9; nx /= L; ny /= L   # outward normal
    # require thick stroke: inside point at 6px has inward distance >= minwidth/2
    px = np.clip(np.round(xs - 6*nx).astype(int), 0, W-1); py = np.clip(np.round(ys - 6*ny).astype(int), 0, H-1)
    ok = din[py, px] >= minwidth/2 - 0.5
    # outside points must be outside the mask
    ox = np.clip(np.round(xs + 11*nx).astype(int), 0, W-1); oy = np.clip(np.round(ys + 11*ny).astype(int), 0, H-1)
    ok &= M[oy, ox] < 0.5
    xs, ys, nx, ny = xs[ok], ys[ok], nx[ok], ny[ok]
    sx = (xs[None, :] + D[:, None]*nx[None, :]).astype(np.float32)
    sy = (ys[None, :] + D[:, None]*ny[None, :]).astype(np.float32)
    S = np.stack([cv2.remap(I[..., c], sx, sy, cv2.INTER_LINEAR) for c in range(3)], -1)  # len(D) x N x 3
    B = S[-3:].mean(0)                         # 10..12 px outside
    v = 1 - 2*B; vv = (v*v).sum(-1)
    good = vv > 3*0.3**2
    A = ((S[:, good] - B[good]) * v[good]).sum(-1) / vv[good]
    w = vv[good]
    prof = (A*w).sum(1)/w.sum()
    return prof, int(good.sum())
def fit(prof):
    def model(p):
        al, d0, sg = p
        return al*0.5*(1 - erf((D - d0)/(np.sqrt(2)*max(sg, 0.3))))
    r = least_squares(lambda p: model(p) - prof, [1.0, 0.0, 1.0], bounds=([0, -4, 0.3], [1.5, 4, 12]))
    return r.x, float(np.sqrt(np.mean(r.fun**2)))
if __name__ == '__main__':
    T = np.load(f'{C}/mask_They_f098.npy').astype(np.float32)
    Dn = (np.load(f'{C}/rend_dont.npy').astype(np.float32) > 0.5).astype(np.float32)
    Wn = (np.load(f'{C}/rend_wanna.npy').astype(np.float32) > 0.5).astype(np.float32)
    res = {}
    for name, M, ylim, mw in [('They', T, (340, 560), 10), ("don't", Dn, None, 12), ('wanna', Wn, None, 12)]:
        res[name] = {}
        for t in range(86, 100):
            A = sim_matrix(98, t)
            Mt = cv2.warpAffine(M, A[:2].astype(np.float32), (W, H), flags=cv2.INTER_LINEAR)
            yl = None
            if ylim:
                c = A[:2] @ np.array([0, ylim[0], 1]); c2 = A[:2] @ np.array([0, ylim[1], 1]); yl = (int(c[1]), int(c2[1]))
            prof, n = profile(F(t), Mt, mw, yl)
            (al, d0, sg), rm = fit(prof)
            res[name][t] = dict(alpha=round(float(al), 3), sigma_px=round(float(sg), 2), edge_offset_px=round(float(d0), 2), rmse=round(rm, 4), n=n,
                                profile=[round(float(x), 3) for x in prof])
            print(f'{name:6s} f{t:03d} alpha={al:.2f} sigma={sg:.2f}px offset={d0:+.2f} rmse={rm:.3f} n={n}  inside(-8..-5)={prof[4:8].mean():.2f}')
    json.dump(res, open(f'{C}/they_edgeprofile.json', 'w'), indent=1)
