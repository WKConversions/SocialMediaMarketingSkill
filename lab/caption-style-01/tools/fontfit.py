"""Fit a candidate sans font (size, tracking) to clean caption masks."""
import numpy as np, json, sys
from PIL import Image, ImageDraw, ImageFont
from words import WORDS
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
FONTS = {
 'InterDisplay-Bold': '/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf',
 'InterDisplay-ExtraBold': '/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf',
 'InterDisplay-SemiBold': '/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',
 'Inter-Bold': '/usr/share/fonts/opentype/inter/Inter-Bold.otf',
 'FreeSansBold': '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
 'LiberationSans-Bold': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
}
def render(text, fontpath, size, track_em, ss=2):
    f = ImageFont.truetype(fontpath, int(round(size*ss)))
    W = int(size*ss*(len(text)+2)); H = int(size*ss*2)
    im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    x = size*ss*0.5
    for i, ch in enumerate(text):
        xi = x + f.getlength(text[:i]) + i*track_em*size*ss
        d.text((xi, size*ss*0.3), ch, font=f, fill=255)
    im = im.resize((W//ss, H//ss), Image.LANCZOS)
    m = np.asarray(im) > 127
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max()+1, xs.min():xs.max()+1]
def iou(a, b):
    h = max(a.shape[0], b.shape[0]); w = max(a.shape[1], b.shape[1])
    A = np.zeros((h, w), bool); B = np.zeros((h, w), bool)
    A[:a.shape[0], :a.shape[1]] = a; B[:b.shape[0], :b.shape[1]] = b
    return (A & B).sum() / (A | B).sum()
def target(wid):
    M = np.load(f'{CACHE}/ref_{wid}.npy')
    ys, xs = np.nonzero(M)
    return M[ys.min():ys.max()+1, xs.min():xs.max()+1]
def fit(text, T, fontpath):
    best = None
    for track in np.arange(-0.10, 0.031, 0.01):
        # size by height bisection
        lo, hi = 20, 300
        for _ in range(14):
            mid = (lo+hi)/2
            r = render(text, fontpath, mid, track)
            if r.shape[0] < T.shape[0]: lo = mid
            else: hi = mid
        size = (lo+hi)/2
        r = render(text, fontpath, size, track)
        # allow +-2px shifts
        sc = max(iou(r[max(0,dy):, max(0,dx):], T[max(0,-dy):, max(0,-dx):]) for dy in (-1,0,1) for dx in (-2,-1,0,1,2))
        if best is None or sc > best[0]: best = (sc, size, track, r.shape)
    return best
if __name__ == '__main__':
    words = [('w26', 'winning'), ('w27', 'and'), ('w25', "i'm"), ('w04', 'yet'), ('w13', 'still'), ('w08', 'not')]
    res = {}
    for fname, fp in FONTS.items():
        res[fname] = {}
        for wid, text in words:
            T = target(wid)
            sc, size, track, shp = fit(text, T, fp)
            res[fname][wid] = dict(text=text, iou=round(float(sc), 4), size_px=round(size, 2), tracking_em=round(float(track), 3), target_hw=list(T.shape), render_hw=list(shp))
            print(f'{fname:24s} {text:8s} IoU={sc:.4f} size={size:.1f}px track={track:+.2f}em target={T.shape} render={shp}')
    json.dump(res, open(f'{CACHE}/fontfit.json', 'w'), indent=1)
