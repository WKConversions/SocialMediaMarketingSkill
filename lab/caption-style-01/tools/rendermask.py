"""Clean glyph masks for the sans words: render Inter Display ExtraBold (stand-in font), register to the rep frame."""
import numpy as np, cv2, json, sys
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from common import W, H
from register import register
from words import WORDS
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
FONT = '/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf'
def render_full(text, cx, cy, size=110.0, track=-0.07, ss=3):
    f = ImageFont.truetype(FONT, int(round(size*ss)))
    w = int(size*ss*(len(text)+2)); h = int(size*ss*2)
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
    for i, ch in enumerate(text):
        d.text((size*ss*0.5 + f.getlength(text[:i]) + i*track*size*ss, size*ss*0.3), ch, font=f, fill=255)
    im = im.resize((w//ss, h//ss), Image.LANCZOS)
    m = np.asarray(im).astype(np.float32)/255
    ys, xs = np.nonzero(m > 0.5)
    m = m[ys.min():ys.max()+1, xs.min():xs.max()+1]
    full = np.zeros((H, W), np.float32)
    y0 = int(round(cy - m.shape[0]/2)); x0 = int(round(cx - m.shape[1]/2))
    full[y0:y0+m.shape[0], x0:x0+m.shape[1]] = m
    return full
SANS = {'w01': "i'm", 'w02': 'not', 'w03': 'even', 'w04': 'yet', 'w07': "i'm", 'w08': 'not', 'w09': 'at a', 'w11': 'yet',
        'w12': "i'm", 'w13': 'still', 'w14': 'charging', 'w18': 'see', 'w19': 'me', 'w21': 'but i', 'w22': 'know',
        'w23': 'in the', 'w24': 'end', 'w25': "i'm", 'w26': 'winning', 'w27': 'and'}
if __name__ == '__main__':
    ids = sys.argv[1:] or list(SANS)
    reg = json.load(open(f'{CACHE}/registration.json'))
    for wid in ids:
        w = [x for x in WORDS if x[0] == wid][0]
        rep = w[4]
        r0 = [f for f in reg[wid]['frames'] if f['frame'] == rep][0]
        full = render_full(SANS[wid], r0['cx'], r0['cy'])
        rr = register((full > 0.5).astype(np.float32), [rep], smin=0.85, smax=1.15)
        fr = rr['frames'][0]
        s = fr['scale']; cx, cy = rr['ref_cx'], rr['ref_cy']
        A = np.float32([[s, 0, fr['cx'] - s*cx], [0, s, fr['cy'] - s*cy]])
        R = cv2.warpAffine(full, A, (W, H), flags=cv2.INTER_LINEAR)
        np.save(f'{CACHE}/rend_{wid}.npy', R.astype(np.float16))
        print(wid, SANS[wid], 'rep', rep, 'score', fr['score'], 'scale', fr['scale'], 'centre', fr['cx'], fr['cy'])
