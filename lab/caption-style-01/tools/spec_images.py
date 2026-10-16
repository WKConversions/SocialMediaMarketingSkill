"""Helper images for the reference spec."""
import numpy as np, json, cv2
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
from common import frames, LAB, W, H
from blendtest import warped_mask
a = frames()
C = LAB + '/analysis/cache'; OUT = LAB + '/analysis'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Bold.otf', 16)
FONTs = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Regular.otf', 13)

def undiff_panel():
    """actual | text 'un-differenced' (inside mask: 1-I). If the blend is difference, letters vanish."""
    items = [('w03', 13, 'even f013 (watch behind)'), ('w22', 123, 'know f123 (dark jacket / bright sky)'),
             ('w10', 51, 'hundred f051 (red/green chart)'), ('w18', 101, 'see f101 (monitor)'),
             ('w14', 72, 'charging f072 (monitor)'), ('w07', 38, "i'm f038 (hedge / deck / floor)"),
             ('w24', 132, 'end f132 (blurred spire)'), ('w28', 152, 'Regardless f152'),
             ('w20', 113, 'Win f113 (NORMAL blend: does not vanish)')]
    rows = []
    for wid, n, label in items:
        M, sc = warped_mask(wid, n)
        Ms = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 0.7)[..., None]
        I = a[n-1].astype(np.float32)/255
        U = I*(1-Ms) + (1-I)*Ms
        ys, xs = np.nonzero(M)
        x0, x1, y0, y1 = max(0, xs.min()-30), min(W, xs.max()+30), max(0, ys.min()-25), min(H, ys.max()+25)
        pair = np.concatenate([I[y0:y1, x0:x1], np.ones((y1-y0, 6, 3)), U[y0:y1, x0:x1]], 1)
        im = Image.fromarray((np.clip(pair, 0, 1)*255).astype(np.uint8))
        sc_ = min(1.0, 900/im.width); im = im.resize((int(im.width*sc_), int(im.height*sc_)), Image.LANCZOS)
        canvas = Image.new('RGB', (920, im.height+24), (20, 20, 20)); canvas.paste(im, (10, 22))
        ImageDraw.Draw(canvas).text((10, 3), label + '   [left: frame | right: glyph pixels replaced by 1-I]', fill=(255, 220, 0), font=FONTs)
        rows.append(canvas)
    Hh = sum(r.height for r in rows); S = Image.new('RGB', (920, Hh), (20, 20, 20)); y = 0
    for r in rows: S.paste(r, (0, y)); y += r.height
    S.save(f'{OUT}/spec_difference_evidence.png')

def zoom_plot(z):
    Wp, Hp = 1600, 520
    im = Image.new('RGB', (Wp, Hp), (250, 250, 250)); d = ImageDraw.Draw(im)
    x = lambda f: 60 + (f-1)*(Wp-90)/162
    # cumulative log zoom per shot segment (reset at cuts)
    CUTS = [1, 9, 11, 17, 23, 37, 41, 50, 53, 70, 86, 100, 106, 111, 122, 132, 137, 148]
    y = lambda v: Hp/2 - v*1100
    d.line([(60, y(0)), (Wp-30, y(0))], fill=(150, 150, 150))
    for c in CUTS:
        d.line([(x(c), 40), (x(c), Hp-60)], fill=(210, 210, 210))
    cum = 0; pts = []
    for f in range(1, 164):
        if f in CUTS and f not in (53,): cum = 0
        pts.append((x(f), y(cum)))
        if str(f) in z: cum += np.log(z[str(f)][0])
        elif f == 52: cum += np.log(1.0283)        # hundred continues across the f053 cut (text-measured)
    for i in range(len(pts)-1):
        if (i+2) in CUTS and (i+2) != 53: continue
        d.line([pts[i], pts[i+1]], fill=(20, 90, 200), width=3)
    for v in (-0.2, -0.1, 0.1, 0.2):
        d.text((5, y(v)-8), f'{np.exp(v)*100-100:+.0f}%', fill=(90, 90, 90), font=FONTs)
        d.line([(55, y(v)), (Wp-30, y(v))], fill=(235, 235, 235))
    words = json.load(open(f'{C}/words_final.json'))
    for w in words:
        col = (200, 40, 40) if w['font_class'] == 'script' else (60, 60, 60)
        d.text((x(w['first_frame']), Hp-52 + (12 if w['id'] in ('w12', 'w27', 'w25') else 0)), w['text'], fill=col, font=FONTs)
    d.text((60, 8), 'Digital zoom of the whole composite (footage + captions), cumulative per shot. Blue = measured. Red labels = script words (punch-ins start on their first frame).', fill=(0, 0, 0), font=FONTs)
    im.save(f'{OUT}/spec_zoom.png')

def timeline(words):
    Wp, Hp = 1700, 330
    im = Image.new('RGB', (Wp, Hp), (250, 250, 250)); d = ImageDraw.Draw(im)
    x = lambda f: 40 + (f-1)*(Wp-60)/163
    CUTS = [1, 9, 11, 17, 23, 37, 41, 50, 53, 70, 86, 100, 106, 111, 122, 132, 137, 148]
    for i, c in enumerate(CUTS):
        e = CUTS[i+1] if i+1 < len(CUTS) else 164
        d.rectangle([x(c), 30, x(e), 60], fill=(225, 232, 245) if i % 2 == 0 else (205, 215, 235), outline=(120, 120, 140))
        d.text((x(c)+2, 34), f'{c}', fill=(40, 40, 80), font=FONTs)
    d.text((40, 8), 'Shots (new shot starts at the labelled frame)', fill=(0, 0, 0), font=FONTs)
    for k, w in enumerate(words):
        col = {'difference': (40, 40, 40), 'normal': (230, 230, 230), 'difference_pink': (226, 139, 139)}[w['blend_key']]
        y0 = 200 if w['id'] in ('w16', 'w17') else 90 + (k % 2)*60
        d.rectangle([x(w['first_frame']), y0, x(w['last_frame']+1), y0+40], fill=col, outline=(0, 0, 0))
        tc = (255, 255, 255) if w['blend_key'] == 'difference' else (0, 0, 0)
        d.text((x(w['first_frame'])+3, y0+3), w['text'][:10], fill=tc, font=FONT if w['font_class'] == 'script' else FONTs)
        d.text((x(w['first_frame'])+3, y0+24), f"{w['first_frame']}-{w['last_frame']}", fill=tc, font=FONTs)
    d.text((40, 270), 'Fill: dark = white text in Difference;  light = white Normal (Win, but i);  pink = Difference with pink #E28B8B (at my, Prime).  Bold label = script font.', fill=(0, 0, 0), font=FONTs)
    d.text((40, 290), 'All entrances/exits are hard cuts on frame boundaries except the stacked They/don\'t/wanna block (staggered blur+fade in).', fill=(0, 0, 0), font=FONTs)
    im.save(f'{OUT}/spec_timeline.png')

def they_panel():
    I = a[97].copy()
    im = Image.fromarray(I).crop((200, 300, 900, 800)); d = ImageDraw.Draw(im)
    sub = json.load(open(f'{C}/they_layout.json'))
    for name, col in [('They', (255, 0, 255)), ("don't", (0, 255, 0)), ('wanna', (255, 255, 0))]:
        b = sub[name]['bbox_f098']
        d.rectangle([b[0]-200, b[1]-300, b[2]-200, b[3]-300], outline=col, width=2)
        d.text((b[2]-200+4, b[1]-300), name, fill=col, font=FONT)
    curves = json.load(open(f'{C}/they_edgeprofile.json'))
    P = Image.new('RGB', (700, 260), (250, 250, 250)); g = ImageDraw.Draw(P)
    g.text((10, 5), 'normalised opacity (solid) and extra blur sigma px (dashed) per frame', fill=(0, 0, 0), font=FONTs)
    X = lambda f: 40 + (f-86)*640/13
    for name, col in [('They', (200, 0, 200)), ("don't", (0, 150, 0)), ('wanna', (190, 150, 0))]:
        c = curves[name]; final = c['98']['alpha']
        pa = []; pb = []
        for f in range(86, 100):
            al = max(0.0, c[str(f)]['alpha']/final); sg = c[str(f)]['sigma_px']
            ex = np.sqrt(max(sg**2 - 1.0, 0)) if al > 0.3 else None
            pa.append((X(f), 230 - 180*min(al, 1.1)))
            if ex is not None: pb.append((X(f), 230 - 180*ex/8))
        g.line(pa, fill=col, width=3)
        for i in range(0, len(pb)-1): g.line([pb[i], pb[i+1]], fill=col, width=1)
    for f in range(86, 100): g.text((X(f)-8, 236), str(f), fill=(60, 60, 60), font=FONTs)
    S = Image.new('RGB', (700, 760), (20, 20, 20)); S.paste(im, (0, 0)); S.paste(P, (0, 500))
    S.save(f'{OUT}/spec_they_block.png')

if __name__ == '__main__':
    import sys
    what = sys.argv[1:]
    if 'undiff' in what: undiff_panel()
    if 'zoom' in what: zoom_plot(json.load(open(f'{C}/zoom_inc.json')))
    if 'timeline' in what: timeline(json.load(open(f'{C}/words_final.json')))
    if 'they' in what: they_panel()
