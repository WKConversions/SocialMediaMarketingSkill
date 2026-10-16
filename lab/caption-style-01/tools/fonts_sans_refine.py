"""Fine grid search for the chosen sans (Inter, opsz 32, ss07 square dots)."""
import json, itertools
from fonts_fit import *
A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
refs = prep(SANS_REFS)
path = font_path('inter')
best = None
rows = []
for w, opsz, size, trk in itertools.product([725, 750, 775, 800, 825], [28, 32], np.arange(109, 114.01, 0.75), np.arange(-0.085, -0.0549, 0.005)):
    s = score(path, {'wght': w, 'opsz': opsz}, size, trk, refs, r=2, features=['ss07'])[0]
    rows.append((s, w, opsz, size, trk))
rows.sort(reverse=True)
for r in rows[:12]:
    print(f'iou={r[0]:.4f} wght={r[1]} opsz={r[2]} size={r[3]:.2f} trk={r[4]:+.4f}')
s, w, opsz, size, trk = rows[0]
final, per = score(path, {'wght': w, 'opsz': opsz}, size, trk, refs, r=3, features=['ss07'])
print('final', final, per)
