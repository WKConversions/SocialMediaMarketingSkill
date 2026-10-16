import numpy as np
from common import frames, N
a = frames()
def feat(i):
    f = a[i, ::4, ::4].astype(np.float32)/255
    l = np.log(0.02 + f.mean(-1))
    gx = np.diff(l, axis=1)[:-1]; gy = np.diff(l, axis=0)[:, :-1]
    g = np.sqrt(gx**2+gy**2)
    # box blur 5x5 to tolerate small motion
    k=5
    c = np.cumsum(np.cumsum(np.pad(g,((k,0),(k,0))),0),1)
    b = (c[k:,k:]-c[:-k,k:]-c[k:,:-k]+c[:-k,:-k])/(k*k)
    # mask out central caption band (rows 35%..60%, cols 25..75%) to avoid text influence
    h,w = b.shape
    m = np.ones_like(b,bool); m[int(h*.30):int(h*.62), int(w*.05):int(w*.80)] = False
    v = b[m]
    return (v-v.mean())/(v.std()+1e-6)
fs = [feat(i) for i in range(N)]
for i in range(1,N):
    c = float((fs[i]*fs[i-1]).mean())
    flag = '  <== CUT?' if c < 0.5 else ''
    print(f'f{i+1:03d} structcorr={c:.3f}{flag}')
