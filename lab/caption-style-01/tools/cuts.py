import numpy as np
from common import frames, N
a = frames()
small = []
for i in range(N):
    f = a[i, ::8, ::8].astype(np.float32)/255
    small.append(f)
small = np.stack(small)
# histogram metric (HSV-ish via RGB 8x8x8 bins)
def hist(f):
    q = np.clip((f*8).astype(int),0,7)
    idx = q[...,0]*64+q[...,1]*8+q[...,2]
    h = np.bincount(idx.ravel(), minlength=512).astype(np.float32)
    return h/h.sum()
hs = [hist(f) for f in small]
rows=[]
for i in range(1,N):
    mad = np.abs(small[i]-small[i-1]).mean()
    hd = 0.5*np.abs(hs[i]-hs[i-1]).sum()
    # correlation of luma
    l1 = small[i].mean(-1).ravel(); l0 = small[i-1].mean(-1).ravel()
    c = np.corrcoef(l0,l1)[0,1]
    rows.append((i+1, mad, hd, c))
for r in rows:
    flag = '  <==' if (r[2]>0.25 or r[3]<0.6) else ''
    print(f'f{r[0]:03d}  mad={r[1]:.3f}  hist={r[2]:.3f}  corr={r[3]:.3f}{flag}')
