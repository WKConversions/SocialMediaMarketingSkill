import numpy as np, sys
from common import frames
a = frames()
def g(i):
    f = a[i-1, ::2, ::2].astype(np.float32)/255
    l = np.log(0.02+f.mean(-1))
    gx = np.diff(l,axis=1)[:-1]; gy=np.diff(l,axis=0)[:,:-1]
    m = np.sqrt(gx**2+gy**2)
    win = np.outer(np.hanning(m.shape[0]), np.hanning(m.shape[1]))
    return (m-m.mean())*win
def pc(i,j):
    A=np.fft.fft2(g(i)); B=np.fft.fft2(g(j))
    R=A*np.conj(B); R/=np.abs(R)+1e-9
    r=np.real(np.fft.ifft2(R))
    k=np.unravel_index(np.argmax(r), r.shape)
    dy,dx=k
    if dy>r.shape[0]//2: dy-=r.shape[0]
    if dx>r.shape[1]//2: dx-=r.shape[1]
    return dx*2, dy*2, r.max()
for p in [(2,3),(4,5),(6,7),(1,2),(1,8),(9,10),(59,60),(56,57),(52,53),(77,78)]:
    print(p, pc(*p))
