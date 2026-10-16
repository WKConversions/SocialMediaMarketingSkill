import numpy as np, os
from PIL import Image
LAB = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01'
FR = LAB + '/ref/frames'
CACHE = LAB + '/analysis/cache/frames.npy'
W, H, N = 1916, 1078, 163

def build_cache():
    arr = np.lib.format.open_memmap(CACHE, mode='w+', dtype=np.uint8, shape=(N, H, W, 3))
    for i in range(N):
        arr[i] = np.asarray(Image.open(f'{FR}/f{i+1:03d}.png').convert('RGB'))
    arr.flush()

def frames():
    if not os.path.exists(CACHE):
        build_cache()
    return np.load(CACHE, mmap_mode='r')

def F(n):
    """frame n (1-based) as float32 0..1"""
    return frames()[n-1].astype(np.float32) / 255.0

def save(img, path):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
