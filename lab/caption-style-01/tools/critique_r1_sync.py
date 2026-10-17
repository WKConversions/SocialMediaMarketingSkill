"""Speech-band energy envelope vs caption on-screen spans and whisper word spans."""
import sys, wave, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
S = sys.argv[1]; LAB = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01'
w = wave.open(S + '/a_band.wav'); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
hop = 160; n = len(x) // hop
db = np.array([10 * np.log10((x[i*hop:(i+1)*hop]**2).mean() + 1e-10) for i in range(n)])
floor = np.percentile(db, 10); e = np.clip(db - floor, 0, 35) / 35
rep = json.load(open(LAB + '/out/user_v1.render.json')); words = json.load(open(LAB + '/footage/user/clip_words.json'))
W, H = 2400, 520; px = W / 4.8
im = Image.new('RGB', (W, H), (16, 16, 16)); d = ImageDraw.Draw(im)
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
for t in np.arange(0, 4.81, 0.1):
    X = int(t * px); d.line([X, 0, X, H], fill=(40, 40, 40) if round(t*10) % 5 else (70, 70, 70))
    if round(t * 10) % 5 == 0: d.text((X + 2, H - 22), f'{t:.1f}s', font=F, fill=(150, 150, 150))
for i in range(n - 1):
    X = int(i * 0.01 * px); d.line([X, 240, X, 240 - int(e[i] * 200)], fill=(80, 170, 255))
d.text((6, 6), 'speech-band energy (250-3500 Hz), dB above floor', font=F, fill=(80, 170, 255))
cols = [(255, 210, 0), (255, 140, 60)]
for k, u in enumerate(rep['units']):
    a, b = u['first'] / 30, (u['last'] + 1) / 30
    d.rectangle([int(a*px), 270, int(b*px) - 2, 330], fill=cols[k % 2]); d.text((int(a*px) + 4, 290), u['text'], font=F, fill=(0, 0, 0))
d.text((6, 250), 'caption on screen (render)', font=F, fill=(255, 210, 0))
for k, wd in enumerate(words):
    a, b = wd['start'], wd['end']
    d.rectangle([int(a*px), 370, int(b*px) - 2, 420], fill=(120, 220, 120) if k % 2 else (60, 160, 60)); d.text((int(a*px) + 3, 385), wd['word'], font=F, fill=(0, 0, 0))
d.text((6, 350), 'whisper word spans (clip_words.json)', font=F, fill=(120, 220, 120))
im.save(LAB + '/analysis/critique_round1_sync.png')
