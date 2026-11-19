# Tiles the test frames (test/f*.png, from `node render.mjs test ...`) into one labelled sheet: test/sheet.png.
# Needs Pillow.
from PIL import Image, ImageDraw
import glob
fs = sorted(glob.glob('test/f*.png')); cols, tw = 8, 216
im0 = Image.open(fs[0]); th = round(tw * im0.height / im0.width)
sheet = Image.new('RGB', (cols * tw, -(-len(fs) // cols) * (th + 20)), 'white'); d = ImageDraw.Draw(sheet)
for i, f in enumerate(fs):
    x, y = i % cols * tw, i // cols * (th + 20)
    sheet.paste(Image.open(f).convert('RGB').resize((tw, th)), (x, y + 20))
    d.text((x + 4, y + 4), f.split('/')[-1][:-4], fill='red')
sheet.save('test/sheet.png')
