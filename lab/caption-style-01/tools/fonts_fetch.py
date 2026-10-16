"""Download Google Fonts families (font files + license) from the google/fonts repo.

usage: python3 -I fonts_fetch.py family1 family2 ...
Each family goes to downloads/fontcands/<family>/ (untrusted data folder).
"""
import os, re, subprocess, sys, urllib.parse

LAB = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01'
DEST = LAB + '/downloads/fontcands'
RAW = 'https://raw.githubusercontent.com/google/fonts/main'


def curl(url, out=None):
    cmd = ['curl', '-sS', '-f', '-L', url]
    if out:
        cmd += ['-o', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout if out is None else True


def fetch(fam):
    d = os.path.join(DEST, fam)
    os.makedirs(d, exist_ok=True)
    for lic in ('ofl', 'apache', 'ufl'):
        meta = curl(f'{RAW}/{lic}/{fam}/METADATA.pb')
        if meta:
            break
    else:
        print(f'{fam}: not found')
        return
    meta = meta.decode('utf8', 'replace')
    open(os.path.join(d, 'METADATA.pb'), 'w').write(meta)
    files = re.findall(r'filename:\s*"([^"]+)"', meta)
    licfile = {'ofl': 'OFL.txt', 'apache': 'LICENSE.txt', 'ufl': 'UFL.txt'}[lic]
    curl(f'{RAW}/{lic}/{fam}/{licfile}', os.path.join(d, licfile))
    got = []
    for fn in files:
        if 'Italic' in fn and len(files) > 1 and fam not in ('playfairdisplay', 'bodonimoda'):
            continue
        out = os.path.join(d, fn)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            got.append(fn)
            continue
        if curl(f'{RAW}/{lic}/{fam}/' + urllib.parse.quote(fn), out):
            got.append(fn)
    print(f'{fam} [{lic}]: {got}')


if __name__ == '__main__':
    for fam in sys.argv[1:]:
        fetch(fam)
