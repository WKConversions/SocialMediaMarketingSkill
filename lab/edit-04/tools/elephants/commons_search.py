import sys, json, urllib.parse, subprocess, os, tempfile
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'http.sh')
def get(url):
    fd, p = tempfile.mkstemp(); os.close(fd)
    r = subprocess.run([H, url, p])
    data = open(p,'rb').read(); os.unlink(p)
    if r.returncode != 0: raise SystemExit('fail '+url)
    return json.loads(data)
q = sys.argv[1]; n = sys.argv[2] if len(sys.argv)>2 else '30'
url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode({
  'action':'query','format':'json','generator':'search','gsrsearch':q,'gsrnamespace':'6','gsrlimit':n,
  'prop':'imageinfo','iiprop':'url|size|mime|extmetadata','iiextmetadatafilter':'LicenseShortName|Artist|DateTimeOriginal|ImageDescription|Credit'})
d = get(url)
for p in sorted(d.get('query',{}).get('pages',{}).values(), key=lambda x: x.get('index',0)):
    ii = p.get('imageinfo',[{}])[0]; em = ii.get('extmetadata',{})
    g = lambda k: (em.get(k,{}).get('value','') or '')[:90].replace('\n',' ')
    print(f"{p['title']} | {ii.get('width')}x{ii.get('height')} {ii.get('mime')} | {g('LicenseShortName')} | {g('DateTimeOriginal')} | {g('Artist')}")
