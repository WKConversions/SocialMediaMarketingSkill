import sys, json, urllib.parse, subprocess, os, tempfile
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'http.sh')
q = sys.argv[1]; rows = sys.argv[2] if len(sys.argv)>2 else '40'
url = 'https://archive.org/advancedsearch.php?' + urllib.parse.urlencode([('q',q),('fl[]','identifier'),('fl[]','title'),('fl[]','date'),('fl[]','licenseurl'),('fl[]','mediatype'),('fl[]','collection'),('fl[]','creator'),('rows',rows),('output','json')])
fd, p = tempfile.mkstemp(); os.close(fd)
r = subprocess.run([H, url, p]); d = json.load(open(p)); os.unlink(p)
print('found', d['response']['numFound'])
for x in d['response']['docs']:
    c = x.get('collection'); c = ','.join(c) if isinstance(c,list) else c
    print(f"{x['identifier']} | {str(x.get('title'))[:70]} | {x.get('date','')[:10]} | {x.get('mediatype')} | {c} | {x.get('licenseurl','')} | {str(x.get('creator',''))[:40]}")
