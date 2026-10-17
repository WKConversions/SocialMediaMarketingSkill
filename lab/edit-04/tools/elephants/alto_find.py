# usage: python3 -I alto_find.py ALTO.xml word1,word2 ... prints text lines (with coords) around matches
import sys, re, xml.etree.ElementTree as ET
p, words = sys.argv[1], [w.lower() for w in sys.argv[2].split(',')]
ns = {'a':'http://schema.ccs-gmbh.com/ALTO'}
root = ET.parse(p).getroot()
lines = []
for tb in root.iter('{http://schema.ccs-gmbh.com/ALTO}TextBlock'):
    for tl in tb.findall('a:TextLine', ns):
        strs = tl.findall('a:String', ns)
        txt = ' '.join(s.get('CONTENT','') for s in strs)
        lines.append((tb.get('ID'), int(float(tl.get('HPOS'))), int(float(tl.get('VPOS'))), int(float(tl.get('WIDTH'))), int(float(tl.get('HEIGHT'))), txt))
ctx = int(sys.argv[3]) if len(sys.argv)>3 else 3
hits = [i for i,l in enumerate(lines) if any(w in l[5].lower() for w in words)]
shown=set()
for i in hits:
    for j in range(max(0,i-ctx), min(len(lines), i+ctx+1)):
        if j in shown: continue
        shown.add(j); l=lines[j]
        print(f"{'>>' if j==i else '  '} {l[0]} x={l[1]} y={l[2]} w={l[3]} h={l[4]} | {l[5]}")
    print('--')
