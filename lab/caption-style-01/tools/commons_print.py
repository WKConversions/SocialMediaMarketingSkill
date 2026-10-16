"""Print Commons search JSON (untrusted data) saved by commons_search.sh. Usage: python3 -I commons_print.py file.json"""
import sys, json, re
def strip(s): return re.sub(r"<[^>]+>", "", s or "").strip()
d = json.load(open(sys.argv[1]))
pages = d.get("query", {}).get("pages", {})
for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
    ii = (p.get("imageinfo") or [{}])[0]
    em = ii.get("extmetadata", {})
    lic = em.get("LicenseShortName", {}).get("value", "?")
    print(f"{p['title']}\n   {ii.get('width')}x{ii.get('height')} dur={ii.get('duration')} size={ii.get('size',0)/1e6:.1f}MB lic={lic}")
    print(f"   artist={strip(em.get('Artist',{}).get('value',''))[:100]}")
    print(f"   desc={strip(em.get('ImageDescription',{}).get('value',''))[:160]}")
    print(f"   url={ii.get('url')}")
