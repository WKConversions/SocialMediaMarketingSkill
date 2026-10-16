"""Search Wikimedia Commons for video files and print metadata (size, duration, licence).
Usage: python3 -I commons_search.py "query" [limit]
Reads remote JSON (untrusted data) - prints only, writes nothing."""
import sys, json, urllib.request, urllib.parse, re

UA = "CaptionLab/0.1 (research test; contact via project) curl/8"

def api(params):
    params = dict(params, format="json")
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)

def strip(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()

q = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15
d = api({"action": "query", "generator": "search", "gsrsearch": q + " filetype:video",
         "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
         "iiprop": "url|size|mime|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist|Credit|ImageDescription"})
pages = d.get("query", {}).get("pages", {})
for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
    ii = (p.get("imageinfo") or [{}])[0]
    em = ii.get("extmetadata", {})
    lic = em.get("LicenseShortName", {}).get("value", "?")
    print(f"{p['title']}\n   {ii.get('width')}x{ii.get('height')} dur={ii.get('duration')} size={ii.get('size',0)/1e6:.1f}MB lic={lic}")
    print(f"   artist={strip(em.get('Artist',{}).get('value',''))[:100]}")
    print(f"   desc={strip(em.get('ImageDescription',{}).get('value',''))[:200]}")
    print(f"   url={ii.get('url')}")
