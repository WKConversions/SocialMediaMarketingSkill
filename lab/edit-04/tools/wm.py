#!/usr/bin/env python3
"""Small helper for Wikimedia Commons / LoC / archive.org JSON APIs with backoff.

Usage:
  wm.py search "<query>" [limit]            -> Commons file search (ns 6)
  wm.py info "File:Name.ext" [...]          -> imageinfo + extmetadata (licence, author, date)
  wm.py get <url> <outfile>                 -> download with UA + backoff
  wm.py json <url>                          -> fetch any JSON url with backoff, pretty print
"""
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

UA = "WKC-social-video-editor/0.1 (https://wkconversions.com)"
API = "https://commons.wikimedia.org/w/api.php"


def fetch(url, binary=False, tries=8):
    delay = 3.0
    last = None
    for i in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else delay
                sys.stderr.write(f"[{e.code}] retry in {wait:.0f}s: {url[:120]}\n")
                time.sleep(min(wait, 90))
                delay *= 1.8
                continue
            raise
        except Exception as e:  # network hiccup
            last = e
            sys.stderr.write(f"[err {e}] retry in {delay:.0f}s\n")
            time.sleep(delay)
            delay *= 1.8
    raise RuntimeError(f"failed after {tries}: {url} ({last})")


def api(params):
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    url = API + "?" + urllib.parse.urlencode(params)
    txt = fetch(url)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        sys.stderr.write("non-json: " + txt[:300] + "\n")
        raise


def search(q, limit=50):
    d = api({"action": "query", "list": "search", "srsearch": q, "srnamespace": 6,
             "srlimit": limit})
    for x in d["query"]["search"]:
        print(x["title"], "|", x.get("size"))


def strip(s):
    import re
    s = re.sub(r"<[^>]+>", "", s or "")
    return " ".join(s.split())


def info(titles):
    out = []
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        d = api({"action": "query", "titles": "|".join(chunk), "prop": "imageinfo",
                 "iiprop": "url|size|mime|extmetadata|mediatype", "iiurlwidth": 2400})
        for p in d["query"]["pages"]:
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata", {})
            g = lambda k: strip(em.get(k, {}).get("value", ""))
            out.append({
                "title": p.get("title"),
                "missing": p.get("missing", False),
                "url": ii.get("url"),
                "thumburl": ii.get("thumburl"),
                "descriptionurl": ii.get("descriptionurl"),
                "width": ii.get("width"), "height": ii.get("height"),
                "size": ii.get("size"), "mime": ii.get("mime"),
                "duration": ii.get("duration"),
                "license": g("LicenseShortName"), "license_url": g("LicenseUrl"),
                "usage_terms": g("UsageTerms"),
                "artist": g("Artist"), "credit": g("Credit"),
                "date": g("DateTimeOriginal") or g("DateTime"),
                "description": g("ImageDescription")[:500],
                "attribution_required": g("AttributionRequired"),
                "copyrighted": g("Copyrighted"),
            })
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 50)
    elif cmd == "info":
        info(sys.argv[2:])
    elif cmd == "get":
        data = fetch(sys.argv[2], binary=True)
        with open(sys.argv[3], "wb") as f:
            f.write(data)
        print(len(data), "bytes ->", sys.argv[3])
    elif cmd == "json":
        print(json.dumps(json.loads(fetch(sys.argv[2])), indent=1, ensure_ascii=False)[:20000])
