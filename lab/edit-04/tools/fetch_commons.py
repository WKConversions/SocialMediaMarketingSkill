#!/usr/bin/env python3
"""Download Commons files (original) + metadata sidecar.

Usage: fetch_commons.py <dest_root> <slug>=<File:Title> [<slug>=<File:Title> ...]
Each file goes to <dest_root>/<slug>/ (fresh folder) with meta.json next to it.
"""
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm  # noqa: E402


def main():
    root = sys.argv[1]
    pairs = [a.split("=", 1) for a in sys.argv[2:]]
    titles = [t for _, t in pairs]
    d = wm.api({"action": "query", "titles": "|".join(titles), "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata"})
    norm = {n["from"]: n["to"] for n in d["query"].get("normalized", [])}
    pages = {p["title"]: p for p in d["query"]["pages"]}
    for slug, t in pairs:
        p = pages.get(norm.get(t, t))
        if not p or "imageinfo" not in p:
            print("MISSING", t)
            continue
        ii = p["imageinfo"][0]
        em = ii.get("extmetadata", {})
        g = lambda k: wm.strip(em.get(k, {}).get("value", ""))
        url = ii["url"].split("?")[0]
        ext = os.path.splitext(urllib.parse.unquote(url))[1].lower()
        dest = os.path.join(root, slug)
        os.makedirs(dest, exist_ok=True)
        out = os.path.join(dest, "source" + ext)
        if not os.path.exists(out) or os.path.getsize(out) != ii.get("size"):
            data = wm.fetch(url, binary=True)
            with open(out, "wb") as f:
                f.write(data)
            time.sleep(4)
        meta = {
            "commons_title": p["title"], "source_url": ii.get("descriptionurl"),
            "file_url": url, "width": ii.get("width"), "height": ii.get("height"),
            "size": ii.get("size"), "mime": ii.get("mime"),
            "license": g("LicenseShortName"), "license_url": g("LicenseUrl"),
            "artist": g("Artist"), "credit": g("Credit"),
            "date": g("DateTimeOriginal") or g("DateTime"),
            "description": g("ImageDescription"),
            "object_name": g("ObjectName"),
        }
        with open(os.path.join(dest, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        print("OK", slug, out, os.path.getsize(out), meta["license"])


if __name__ == "__main__":
    main()
