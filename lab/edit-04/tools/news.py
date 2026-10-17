#!/usr/bin/env python3
"""Chronicling America (loc.gov) newspaper page fetcher.

  news.py fetch <dest_dir> <lccn> <YYYY-MM-DD> <seq> [pct]
      -> dest_dir/page.jpg (IIIF, pct of full res, default 50), page.xml (ALTO), meta.json
  news.py find <dest_dir> <regex> [min_height]
      -> lines in ALTO matching regex with bbox (in full-res px) and glyph height
"""
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm  # noqa: E402


def fetch(dest, lccn, date, seq, pct="50"):
    os.makedirs(dest, exist_ok=True)
    url = f"https://www.loc.gov/resource/{lccn}/{date}/ed-1/?sp={seq}&fo=json"
    d = json.loads(wm.fetch(url))
    full = [p for p in d["page"] if p.get("mimetype") == "image/jp2"][0]
    info = full["info"]
    base = info.rsplit("/info.json", 1)[0]
    img_url = f"{base}/full/pct:{pct}/0/default.jpg"
    xml_url = [p for p in d["page"] if p.get("mimetype") == "text/xml"][0]["url"]
    time.sleep(2)
    data = wm.fetch(img_url, binary=True)
    open(os.path.join(dest, "page.jpg"), "wb").write(data)
    time.sleep(2)
    x = wm.fetch(xml_url, binary=True)
    open(os.path.join(dest, "page.xml"), "wb").write(x)
    item = d.get("item", {})
    meta = {
        "title": item.get("title") or d.get("cite_this", {}).get("mla"),
        "newspaper_title": (item.get("newspaper_title") or [None])[0] if isinstance(item.get("newspaper_title"), list) else item.get("newspaper_title"),
        "date": date, "seq": seq, "lccn": lccn,
        "source_url": f"https://www.loc.gov/resource/{lccn}/{date}/ed-1/?sp={seq}",
        "chronam_url": f"https://chroniclingamerica.loc.gov/lccn/{lccn}/{date}/ed-1/seq-{seq}/",
        "image_url": img_url, "iiif_base": base, "full_width": full.get("width"),
        "full_height": full.get("height"), "pct": pct,
        "rights": item.get("rights") or d.get("rights"),
        "cite_mla": d.get("cite_this", {}).get("mla"),
    }
    json.dump(meta, open(os.path.join(dest, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print("OK", dest, len(data), "bytes", meta["full_width"], meta["full_height"])


def find(dest, rx, min_h=0):
    tree = ET.parse(os.path.join(dest, "page.xml"))
    root = tree.getroot()
    ns = {"a": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = ".//a:TextLine" if ns else ".//TextLine"
    sq = "a:String" if ns else "String"
    # ALTO measurement unit for NDNP is 1/1200 inch; scale to pixels via Page WIDTH
    page = root.find(".//a:Page" if ns else ".//Page", ns)
    pw = float(page.get("WIDTH"))
    meta = json.load(open(os.path.join(dest, "meta.json")))
    scale = meta["full_width"] / pw
    r = re.compile(rx, re.I)
    for tl in root.findall(q, ns):
        words = [s.get("CONTENT", "") for s in tl.findall(sq, ns)]
        text = " ".join(words)
        h = float(tl.get("HEIGHT", 0)) * scale
        if r.search(text) and h >= min_h:
            x, y = float(tl.get("HPOS")) * scale, float(tl.get("VPOS")) * scale
            w = float(tl.get("WIDTH")) * scale
            print(f"x={x:.0f} y={y:.0f} w={w:.0f} h={h:.0f} | {text}")


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(*sys.argv[2:])
    elif sys.argv[1] == "find":
        find(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0)
