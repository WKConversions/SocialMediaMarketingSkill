#!/usr/bin/env python3
"""loc.gov helpers (JSON API, backoff via wm.fetch).
  loc.py search <url-with-fo=json>              -> list results (title | date | id | rights)
  loc.py item <item_id>                         -> print item summary + resource files
  loc.py get <dest_dir> <item_id> [maxpx]       -> largest image (jpg; tif fallback) + meta.json
  loc.py getfile <dest_dir> <item_id> <url>     -> download given file url + meta.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm  # noqa: E402


def item(i):
    return json.loads(wm.fetch(f"https://www.loc.gov/item/{i}/?fo=json"))


def summarize(d):
    it = d.get("item", {})
    return {
        "title": it.get("title"),
        "date": it.get("date"),
        "created_published": it.get("created_published"),
        "contributors": it.get("contributor_names") or it.get("contributors"),
        "rights": it.get("rights_advisory") or it.get("rights"),
        "medium": it.get("medium"),
        "reproduction_number": it.get("reproduction_number"),
        "notes": (it.get("notes") or [])[:6],
        "source_url": (d.get("item", {}).get("id") or "").replace("http://", "https://"),
    }


def files(d):
    out = []
    for r in d.get("resources", []):
        for grp in r.get("files", []):
            for f in grp:
                out.append(f)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        d = json.loads(wm.fetch(sys.argv[2]))
        for r in d.get("results", []):
            print(r.get("id"), "|", (r.get("title") or "")[:110], "|", r.get("date"), "|",
                  str(r.get("rights_advisory") or r.get("rights") or "")[:60])
    elif cmd == "item":
        d = item(sys.argv[2])
        print(json.dumps(summarize(d), indent=1, ensure_ascii=False))
        for f in files(d):
            print(f.get("mimetype"), f.get("width"), f.get("height"), f.get("size"), f.get("url"))
    elif cmd in ("get", "getfile"):
        dest, i = sys.argv[2], sys.argv[3]
        d = item(i)
        fs = files(d)
        if cmd == "get":
            maxpx = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
            jpgs = [f for f in fs if f.get("mimetype") == "image/jpeg" and f.get("width")
                    and max(f["width"], f.get("height") or 0) <= maxpx]
            jpgs.sort(key=lambda f: f["width"] * (f.get("height") or 1))
            url = jpgs[-1]["url"]
        else:
            url = sys.argv[4]
        os.makedirs(dest, exist_ok=True)
        ext = os.path.splitext(url.split("?")[0])[1] or ".bin"
        data = wm.fetch(url, binary=True)
        open(os.path.join(dest, "source" + ext), "wb").write(data)
        meta = summarize(d)
        meta["file_url"] = url
        meta["loc_item_id"] = i
        json.dump(meta, open(os.path.join(dest, "meta.json"), "w"), indent=1, ensure_ascii=False)
        print("OK", dest, len(data), url)
