#!/usr/bin/env python3
"""MET collection API helpers.
  met.py list <id> [<id> ...]                 -> one line per object (PD only flagged)
  met.py get <dest_dir> <id>                   -> source.jpg + meta.json in dest_dir
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm  # noqa: E402

API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"


def obj(i):
    return json.loads(wm.fetch(API + str(i)))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        for i in sys.argv[2:]:
            try:
                d = obj(i)
            except Exception as e:
                print(i, "ERR", e)
                continue
            if not d.get("objectID"):
                continue
            print(d["objectID"], "|PD" if d.get("isPublicDomain") else "|--", "|",
                  d.get("title", "")[:100], "|", d.get("artistDisplayName", "")[:40], "|",
                  d.get("objectDate"), "|", d.get("classification"), "|", d.get("medium", "")[:40])
            time.sleep(0.3)
    elif sys.argv[1] == "get":
        dest, i = sys.argv[2], sys.argv[3]
        d = obj(i)
        assert d.get("isPublicDomain"), "not open access"
        os.makedirs(dest, exist_ok=True)
        data = wm.fetch(d["primaryImage"], binary=True)
        open(os.path.join(dest, "source.jpg"), "wb").write(data)
        meta = {k: d.get(k) for k in ["objectID", "title", "artistDisplayName", "artistDisplayBio",
                                      "objectDate", "medium", "creditLine", "objectURL",
                                      "primaryImage", "isPublicDomain", "accessionNumber"]}
        meta["license"] = "CC0 1.0 (The Met Open Access)"
        json.dump(meta, open(os.path.join(dest, "meta.json"), "w"), indent=1, ensure_ascii=False)
        print("OK", dest, len(data))
