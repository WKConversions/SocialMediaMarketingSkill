#!/bin/bash
# usage: http.sh URL OUTFILE  -- polite GET with UA and 429 backoff
UA="WKC-social-video-editor/0.1 (https://wkconversions.com)"
url="$1"; out="$2"
for i in 1 2 3 4 5 6; do
  code=$(curl -sS -L -A "$UA" -o "$out" -w '%{http_code}' "$url")
  if [ "$code" = "200" ]; then exit 0; fi
  echo "HTTP $code for $url (try $i)" >&2
  if [ "$code" = "429" ] || [ "$code" = "503" ] || [ "$code" = "000" ]; then sleep $((i*8)); else exit 1; fi
done
exit 1
