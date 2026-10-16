#!/bin/bash
# Usage: commons_search.sh "query" limit outfile.json  -- fetch Commons video search results (JSON) with curl
curl -sS --max-time 30 -A "CaptionLab/0.1 (research test; contact via project) curl/8" -G "https://commons.wikimedia.org/w/api.php" \
  --data-urlencode "action=query" --data-urlencode "generator=search" --data-urlencode "gsrsearch=$1 filetype:video" \
  --data-urlencode "gsrnamespace=6" --data-urlencode "gsrlimit=$2" --data-urlencode "prop=imageinfo" \
  --data-urlencode "iiprop=url|size|mime|extmetadata" --data-urlencode "iiextmetadatafilter=LicenseShortName|Artist|ImageDescription" \
  --data-urlencode "format=json" -o "$3" -w "%{http_code}\n"
