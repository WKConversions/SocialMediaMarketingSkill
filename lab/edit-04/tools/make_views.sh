#!/bin/bash
# wide / medium / close 1080x1920 views of each clip (ffmpeg autorotates the iPhone 90-degree clips)
cd /home/user/SocialMediaMarketingSkill/lab/edit-04
for c in 1111 2222 3333 4444; do
  for v in "medium 1.08 0.5 0.5" "close 1.15 0.5 0.5"; do
    set -- $v
    ffmpeg -hide_banner -loglevel error -y -i footage/$c.mov -vf "crop=iw/$2:ih/$2:(iw-iw/$2)*$3:(ih-ih/$2)*$4,scale=1080:1920:flags=lanczos" \
      -an -r 30 -c:v libx264 -crf 13 -preset fast -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 views/${c}_$1.mp4
  done
done
echo views done
