#!/bin/bash
# wide (the client's own framing) / medium 1.08 / close 1.15 views of each clip, 1080x1920, centred crops only
# (client rule: gentle crops). ffmpeg autorotates the iPhone 90-degree clips.
cd /home/user/SocialMediaMarketingSkill/lab/edit-07
for c in 6952 6953 6954 6955 6956 6957 6958; do
  for v in "wide 1.0" "medium 1.08" "close 1.15"; do
    set -- $v
    ffmpeg -hide_banner -loglevel error -y -i footage/$c.mov -vf "crop=iw/$2:ih/$2:(iw-iw/$2)*0.5:(ih-ih/$2)*0.5,scale=1080:1920:flags=lanczos" \
      -an -r 24 -c:v libx264 -crf 13 -preset fast -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 views/${c}_$1.mp4
  done
done
echo views done
