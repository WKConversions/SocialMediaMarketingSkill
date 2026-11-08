#!/bin/bash
# wide (the client's own framing) and medium (1.07, gentle) views of each clip, 1080x1920 at 24 fps, centred crops
# only (client rule: gentle crops). ffmpeg autorotates the iPhone 90-degree clips.
cd /home/user/SocialMediaMarketingSkill/lab/edit-11
for c in 7145 7146 7147 7148 7149 7150; do
  for v in "wide 1.0" "medium 1.07"; do
    set -- $v
    ffmpeg -hide_banner -loglevel error -y -i footage/$c.mov -vf "crop=iw/$2:ih/$2:(iw-iw/$2)*0.5:(ih-ih/$2)*0.42,scale=1080:1920:flags=lanczos" \
      -an -r 24 -c:v libx264 -crf 13 -preset fast -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 views/${c}_$1.mp4
  done
done
echo views done
