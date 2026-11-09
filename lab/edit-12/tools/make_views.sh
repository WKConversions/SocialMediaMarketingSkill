#!/bin/bash
# wide (the client's own framing) and medium (1.07, gentle) views of each clip, 1080x1920 at 24 fps, centred crops
# only (client rule: gentle crops). ffmpeg autorotates the iPhone 90-degree clips.
cd /home/user/SocialMediaMarketingSkill/lab/edit-12
for c in f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12 f13 f14 f15 f16 f17; do
  for v in "wide 1.0" "medium 1.07"; do
    set -- $v
    ffmpeg -hide_banner -loglevel error -y -i footage/$c.mov -vf "crop=iw/$2:ih/$2:(iw-iw/$2)*0.5:(ih-ih/$2)*0.42,scale=1080:1920:flags=lanczos" \
      -an -r 24 -c:v libx264 -crf 15 -preset fast -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 views/${c}_$1.mp4
  done
done
echo views done
