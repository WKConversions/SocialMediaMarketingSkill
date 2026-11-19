# MARSOPS launch teaser (15 s, 9:16)

Deliverable: `out/MARSOPS_teaser_15s.mp4` (1080x1920, 30 fps, H.264 + AAC 48 kHz, -14 LUFS, true peak -1.5 dBTP).
Storyboard: `out/MARSOPS_storyboard.png`. Plan, thesis and cue sheet: `storyboard/plan.md`.

## How it is built
A coded render (the senior-motion-designer skill's fallback route: no After Effects in this session, Remotion not
needed for a one-off): `build/render.html` is one frame-pure `render(f)` (SVG sketch strokes drawn by dash offset,
hatching revealed by a hand's sweep, mirror-script notes written right to left, rays with their own schedule),
rendered by `build/render.mjs` (Playwright + sharp, 4 motion-blur samples per frame across a 180-degree shutter).

    cd build && npm i && node render.mjs full 0 449 4 4
    ffmpeg -framerate 30 -i frames/f%04d.jpg -i sound/mix.wav -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
      -movflags +faststart -c:a aac -b:a 192k -ar 48000 -shortest ../out/MARSOPS_teaser_15s.mp4

Sound: `build/sound/music_make.py music_brief.json music.wav` (composed for the film, custom "cinematic-pulse" vibe:
120 BPM, D minor, i-VI-III-VII, bars on the scene turns, the final chord on the logo), then
`build/sound/mix_teaser.py cues.json music.wav mix.wav` (Geiger clicks, shield tinks, shimmer, riser, impact on the
cues exported from the page; mastered to -14 LUFS).

## Credits and rights
- Music and sound effects: composed and synthesized for this film (royalty-free by construction).
- Fonts: Montserrat (SIL OFL 1.1, `build/fonts/OFL-Montserrat.txt`); La Belle Aurore (SIL OFL 1.1,
  `build/fonts/OFL-LaBelleAurore.txt`) for the mirror-script notes.
- Logo: the supplied raster (212x48 px, `assets/logo_supplied.png`) rebuilt as live type at the measured match:
  Montserrat ExtraBold 800, letter-spacing 0.10 em (IoU 0.80 against the supplied file).
- All drawings are original (drawn in code); no third-party images.
