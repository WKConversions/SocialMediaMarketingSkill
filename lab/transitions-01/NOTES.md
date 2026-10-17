# Reference: quick view switching, negative flicker, clean transitions

Reel by @imdebjiiit ("Dream of yours"), 13.5 s, 1920x1080, 30 fps, 403 frames.
Client brief: learn ONLY the transitions, the flickering and the quick switching between views.
Ignore its captions. Measurements: `analysis/measurements.json` (frame numbers are 1-based).

## Measured
- 62 shots in 13.5 s. Median shot 3 frames (0.1 s); 44 of 62 shots are 5 frames or shorter.
  Rhythm = bursts of 1-5-frame shots, then a few long holds (79, 53, 30 frames), then another burst.
- A handful of views keep coming back (eye, keyboard, headphones and book, photo wall, hand): it
  switches back and forth between known views rather than always cutting to new footage.
- The flicker is negative (inverted) frames: 31 flashes, 48 frames (12% of the video).
  Length: 1 frame x16, 2 frames x13, 3 frames x2.
  Half sit on a cut (the first 1-2 frames of the incoming view are shown in negative); half sit
  inside a shot (the view strobes normal -> negative -> normal).
- Other devices, used sparingly: a white silhouette flash (the foreground object filled solid white
  for single frames, alternating with normal frames, f274-f278); one motion-blur whip (~8 frames,
  f257-f264); black frames at the very start and end.
- Clean: no wipes, slides, spins or zoom transitions. Every transition is a hard event of 1-3 frames.
- Footage is mostly desaturated / black-and-white with small colour accents.
- Not tightly beat-synced (13 of 81 cuts/flashes within 1 frame of a music onset by my detector), so
  for talking-head work, drive the events from speech (phrase starts, key words), not from music.

## How it maps to the client's talking-head videos
- "Views" from one 4K camera: wide / medium / close crops, switched on phrase boundaries.
- Incoming view shown in negative for 1-2 frames on a switch; occasional in-shot negative strobe.
- Short bursts of quick view switches at energetic moments, then calm holds.
- Photosensitivity: the client is fine with more than 3 flashes per second (WCAG 2.3.1 would cap it
  at 3; the reference exceeds it in its bursts). Mention it once for client brands with sensitive audiences.
