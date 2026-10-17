# Client editing preferences

Rules the client has given while we built the first caption style. The video-editing skill follows
these on every video unless the client says otherwise for a specific project.

## Captions (Prime style)
- Keep every caption in the centre of the frame, also on talking-head footage, exactly like the
  reference reel. No words hidden behind the speaker.
- Words are white (or the style's pink) in a Difference blend, so the letters react to whatever
  moves behind them. That interaction is the point of the style.
- Close free font matches are fine (Inter; Pinyon Script with Imperial Script capitals).
- Small transcription errors in drafts are acceptable; don't spend extra time perfecting a draft
  transcript. Use the larger transcription model for final client videos.

## Timing
- Word timings come from forced alignment of the transcript to the audio (tools/align_words.py),
  never from Whisper's word timestamps alone: those were 100-400 ms early on the first test.
- Captions appear one frame before the word is spoken; sounds sit on the caption frames. Captions
  and sounds read the same plan file so they can't drift apart.
- Verify sync on the finished file (re-align the final audio, detect caption changes in the final
  video) before sending anything.
- Trim dead air before the first word to about 0.1 s.

## Sound
- Never use magical, glowy, sparkle, shimmer or chime sounds, anywhere.
- Caption sound = the client's own camera shutter (`lab/caption-style-01/assets/sfx/camera_shutter.wav`),
  the full sound with all its clicks; script (emphasis) words add a short hit.
- ALIGNMENT RULE: the biggest wave of the shutter (its loudest sample, ~194 ms into the file) lands
  exactly on the first frame a caption is visible. Not its first click, not the planned word time:
  the caption's first frame. The softer clicks before it may come just before.
- A sound plays only where a new caption appears: no whooshes, no run-ups, nothing in between.
- The mix limiter must use latency compensation (ffmpeg alimiter latency=1); without it every sound
  is 5 ms late.
- Don't spam: roughly half the captions get a sound (the first caption, the script words and a
  few others); vary pitch and level slightly so it doesn't sound like a loop.
- Verify on the FINAL video file that each biggest wave sits at 0 ms from its caption frame and that
  there are no other sounds (tools/plot_shutter.py --align makes the proof image).
- The effects must be clearly audible on a phone speaker; mix the voice to about -15 LUFS and the
  finished video to about -14 LUFS.

## Working style
- Show a first viewable version fast (within 20 minutes); polish afterwards.
