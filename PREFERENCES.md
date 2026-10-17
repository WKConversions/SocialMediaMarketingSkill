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
- Verify sync on the finished file before sending anything. Re-align the CLEAN voice track on the
  final timeline, not the final mix: sound effects in the mix shift the aligner by up to ~100 ms.
- Trim dead air before the first word to about 0.1 s.

## Sound
- Never use magical, glowy, sparkle, shimmer or chime sounds, anywhere.
- Caption sound = the client's own camera shutter (`lab/caption-style-01/assets/sfx/camera_shutter.wav`),
  the full sound with all its clicks; script (emphasis) words add a short hit.
- ALIGNMENT RULE: the biggest wave of the shutter (its loudest sample, ~194 ms into the file) lands
  exactly on the first frame a caption is visible. Not its first click, not the planned word time:
  the caption's first frame. The softer clicks before it may come just before.
- Sounds play only where a new caption appears: no whooshes, no run-ups, nothing in between.
- The mix limiter must use latency compensation (ffmpeg alimiter latency=1); without it every sound
  is 5 ms late.
- EVERY new caption gets the shutter, without exception (body words: the shutter; script words: the
  shutter plus a short hit). Vary pitch and level slightly per use so it doesn't sound like a loop.
- Verify on the FINAL video file that each biggest wave sits at 0 ms from its caption frame and that
  there are no other sounds (tools/plot_shutter.py --align makes the proof image).
- The effects must be clearly audible on a phone speaker; mix the voice to about -15 LUFS and the
  finished video to about -14 LUFS.

## Sound effects per moment (from the client's sound reference, lab/sfx-01/NOTES.md)
- caption appears: the client's camera shutter (see above)
- element pops up on screen (canvas cards, icons, numbers): click
- quick cut / view switch: flash, on the cut frame
- scene to scene (into/out of canvas mode, whip transitions): whoosh, peaking on the cut
- big text graphic pops up (canvas titles, big words): glitch, starting on its first frame
- build tension into a key point or reveal: riser, peaking on the moment

## Transitions and flashes
- Negative (inverted) flash frames, 1-3 frames, on view switches and as in-shot flicker.
- The client is fine with more than 3 flashes per second (no photosensitivity cap requested).

## Editing
- Respect the client's framing: they frame themselves correctly. The original framing is the base view;
  view switches use gentle centred crops only (1.08x, at most 1.15x), never a big push into the face.
  Punch-in on script words is small (about 4%).
- Pacing must be CONSTANT through the whole video: every pause longer than ~0.22 s becomes the same
  short pause (~0.18 s), inside clips and between clips (tools: lab/edit-04/tools/pacing.py). Hide the
  jump cuts with real footage, gentle view switches, 1-2 negative frames and a flash sound.
- Canvas moments for lists, numbers and steps the client says: services check off as they're named,
  numbers count up to the spoken value, titles glitch in on the word they come from.

## Working style
- Show a first viewable version fast (within 20 minutes); polish afterwards.
