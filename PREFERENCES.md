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

## Music
- When the client supplies a track: its first audible sample lands on frame 0 (trim the file's leading
  silence; measure the offset over the whole mix, not the first second). It must be CLEARLY audible:
  about 6-7 dB under the voice while the client speaks (measured on stems), gentle ducking (ratio ~2.5),
  up in pauses and on the end card, fade out over the last ~1.2 s. -13 dB plus heavy ducking was inaudible.
- Approved tracks are saved in `social-video-editor/music/library.json` with their start offset, level,
  ducking and when they suit (and when not). Use one with `edit_video.py ... --music NAME`, only on
  videos its `suits` list fits. The level follows the part of the track the video actually uses (a
  track's intro can be quieter than the rest), so a short video gets the same ~7 dB under the voice as
  the approved edit. The audio files are not in git (the repo is public); they live in
  `social-video-editor/assets/music/`.
- Saved so far: `tyler-gone-gone` (GONE, GONE / THANK YOU instrumental) for storytelling and
  serious-but-positive promos, approved on the Brooklyn Bridge -> WKC story.

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

## Covers (story style, scripts/cover.py)
- Two centred lines in a heavy geometric sans (Montserrat): the setup in white, the payoff in red (#EA1B23), on a
  dark, moody backdrop with a dark fade under the chin (reference: "How did they / gain trust").
- Pick a moment where the client is explaining or neutral, eyes open and on the camera.
- Keep the client's own framing: don't zoom in on the head. Keep the face natural: no coloured or heavily graded
  skin; the mood comes from the dark backdrop, vignette and fade.
- Deliver the 9:16 reel cover and the 3:4 grid view.

## Working style
- Show a first viewable version fast (within 20 minutes); polish afterwards.
