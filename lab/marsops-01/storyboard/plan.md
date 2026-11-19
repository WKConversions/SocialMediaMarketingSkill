# MARSOPS launch teaser: plan

## Brief (from Karl, 2026-11-19)
- 15 s, vertical 9:16, 1080x1920, 30 fps. No voice-over.
- Leonardo da Vinci / renaissance style: hand-drawn concept sketches of space suits and space habitats come to life.
- The mission: protecting astronauts against radiation.
- Classic sketch lines + dark blue + white typography + subtle neon-blue accents. Serious and modern.
- Building music. Ends on "MARSOPS - Coming soon". Logo supplied (212x48 px raster: rebuilt as live type,
  measured match: Montserrat ExtraBold 800, tracking 0.10 em, IoU 0.80 against the supplied file).

Assumptions: copy in English (the brand line is English); no product claims beyond the brief (no numbers,
materials or features invented); music composed for the film (royalty-free), no stock track.

## Thesis
MARSOPS exists to protect astronauts from radiation, shown the way Leonardo would have worked it out:
one circle on dark blue paper becomes Mars under radiation, then the circle of a Vitruvian astronaut whose
suit gets a shield, then the dome of a shielded habitat, and finally the line under the MARSOPS name.

Allowed patterns:
- ink sketch lines with construction geometry (compass circle, square, proportion ticks), double-pass
  strokes with overshoot, hatching (Leonardo's drawing language, asked for by the brief);
- mirror-script margin notes in Italian, written right to left (Leonardo's own convention; short, real
  words: "i raggi", "lo scudo dell'uomo", "la casa su Marte");
- radiation drawn as fine straight rays with a travelling head (the threat the brief names);
- neon blue ONLY for protection: the suit shield, the habitat shield, rays stopping on a shield, and
  the line that becomes the end card (the brief's "subtle neon-blue accents");
- dark blue paper (the brief's colour), white Montserrat type; the em dash of "MARSOPS - Coming soon"
  is shown as the wordmark over "COMING SOON", not typed.

## Thread object
The circle. Compass circle -> Mars -> Vitruvian circle -> habitat dome -> flattened into the logo line.
Carry, don't duplicate: each scene inherits the circle and the shield from the one before.

## Beats (30 fps)
| # | Time | Meaning | Visual event | Line on screen |
|---|---|---|---|---|
| 1 | 0.0-4.0 | The threat | a compass draws a circle; it becomes Mars (hatching, craters); rays rain in from the upper left and strike the surface (Geiger clicks) | "On Mars, radiation / never stops." |
| 2 | 4.0-8.0 | The suit | the circle grows into a Vitruvian circle, a square, an astronaut sketched in the double pose; rays hit him until a neon shield is drawn around the suit; then the rays stop on it, the clicks fall silent | "Protecting / every astronaut," |
| 3 | 8.0-12.0 | The habitat | the circle's lower half sinks into the ground: a dome in cross-section, layered shell, two people inside; the shield is carried into the dome; rays stop on it | "from suit / to habitat." |
| 4 | 12.0-15.0 | The brand | the dome's shield flattens into a neon line; MARSOPS rises out of it; COMING SOON | MARSOPS / COMING SOON |

## Motion identity
- Personality: precise premium (serious, modern). Signature curve cubic-bezier(.22,1,.36,1); the pen
  draws on an ease-in-out (a hand accelerates and slows), lines never bounce.
- Duration palette (frames): quick 8, standard 14, slow 30.
- Entrance pattern: lines are drawn (never faded in); type rises out of a mask by one line height.
- Camera: breathes only (1.00 -> 1.03 per beat, around the circle's centre), no pans or zooms on changes.
- Transitions: the circle itself scales and changes role; nothing cuts.

## Sound cue sheet
- Music: composed (music_make.py, custom "cinematic-pulse" vibe), 120 BPM so bars fall every 2 s:
  D minor, i-VI-III-VII; energy 0 at 0 s, 1 at 4 s (suit), 2 at 8 s (habitat, riser into it), 3 at 10 s,
  "end" at 12 s (the logo), ringing out.
- Geiger clicks on every ray strike before the shield closes (beats 1-2); none after.
- A soft shimmer when the suit's shield closes (7.2 s).
- Riser 10.4 -> 12.0, impact at 12.0 (the line becomes the logo).
- Master -14 LUFS (social), true peak under -1 dBTP.
