# MARSOPS launch teaser v3

15 s, 9:16 (1080x1920, 30 fps). A new film in the language of the client's site
(beyond-earth-mars-hero.wkconversions.chatgpt.site), after the client rejected the old-style v2 ("not the style, not
the colouring, definitely not the music; minimal music in the background, not old school"). Plan:
`storyboard/plan.md`.

## Build
- `build/prep_v3.py`: the starfield, the Mars base band and the Sun's eruption frames (`build/img/`, not in git).
- `build/render3.html`: one frame-pure `render(f)`; the site's 3D Mars in three.js (WebGL, rendered with SwiftShader in
  headless Chromium), canvas radiation streaks, SVG shield, HTML type. Served over http (ES modules):
  `python3 -m http.server 8765` in `build/`, then `BASE=http://127.0.0.1:8765 PAGE=render3.html node render.mjs full 0 449 4 4`.
- `build/sound/mix_v3.py picks.json out.wav`: music low in the background, effects on the picture's cues, -16 LUFS.

## Credits and rights
- Mars 3D model, suit prototype photo, logo: from the client's own site (provenance check in the sourcing report).
- Milky Way: ESO/S. Brunier (GigaGalaxy Zoom), CC BY 4.0.
- The Sun: NASA/SDO (AIA 304 A), SVS 3838, public domain.
- Mars base concept: NASA (PIA23302), public domain.
- Fonts: Manrope and Montserrat, SIL Open Font License.
- Music and sound effects: see the sourcing report (filled in with the final picks).
