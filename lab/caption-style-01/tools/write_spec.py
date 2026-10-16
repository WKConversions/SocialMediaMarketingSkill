import json
LAB = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01'
C = LAB + '/analysis/cache'
words = json.load(open(f'{C}/words_final.json'))
zsegs = json.load(open(f'{C}/zoom_segments.json'))
prime = json.load(open(f'{C}/prime_fit.json'))
pbins = json.load(open(f'{C}/prime_bins.json'))
lay = json.load(open(f'{C}/they_layout.json'))
font = json.load(open(f'{C}/fontfit.json'))
for w in words:
    w.pop('blend_key', None)
spec = {
 "source": {"file": LAB + "/ref/reference.mp4", "creator": "Toby de Boer (@tobydeboer), Instagram reel DeOv0aupBd_",
            "width": 1916, "height": 1078, "fps": 24, "frames": 163, "duration_s": 6.79,
            "frame_numbering": "fNNN = frame NNN, 1-based; time = (N-1)/24 s",
            "transcript": "i'm not even yet at my Prime / i'm not at a hundred yet / i'm still charging / They don't wanna see me Win / but i know in the end i'm winning and Regardless"},
 "method_summary": [
   "Shot cuts: consecutive-frame histogram distance + luma correlation + gradient-structure correlation (tools/cuts.py, cuts2.py), every candidate confirmed on contact sheets.",
   "Glyph masks: iterative two-model segmentation (pixel = B or 1-B with smooth B) on tight boxes; sans masks cleaned with an Inter Display ExtraBold rendering registered to the frame (tools/textmask.py, refmask.py, rendermask.py).",
   "Per-frame geometry: similarity registration (scale + translation) of each word's mask against every frame by normalised correlation of gradient magnitude (tools/register.py). Scores 0.8-0.99 inside a word's lifetime, <0.65 outside -> exact first/last frames.",
   "Background motion: ORB + RANSAC similarity between consecutive frames with caption boxes masked (tools/bgmotion.py).",
   "Blend tests: (a) edge pairs, pixel just inside a glyph vs its mirror just outside, fitted I_in = a + b*I_out per channel (difference -> a~1, b~-1; normal -> a~1, b~0) (tools/blendtest.py); (b) inpainted-background opacity fit (alpha_est.py); (c) chroma-flip test (chroma_test.py); (d) clean-plate tests for at my / Prime (prime.py); (e) 'un-difference' visual check: replacing glyph pixels by 1-I makes difference text disappear (spec_difference_evidence.png).",
   "Entrance animation of the They block: alpha profile across thick glyph edges, background read 10-12 px outside, erf fit -> opacity and blur sigma per frame (tools/edgeprofile.py)."],
 "shots": {
   "shot_cut_frames": [1, 9, 11, 17, 23, 37, 41, 50, 53, 70, 86, 100, 106, 111, 122, 132, 137, 148],
   "description": {"1": "dark trading room, LED strip (colour changes every 2 frames)", "9": "blurred chart (fast pan)", "11": "watch close-up (hand moves through frame)",
                   "17": "face, dark", "23": "bedroom with desk, man doing push-ups (static camera)", "37": "living room wide", "41": "man at window, rice fields",
                   "50": "blurred red/green candles", "53": "overhead desk (lights change blue/cyan/red, then off at f060)", "70": "man at monitors, dark (handheld)",
                   "86": "night market, passers-by", "100": "man at curved monitor, side view", "106": "close-up, hand at face", "111": "pool, arms open (bright sky)",
                   "122": "low angle, Burj Khalifa", "132": "foggy spire (frames held in pairs: f132=f133, f135=f136)", "137": "red neon room", "148": "penthouse lounge, bright window"},
   "lighting_jumps_not_counted_as_cuts": {"3": "LED colour jump, same locked set-up", "5": "LED colour jump", "7": "LED colour jump",
                   "55": "desk light colour change", "57": "desk light colour change", "60": "desk lights off, same set-up (zoom reverses here)"},
   "notes": "Several clips are on 2s (12 fps footage held for 2 frames: f001-f008, f053-f059, f086-f099); captions and the digital zoom still update every frame at 24 fps."},
 "typography": {
   "sans": {"class": "bold lowercase neo-grotesk, tight tracking, all lowercase (incl. 'i', \"i'm\" with a straight apostrophe)",
            "best_available_match": "Inter Display ExtraBold (IoU 0.84-0.91 against clean masks; Inter Display Bold 0.74-0.93, FreeSans/Liberation Bold 0.70-0.85)",
            "font_size_px_at_1078p": 108, "font_size_frac_of_height": 0.100,
            "tracking_em": -0.07, "tracking_range_em": [-0.09, -0.06],
            "x_height_px": 58, "ascender_height_px": 80, "baseline_y_px": 579, "baseline_y_frac": 0.537,
            "layout": "single line, horizontally centred on x = W/2 (ink-bbox centre 0.499-0.503 W), vertically: the ascender-to-baseline box is centred on H/2 (ascender top ~499 px, baseline ~579 px); words with descenders therefore sit with their ink centre slightly lower (0.51-0.52 H)",
            "font_fit_details": font},
   "script": {"class": "large high-contrast formal/copperplate script, ~20-25 deg slant, hairline joins, swash capitals (P, T, W, R) with big entry loops; lowercase 'hundred' has straight looped-less ascenders",
              "words": ["Prime", "hundred", "They", "Win", "Regardless"],
              "capitalisation": "4/5 script words get a swash capital (Prime, They, Win, Regardless); 'hundred' stays lowercase",
              "sizes_px_at_entrance": {"Prime": {"ink_h": 272, "ink_w": 753}, "They": {"ink_h": 272, "ink_w": 414}, "Win": {"ink_h": 249, "ink_w": 462},
                                       "Regardless": {"ink_h": 213, "ink_w": 642, "x_height": 63, "ascender": 131}, "hundred": {"ink_h": 133, "ink_w": 576, "x_height": 74, "ascender": 137}},
              "size_relation_to_sans": "swash capitals are ~3x the sans ascender height (250-272 px vs 80 px); script lowercase x-height is 1.1-1.3x the sans x-height (63-74 px vs 58 px)",
              "layout": "centred like the sans line; ink centre x 0.51-0.54 W (swash overhang pushes it right), y 0.49-0.52 H",
              "font_identification": "not identified; candidates to audition: a Didone copperplate script with swash caps (e.g. 'Kugile', 'Bodoni Script', 'Snell Roundhand', 'Pinyon Script', 'Great Vibes' are weaker matches)"},
   "see_also": "analysis/fonts_choice.json (separate font-matching step: Inter variable wght 750 opsz 32 ss07, 113.5 px, -0.075 em; Pinyon Script for the script) - consistent with the metrics measured here (x-height 58-59 px, tracking -0.07 em)",
   "outline_box_shadow": "none (no stroke, no box, no drop shadow, no glow; edges are 1-px anti-aliased, sigma ~0.5-0.7 px)"},
 "blend": {
   "default": "white text (#FFFFFF, 100% opacity) in Difference blend mode over the footage: out = |1 - B| = 1 - B per channel (Exclusion gives the identical result for pure white).",
   "why_it_looks_interactive": "every glyph shows the photographic negative of whatever is behind it, so a hand, watch, monitor, sky edge or LED strip moving behind a letter changes the letter's colour in real time (watch face -> navy letters; skin -> light blue; red LED -> cyan; teal monitor -> salmon; bright sky -> near-black). There is NO occlusion anywhere: the text is always on top; 'even' f011-f015 and 'They' f088-f095 only look occluded because the passing object is inverted inside the letters.",
   "per_word": {w['id']: {"text": w['text'], "blend": w['blend'], "fill_rgb": w['fill_rgb']} for w in words},
   "evidence": "spec_difference_evidence.png; edge-pair fits: difference words give b = -0.75..-1.1 with a = 0.89..0.97 (e.g. know a=(0.93,0.88,0.88) b=(-1.02,-0.99,-1.01); hundred b~-0.9; in the b~-0.96); normal words give b~0 (Win a=(0.97,0.98,1.0) b=(-0.04,-0.05,-0.2); but i a=(0.98,0.99,0.99) b=(0.01,0.01,-0.02)). Words over almost-black backgrounds (yet f017-022, not f041-043, at a, still, i'm f063/f137, winning, and) cannot separate the two models numerically (both predict white) but show inverted bright objects inside the glyphs where they exist (yet/i'm f060-064: inverted laptop inside 'e' and 'm'), and belong to the same difference style.",
   "measured_effective_white": "interior of difference text over black reads ~ (249,245,250) and the best |C-B| fits give C = 0.87-1.0; 'end' fits C ~ (213,222,227). Treat as white + the final grade/compression; optional: multiply text by 0.95 to match.",
   "exceptions": {"Win (f111-116) and 'but i' (f117-121)": "Normal blend, opaque white (they stay white over the bright sky)",
                  "at my (f023-027) and Prime (f028-036)": "Difference blend with a pink fill instead of white (see prime_solution)"}},
 "prime_solution": {
   "best_model": "Difference blend with a solid pink fill: out_c = |C_c - B_c|, C = (0.886, 0.545, 0.545) = RGB(226,139,139) = #E28B8B (at my: (0.87,0.605,0.555) = #DE9A8E; use #E28B8B for both)",
   "rmse_by_hypothesis_Prime_f028_vs_clean_plate_f027": {k: round(v['rmse'], 4) for k, v in prime['Prime'].items() if isinstance(v, dict) and 'rmse' in v},
   "rmse_by_hypothesis_at_my_f027_vs_plate_f028": {k: round(v['rmse'], 4) for k, v in prime['at my'].items() if isinstance(v, dict) and 'rmse' in v},
   "fitted_parameters": {k: {kk: vv for kk, vv in v.items() if kk != 'rmse'} for k, v in prime['Prime'].items() if isinstance(v, dict)},
   "spatial_check": pbins,
   "interpretation": "dark background -> light pink letters (P over the dark wall); bright background (white desk, window) -> |0.55-0.8| gives dark red/maroon and, where G/B exceed 0.55, dark teal-black strokes; this is exactly what is seen. A tint-times-inverse (T*(1-B), rmse 0.102), difference then red overlay (0.138), normal colour (0.232), exclusion (0.266), colour-blend (0.289), multiply (0.427) and screen (0.427) all fit worse. Fill is uniform (no gradient: per-region fits agree within ~0.04).",
   "plates": "Prime f028 vs f027 warped by the measured background similarity (s=0.9986); at my f027 vs f028; man doing push-ups and the overlapping other word excluded.",
   "images": ["spec_prime_Prime.png", "spec_prime_at_my.png"]},
 "digital_zoom": {
   "finding": "The captions are part of the composite that receives each shot's digital zoom: caption scale changes frame-by-frame by exactly the same factor as the footage (text vs background per-frame ratios agree within ~0.1-0.3%, e.g. f001->002 1.0278 vs 1.0257, f053->054 1.0197 vs 1.0197, f060->061 0.9594 vs 0.9595, f111->112 1.0244 vs 1.0245), and caption centroids move as if scaled about the frame centre (zoom fixed point measured at (941-966, 518-568)).",
   "renderer_rule": "Render captions over the footage first (with the blend), then apply the shot's zoom to the composite: scale about frame centre (W/2, H/2), s(k) = 1 + A*(1 - exp(-k/tau)), k = frames since segment start, tau ~ 3.5-5.5 frames.",
   "pattern": "Punch-in zooms (A = +11..+23%) start on the first frame of every emphasis/script word with a cut or new phrase (Prime f028, hundred f050, They f086, Win f111) and also on the very first frame (f001). Settle-out zooms (footage starts ~111-121% and eases back to 100%) on several other cuts (even/watch f011, yet f060, charging f070, know f122). Remaining shots drift +-0.1-0.3%/frame. 'Regardless' gets no punch (only -1.4% drift).",
   "segments": zsegs,
   "image": "spec_zoom.png"},
 "entrances_exits": {
   "default": "hard cut in and hard cut out on exact frame boundaries; the next word replaces the previous on the very next frame (no gaps, no overlaps) - 16 of 28 words start exactly on a shot cut",
   "fades": "No word fades except the stacked They/don't/wanna block. 'end' (f132-136) is NOT a fade: its opacity is constant (edge-profile alpha identical across frames; per-letter colours fit |C-B| with constant C ~0.85); the apparent fade-in is the blurred blue spire moving out from behind the letters (f132-133 letters over blue -> salmon, f135-136 over pale fog -> dark).",
   "they_block_entrance": "staggered blur-in: each line fades in while a gaussian blur shrinks to 0. They: f086-f092 (opacity 0.20,0.29,0.41,0.56,0.76,0.80,0.93,...1.0; extra blur sigma 3.9,3.7,3.7,3.2,2.4,1.8,1.0,...0 px). don't: f090-f095 (opacity 0.08,0.18,0.52,0.80,0.94,0.93 then 1.0; blur 6.3,5.6,4.3,3.2,2.4,1.6 -> 0). wanna: f095-f096/97 (0.55, 1.0; blur 3.4, 1.5, 0.9 -> 0). Stagger = 4-5 frames (They f086, don't f090, wanna f094/95). Opacity ramps roughly linearly (ease-out at the end); first frame already ~15-20%.",
   "they_block_exit": "hard cut at the shot cut f100 (all three lines together)",
   "image": "spec_they_block.png"},
 "glitch": {
   "finding": "There is no glitch effect. Per-channel registration of the text edges at the suspected frames gives R-G and B-G offsets below 0.5 px (f001 -0.12/-0.08 px, f003 -0.12/+0.27, f005 0.02/0.19, f060 0.00/0.01, f100 -0.37/-0.49 = same as non-glitch frames), no block displacement, no noise. What looks like an RGB glitch is the Difference blend inverting high-saturation objects behind the letters: the cyan LED strip shows red/orange inside 'i' and 'n' (f001-f008), the red LED shows cyan (f003-f004), a laptop/keyboard shows as a dark negative inside 'e' of yet and 'm' of i'm (f060-f064), the teal monitor shows salmon inside 'ee' of see with dark negative text lines (f100-f105). In addition, the opening frames carry the +11% punch-in zoom and the footage is on 2s with LED colour jumps every 2 frames, which reads as stutter.",
   "how_to_reproduce": "do nothing extra: render difference text over footage with saturated moving light sources; optionally hold footage frames in pairs (12 fps) for the opening stutter"},
 "they_block_layout": {
   "frames": "They f086-f099, don't f090-f099, wanna f095-f099 (hard cut out at f100)",
   "anchor": "left half of frame: block ink bbox centre at (0.28 W, 0.51 H) at f098, (0.30 W, 0.51 H) at the block's own start zoom (f086); the block is NOT centred like the other captions",
   "lines": lay,
   "geometry_at_entrance_zoom_px": {"They": "script, ink 376-790 x 372-644 (h 272, same size as Prime)", "don't": "sans, ink 371-599 x 580-659 (ascender-to-baseline 79 px = standard sans size)", "wanna": "sans, ink 426-734 x 668-725 (x-height 57 px = standard sans size)"},
   "alignment": "don't is left-aligned with the left edge of the They swash (x 303 vs 308 at f098); wanna is centred under They (centre x 536 vs 539) i.e. indented ~60 px; ragged composition",
   "leading": "very tight: don't ascenders (y 585) rise above They's x-height baseline (~600) and sit beside the y-descender; sans baseline-to-baseline 74 px at f098 (= 0.61 em, 66 px at entrance zoom); no gap between don't and wanna (10 px)",
   "blend": "all three Difference, white",
   "fade_order": "They -> don't -> wanna, each a blur+fade-in, starts f086 / f090 / f094-95"},
 "emphasis_rule": {
   "observed": "Script words: Prime, hundred, They, Win, Regardless = 5 of 32 spoken words (28 caption units; ~1 in 6), roughly one per spoken line, ~1 every 1.3 s. They are the payoff content word of the line (noun/number/verb/adverb, usually the last word of the phrase: Prime, hundred, Win, Regardless) or the antagonist pronoun that opens a line (They). Function words, 'i'/'i'm' and the main verb chain are never script. The last word of the video is script and is held longest (16 frames). Consecutive script words never occur back-to-back.",
   "rule_for_new_transcript": "1) Split the transcript into spoken lines/phrases. 2) In each line pick at most one word (two only for a long contrastive line, e.g. a pronoun at the start AND the payoff at the end): prefer the final content word that completes the claim (noun, number word, strong verb, or a sentence-final adverb like 'regardless'); a subject pronoun opening an 'us vs them' line may also be picked. 3) Target ~15-20% of words (about 1 per 1-1.5 s), never function words, never two adjacent words. 4) Always make the final word of the video script. 5) Script words: swash capital initial (keep lowercase for number words like 'hundred'), ~3x height, centred (or stacked block, see below), on-screen 9-16 frames, and start a punch-in zoom of +10-13% (exp ease tau ~4 frames) on their first frame when they start a shot. 6) Optional colour accent: the first emphasis phrase got a pink fill (#E28B8B, still difference) for the whole shot ('at my Prime').",
   "stacked_variant": "Use the left-anchored stacked block (script word + 2 sans lines with staggered blur-ins) for one short 3-word clause per video, starting on a shot cut ('They / don't / wanna').",
   "normal_blend_variant": "Use plain white Normal blend for the climax word and its follow-up when the shot is bright and celebratory ('Win', 'but i')."},
 "words": words,
 "renderer_recipe": [
   "1. Per word: draw the text as a white (#FFFFFF) mask, anti-aliased, no stroke/shadow. Sans: Inter Display ExtraBold stand-in, 108 px (0.100*H), tracking -0.07 em, lowercase, centred, baseline at 0.537*H. Script: formal swash script, capitals ~250-272 px tall, centred.",
   "2. Composite onto the footage with Difference: out = |fill - footage| (fill white -> 1-footage); pink #E28B8B fill for the 'at my Prime' shot; Normal white for 'Win'/'but i'.",
   "3. Hard-cut words in/out on exact frames; next word appears on the very next frame.",
   "4. Apply the shot's digital zoom to the composited frame (captions included), about the frame centre, exponential ease; punch-in +11..+13% (tau ~4 f) starting on each script word's first frame.",
   "5. Stacked block variant: left-anchored, They/don't/wanna with 4-5 frame staggered blur(sigma 4-6 px -> 0)+fade(0 -> 1) entrances over 5-7 frames.",
   "6. No glitch layer, no occlusion matte (optional feature only: the reference never occludes)."],
 "helper_images": ["spec_timeline.png", "spec_zoom.png", "spec_difference_evidence.png", "spec_prime_Prime.png", "spec_prime_at_my.png", "spec_they_block.png"],
 "tools": LAB + "/tools/",
 "confidence_notes": [
   "Font families are not identified; Inter Display ExtraBold is a geometric stand-in (IoU ~0.87).",
   "Opacity of words over near-black backgrounds cannot be distinguished between difference and normal numerically; they are assigned Difference by style consistency and by inverted bright objects where present.",
   "They-block opacity/blur curves come from edge profiles over a busy, moving background (passers-by): +-0.1 opacity, +-1 px sigma. wanna before f095 is uncertain.",
   "Absolute zoom level of each shot is unknown; only relative per-frame changes are measured (they are precise to ~0.2%)."]
}
json.dump(spec, open(LAB + '/analysis/reference_spec.json', 'w'), indent=1)
print('written', len(json.dumps(spec)))
