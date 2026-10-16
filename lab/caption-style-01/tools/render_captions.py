#!/usr/bin/env python3
"""Burn word-by-word captions into a video in a data-driven caption style.

    python3 render_captions.py --video IN.mp4 --plan plan.json --style style.json --out OUT.mp4 \
        [--masks DIR] [--start S --end S] [--layout NAME] [--contact-sheet PNG] [--report JSON]

* style.json holds every visual decision: fonts (+ variation axes, OpenType features, tracking),
  sizes as fractions of frame height, blend presets (difference / normal / tinted difference),
  named layouts (positions, stacked blocks), entrance presets (cut / blur_in / glitch), timing
  rules and the digital-zoom presets. Nothing style-specific is hard-coded here.
* plan.json is the per-caption plan: {"layout": ..., "words": [{"text", "start", "end", "class",
  "blend", "entrance", "behind_subject", "position"?, "zoom"?, "stack"?}, ...]} (a bare list works).

Compositing happens per frame in float, display-referred sRGB (as After Effects / CSS mix-blend-mode):
    difference: out = B*(1-a) + |T - B|*a        normal: out = B*(1-a) + T*a
with a = anti-aliased text alpha x entrance opacity x (1 - subject mask) for behind_subject words.
Because the difference result depends on the footage under every letter, the letter colours change
frame by frame as things move behind them. The digital zoom (if any) is applied to the composite.

Decoding/encoding goes through ffmpeg pipes (rgb24 rawvideo in, H.264 yuv420p out, audio copied).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FFMPEG = os.environ.get('FFMPEG', 'ffmpeg')
FFPROBE = os.environ.get('FFPROBE', 'ffprobe')
AXIS_TAGS = {'weight': 'wght', 'width': 'wdth', 'optical size': 'opsz', 'slant': 'slnt', 'italic': 'ital'}


# --------------------------------------------------------------------------------------------- utils
def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def hex_rgb(value):
    """'#RRGGBB' or [r, g, b] (0-255) -> float32 array 0..1."""
    if isinstance(value, str):
        v = value.lstrip('#')
        return np.array([int(v[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
    return np.asarray(value, np.float32) / 255.0


def interp_curve(values, u):
    """Sample an evenly spaced keyframe list at u in [0, 1]."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    x = min(max(u, 0.0), 1.0) * (len(values) - 1)
    i = min(int(x), len(values) - 2)
    return float(values[i] + (values[i + 1] - values[i]) * (x - i))


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------ video IO
def probe(path):
    out = subprocess.run([FFPROBE, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info['streams'] if s['codec_type'] == 'video')
    rate = Fraction(v.get('avg_frame_rate') or v['r_frame_rate'])
    if rate == 0:
        rate = Fraction(v['r_frame_rate'])
    dur = float(v.get('duration') or info['format']['duration'])
    n = int(v['nb_frames']) if v.get('nb_frames', '').isdigit() else int(round(dur * rate))
    rot = 0
    for sd in v.get('side_data_list', []) or []:
        rot = int(sd.get('rotation', 0) or 0)
    w, h = int(v['width']), int(v['height'])
    if abs(rot) in (90, 270):
        w, h = h, w
    return dict(width=w, height=h, fps=rate, frames=n, duration=dur,
                has_audio=any(s['codec_type'] == 'audio' for s in info['streams']),
                color_space=v.get('color_space', ''), color_range=v.get('color_range', ''))


def open_decoder(path, start, dur, info, matrix):
    vf = []
    if info['color_space'] in ('bt709', 'bt470bg', 'smpte170m') or matrix:
        cm = matrix or info['color_space']
        vf.append(f"scale=in_color_matrix={cm}:in_range={'pc' if info['color_range'] == 'pc' else 'tv'}")
    cmd = [FFMPEG, '-v', 'error', '-nostdin']
    if start > 0:
        cmd += ['-ss', f'{start:.6f}']
    cmd += ['-i', path]
    if dur is not None:
        cmd += ['-t', f'{dur:.6f}']
    if vf:
        cmd += ['-vf', ','.join(vf)]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def open_encoder(out, src, start, dur, info, rcfg):
    w, h, fps = info['width'], info['height'], info['fps']
    m = rcfg.get('color_matrix', 'bt709')
    cmd = [FFMPEG, '-v', 'error', '-y', '-nostdin',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', f'{fps.numerator}/{fps.denominator}',
           '-i', '-']
    if info['has_audio']:
        if start > 0:
            cmd += ['-ss', f'{start:.6f}']
        cmd += ['-i', src]
        if dur is not None:
            cmd += ['-t', f'{dur:.6f}']
        cmd += ['-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'copy']
    cmd += ['-vf', f'scale=out_color_matrix={m}:out_range=tv,format={rcfg.get("pix_fmt", "yuv420p")}',
            '-c:v', rcfg.get('video_codec', 'libx264'), '-crf', str(rcfg.get('crf', 18)),
            '-preset', rcfg.get('preset', 'medium'), '-pix_fmt', rcfg.get('pix_fmt', 'yuv420p'),
            '-colorspace', m, '-color_primaries', m, '-color_trc', m, '-color_range', 'tv',
            '-movflags', '+faststart', out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ------------------------------------------------------------------------------------------ typesetting
class Rendered:
    """A word rendered once: float alpha (1x), baseline row and ink box inside the alpha."""

    def __init__(self, alpha, baseline, ink):
        self.alpha = alpha            # float32 HxW 0..1
        self.baseline = baseline      # row of the baseline inside alpha
        self.ink = ink                # (x0, y0, x1, y1) ink box inside alpha, exclusive end

    @property
    def ink_w(self):
        return self.ink[2] - self.ink[0]


class Typesetter:
    """Renders caption text per style class with PIL/raqm (kerning + features + variation axes),
    tracking applied per glyph, supersampled and cached."""

    def __init__(self, style, style_dir):
        self.style = style
        self.dir = style_dir
        self.ss = int(style.get('render', {}).get('supersample', 2))
        self._fonts = {}
        self._cache = {}

    def path(self, p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.dir, p))

    def font(self, path, size, axes=None):
        key = (path, round(size * 4) / 4, json.dumps(axes or {}, sort_keys=True))
        if key in self._fonts:
            return self._fonts[key]
        f = ImageFont.truetype(self.path(path), size=max(1, int(round(size))), layout_engine=ImageFont.Layout.RAQM)
        if axes:
            try:
                info = f.get_variation_axes()
            except Exception:
                info = []
            vals = []
            for a in info:
                name = a['name'].decode() if isinstance(a['name'], bytes) else str(a['name'])
                tag = AXIS_TAGS.get(name.lower(), name[:4].lower())
                v = axes.get(tag, axes.get(name, a.get('default')))
                vals.append(min(max(v, a['minimum']), a['maximum']))
            if vals:
                f.set_variation_by_axes(vals)
        self._fonts[key] = f
        return f

    @staticmethod
    def _ink_height(font, ch, features=None):
        size = font.size
        im = Image.new('L', (size * 3, size * 3), 0)
        ImageDraw.Draw(im).text((size, size * 2), ch, font=font, fill=255, anchor='ls', features=features)
        rows = np.nonzero(np.asarray(im).max(1) > 127)[0]
        return float(size * 2 - rows.min()) if len(rows) else float(size)

    def apply_case(self, text, cls):
        case = self.style['classes'][cls].get('case', 'as_is')
        if case == 'lower':
            return text.lower()
        if case == 'upper':
            return text.upper()
        if case == 'capitalize':
            return text[:1].upper() + text[1:]
        return text

    def render(self, text, cls, size_px, pad_px=0):
        """Return a Rendered for `text` in style class `cls` at font size `size_px` (output pixels)."""
        key = (text, cls, round(size_px, 2), pad_px)
        if key in self._cache:
            return self._cache[key]
        c = self.style['classes'][cls]
        ss = self.ss
        S = size_px * ss
        feats = c.get('features') or None
        main = self.font(c['font'], S, c.get('axes'))
        track = c.get('tracking_em', 0.0) * S
        stroke = int(round(c.get('stroke_em', 0.0) * S))
        text = self.apply_case(text, cls)
        asc, desc = main.getmetrics()
        cap_cfg = c.get('swash_capital')
        use_cap = bool(cap_cfg) and text[:1].isupper()
        cap_font = None
        if use_cap:
            xh = self._ink_height(main, 'x', feats)
            probe_f = self.font(cap_cfg['font'], 400)
            h0 = self._ink_height(probe_f, text[0])
            cap_font = self.font(cap_cfg['font'], 400 * cap_cfg.get('height_x_ratio', 2.5) * xh / h0)
            casc, cdesc = cap_font.getmetrics()
            asc, desc = max(asc, casc), max(desc, cdesc)
        width = int(main.getlength(text, features=feats) + abs(track) * len(text) + 2 * S
                    + (cap_font.getlength(text[0]) if cap_font else 0))
        height = int(asc + desc + 2 * S)
        im = Image.new('L', (width, height), 0)
        d = ImageDraw.Draw(im)
        margin = int(math.ceil(S))
        by = int(asc) + margin
        x = float(margin)
        body = text
        if cap_font is not None:
            d.text((x, by), text[0], font=cap_font, fill=255, anchor='ls', stroke_width=stroke, stroke_fill=255)
            x += cap_font.getlength(text[0]) * cap_cfg.get('advance_scale', 1.0) + track
            body = text[1:]
        ws = c.get('word_space_em', 0.0) * S

        def pen(i):
            # pen x of glyph i: kerned advance of the prefix + i x tracking (AE tracking / CSS
            # letter-spacing) + extra word space for every space before it
            pre = main.getlength(body[:i + 1], features=feats) - main.getlength(body[i], features=feats)
            return x + pre + i * track + body[:i].count(' ') * ws

        if abs(track) < 1e-6:
            # no tracking: draw whole words so joining scripts keep their connections
            i = 0
            for word in body.split(' '):
                if word:
                    d.text((pen(i), by), word, font=main, fill=255, anchor='ls', features=feats,
                           stroke_width=stroke, stroke_fill=255)
                i += len(word) + 1
        else:
            for i, ch in enumerate(body):
                if ch != ' ':
                    d.text((pen(i), by), ch, font=main, fill=255, anchor='ls', features=feats,
                           stroke_width=stroke, stroke_fill=255)
        shear = c.get('shear_deg', 0.0)
        if shear:
            k = math.tan(math.radians(shear))
            im = im.transform(im.size, Image.AFFINE, (1, k, -k * by, 0, 1, 0), resample=Image.BICUBIC)
        a = np.asarray(im)
        ys, xs = np.nonzero(a > 0)
        if len(xs) == 0:
            r = Rendered(np.zeros((1, 1), np.float32), 0, (0, 0, 1, 1))
            self._cache[key] = r
            return r
        pad = int(math.ceil(pad_px)) * ss + ss
        x0, y0 = int(xs.min()) - pad, int(ys.min()) - pad
        x1, y1 = int(xs.max()) + 1 + pad, int(ys.max()) + 1 + pad
        # align the crop to the supersampling grid
        x0 -= x0 % ss
        y0 -= (y0 - by) % ss
        x1 += (-x1 + x0) % ss
        y1 += (-y1 + y0) % ss
        crop = Image.new('L', (x1 - x0, y1 - y0), 0)
        crop.paste(im.crop((max(x0, 0), max(y0, 0), min(x1, im.width), min(y1, im.height))),
                   (max(0, -x0), max(0, -y0)))
        small = crop.reduce(ss) if ss > 1 else crop
        alpha = np.asarray(small, np.float32) / 255.0
        cols = np.nonzero(alpha.max(0) > 0.02)[0]
        rows = np.nonzero(alpha.max(1) > 0.02)[0]
        ink = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
        r = Rendered(alpha, (by - y0) / ss, ink)
        self._cache[key] = r
        return r


# ----------------------------------------------------------------------------------------- masks
class Masks:
    def __init__(self, folder, cfg):
        self.folder = folder
        self.cfg = cfg or {}
        self.pattern = self.cfg.get('pattern', 'm{frame:05d}.png')
        self.base = int(self.cfg.get('frame_base', 1))
        self._cache = {}

    def raw(self, frame):
        p = os.path.join(self.folder, self.pattern.format(frame=frame + self.base))
        if not os.path.exists(p):
            return None
        return np.asarray(Image.open(p).convert('L'))

    def get(self, frame, size):
        if frame in self._cache:
            return self._cache[frame]
        m = self.raw(frame)
        if m is None:
            out = None
        else:
            im = Image.fromarray(m)
            if im.size != size:
                im = im.resize(size, Image.BILINEAR)
            f = self.cfg.get('feather_px', 0)
            if f:
                im = im.filter(ImageFilter.GaussianBlur(f))
            a = np.asarray(im, np.float32) / 255.0
            lo, hi = self.cfg.get('threshold_lo', 0.0), self.cfg.get('threshold_hi', 1.0)
            out = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        self._cache = {frame: out}      # keep only the current frame
        return out

    def top_row(self, frame, x_band=(0.3, 0.7), q=10):
        """Top of the subject (row) in the central column band, robust to stray hairs."""
        m = self.raw(frame)
        if m is None:
            return None
        h, w = m.shape
        band = m[:, int(x_band[0] * w):int(x_band[1] * w)] > 127
        has = band.any(0)
        if not has.any():
            return None
        tops = band.argmax(0)[has]
        return float(np.percentile(tops, q)) / h


# -------------------------------------------------------------------------------------- the plan
def load_plan(path):
    p = load_json(path)
    if isinstance(p, list):
        p = {'words': p}
    return p


def frame_spans(units, fps, total_frames, t0_frame, timing):
    """Assign [first, last) absolute frame indices to each unit following the style's timing rules."""
    fr = float(fps)
    rnd = (lambda x: int(math.floor(x + 0.5))) if timing.get('frame_rounding', 'nearest') == 'nearest' else math.floor
    min_hold = max(1, rnd(timing.get('min_hold_s', 0.0) * fr))
    end_all = t0_frame + total_frames
    order = sorted(range(len(units)), key=lambda i: units[i]['start'])
    starts = {i: rnd(units[i]['start'] * fr) for i in order}
    if order and timing.get('first_unit_lead_in_s') is not None and units[order[0]]['start'] <= timing['first_unit_lead_in_s']:
        starts[order[0]] = 0
    spans = {}
    for n, i in enumerate(order):
        u = units[i]
        a = starts[i]
        nxt = None
        for j in order[n + 1:]:
            if units[j].get('stack') and units[j].get('stack') == u.get('stack'):
                continue                       # lines of the same stacked block overlap on purpose
            nxt = j
            break
        extra = timing.get('emphasis_extra_hold_s', 0.0) if u.get('class') == 'script' else 0.0
        if nxt is None:
            b = end_all if timing.get('last_unit_hold_to_clip_end', False) else rnd((u['end'] + timing.get('tail_after_end_s', 0) + extra) * fr)
        else:
            gap = units[nxt]['start'] - u['end']
            if timing.get('hold_until_next_start', True) and gap <= timing.get('max_gap_s', 0.0):
                b = starts[nxt]
            else:
                b = min(starts[nxt], rnd((u['end'] + timing.get('tail_after_end_s', 0) + extra) * fr))
            b = max(b, min(a + min_hold, starts[nxt]))
        spans[i] = [a, max(a + 1, b)]
    # lines of a stacked block all leave together with the block's last line
    groups = {}
    for i, u in enumerate(units):
        if u.get('stack'):
            groups.setdefault(u['stack'], []).append(i)
    for idx in groups.values():
        end = max(spans[i][1] for i in idx)
        for i in idx:
            spans[i][1] = end
    return [tuple(spans[i]) for i in range(len(units))]


# ------------------------------------------------------------------------------------------ layout
class Placed:
    def __init__(self, r, x, y):
        self.r, self.x, self.y = r, x, y      # alpha top-left in frame pixels


def resolve_layout(style, plan, args, W, H):
    name = args.layout or plan.get('layout')
    if not name:
        auto = style.get('layout_auto', {})
        name = auto.get('portrait' if H > W else 'landscape') or next(iter(style['layouts']))
    if name not in style['layouts']:
        sys.exit(f'layout {name!r} not in style ({", ".join(style["layouts"])})')
    return name, style['layouts'][name]


def place_units(units, spans, ts, style, layout, W, H, masks):
    """Compute where each caption sits (static per caption)."""
    sans_px = layout['sans_size_frac_of_height'] * H
    pad_px = max_blur_px(style, sans_px) * 3 + 2
    placed = [None] * len(units)
    stacks = {}
    for i, u in enumerate(units):
        if u.get('stack'):
            stacks.setdefault(u['stack'], []).append(i)
            continue
        cls = u.get('class', 'sans')
        pos_all = layout['positions'].get(u.get('position') or 'default') or layout['positions']['default']
        pos = pos_all.get(cls) or layout['positions']['default'][cls]
        size = sans_px * style['classes'][cls].get('size_rel_sans', 1.0) * pos.get('size_scale', 1.0)
        r = ts.render(u['text'], cls, size, pad_px)
        fit = pos.get('fit_width')
        maxw = layout.get('max_width_frac', 1.0) * W
        target = None
        if fit and r.ink_w < fit[0] * W:
            target = fit[0] * W
        elif fit and r.ink_w > fit[1] * W:
            target = fit[1] * W
        elif r.ink_w > maxw:
            target = maxw
        if target:
            size *= target / max(r.ink_w, 1)
            r = ts.render(u['text'], cls, size, pad_px)
        cx = pos.get('x', 0.5) * W
        x = cx - (r.ink[0] + r.ink[2]) / 2.0
        anchor = pos.get('anchor', 'baseline')
        if anchor == 'subject_top':
            tops = []
            if masks is not None:
                for f in range(spans[i][0], spans[i][1]):
                    t = masks.top_row(f)
                    if t is not None:
                        tops.append(t)
            base = (float(np.median(tops)) + pos.get('y_offset', 0.0)) if tops else pos.get('fallback_y', 0.5)
            y = base * H - r.baseline
            u['_subject_top'] = float(np.median(tops)) if tops else None
        elif anchor == 'ink_center':
            y = pos.get('y', 0.5) * H - (r.ink[1] + r.ink[3]) / 2.0
        else:
            y = pos.get('y', 0.5) * H - r.baseline
        placed[i] = Placed(r, int(round(x)), int(round(y)))
    for gid, idx in stacks.items():
        idx.sort(key=lambda i: units[i]['start'])
        st = layout['stack']
        lines = []
        for k, i in enumerate(idx):
            cls = units[i].get('class', 'sans')
            size = sans_px * style['classes'][cls].get('size_rel_sans', 1.0)
            lines.append(ts.render(units[i]['text'], cls, size, pad_px))
        # block coordinates: first line baseline at 0, first line ink left at 0
        pos = []
        first = lines[0]
        base = 0.0
        for k, r in enumerate(lines):
            align = st['line_align'][min(k, len(st['line_align']) - 1)]
            base += st['baseline_step_em'][min(k, len(st['baseline_step_em']) - 1)] * sans_px if k else 0.0
            if align == 'ink_left_of_first' or align == 'self':
                ix = 0.0
            elif align == 'ink_center_of_first':
                ix = (first.ink_w - r.ink_w) / 2.0
            else:
                ix = 0.0
            pos.append((ix - r.ink[0], base - r.baseline))
        bx0 = min(p[0] + r.ink[0] for p, r in zip(pos, lines))
        bx1 = max(p[0] + r.ink[2] for p, r in zip(pos, lines))
        by0 = min(p[1] + r.ink[1] for p, r in zip(pos, lines))
        by1 = max(p[1] + r.ink[3] for p, r in zip(pos, lines))
        ax, ay = st['anchor']['x'] * W, st['anchor']['y'] * H
        ox, oy = ax - (bx0 + bx1) / 2.0, ay - (by0 + by1) / 2.0
        for k, i in enumerate(idx):
            placed[i] = Placed(lines[k], int(round(pos[k][0] + ox)), int(round(pos[k][1] + oy)))
            if not units[i].get('entrance') or units[i]['entrance'] == 'auto':
                units[i]['entrance'] = st['entrances'][min(k, len(st['entrances']) - 1)]
    return placed, sans_px


def max_blur_px(style, sans_px):
    m = 0.0
    for e in style.get('entrances', {}).values():
        if e.get('type') == 'blur_in':
            m = max(m, max(e.get('blur_em', [0])) * sans_px)
        if e.get('type') == 'glitch':
            m = max(m, (e.get('rgb_split_em', 0) + e.get('slice_shift_em', 0)) * sans_px / 3)
    return m


# -------------------------------------------------------------------------------------------- zoom
def zoom_curve(units, spans, style, n_frames, t0_frame, fps):
    """Per-frame scale of the composite from the zoom events attached to captions."""
    z = style.get('zoom', {})
    s = np.ones(n_frames, np.float64)
    if not z.get('enabled', False):
        return s
    events = []
    for i, u in enumerate(units):
        name = u.get('zoom')
        preset = z.get('presets', {}).get(name) if name else None
        if preset:
            events.append((spans[i][0], preset))
    events.sort(key=lambda e: e[0])
    cur, ev, t_ev, s0 = 1.0, None, 0, 1.0
    k = 0
    fr = float(fps)
    for f in range(t0_frame + n_frames):
        while k < len(events) and events[k][0] <= f:
            ev, t_ev, s0 = events[k][1], events[k][0], cur
            k += 1
        if ev is not None:
            tau = max(ev.get('tau_s', 0.15) * fr, 1e-3)
            cur = s0 + (ev['to'] - s0) * (1 - math.exp(-(f - t_ev) / tau))
        if f >= t0_frame:
            s[f - t0_frame] = cur
    return s


def apply_zoom(img, s, origin):
    if abs(s - 1.0) < 1e-5:
        return img
    W, H = img.size
    ox, oy = origin[0] * W, origin[1] * H
    box = (ox - ox / s, oy - oy / s, ox + (W - ox) / s, oy + (H - oy) / s)
    return img.resize((W, H), Image.BICUBIC, box=box)


# --------------------------------------------------------------------------------------- compositing
def blend_region(frame, alpha, x, y, colour, mode, alpha_rgb=None):
    """Blend text with `alpha` (HxW, or HxWx3 when channels differ) at (x, y) into float frame."""
    H, W, _ = frame.shape
    h, w = alpha.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    if a.ndim == 2:
        a = a[..., None]
    B = frame[y0:y1, x0:x1]
    T = colour[None, None, :]
    if mode == 'difference':
        top = np.abs(T - B)
    elif mode == 'normal':
        top = np.broadcast_to(T, B.shape)
    elif mode == 'exclusion':
        top = T + B - 2 * T * B
    elif mode == 'screen':
        top = 1 - (1 - T) * (1 - B)
    elif mode == 'multiply':
        top = T * B
    else:
        raise ValueError(f'unknown blend mode {mode!r}')
    frame[y0:y1, x0:x1] = B * (1 - a) + top * a


def gaussian(alpha, sigma):
    if sigma <= 0.05:
        return alpha
    im = Image.fromarray(np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255.0


def shift(a, dx):
    if dx == 0:
        return a
    out = np.zeros_like(a)
    if dx > 0:
        out[:, dx:] = a[:, :-dx]
    else:
        out[:, :dx] = a[:, -dx:]
    return out


def entrance_state(ent, k, fps, sans_px, seed):
    """Opacity, blur sigma (px), per-channel x-offsets and slice shifts for frame k after entry."""
    st = dict(opacity=1.0, blur=0.0, hold=1, rgb=None, slices=None)
    if not ent or ent.get('type', 'cut') == 'cut':
        return st
    t = k / float(fps)
    dur = ent.get('duration_s', 0.0)
    if t >= dur:
        return st
    u = t / dur if dur > 0 else 1.0
    if ent['type'] == 'blur_in':
        st['opacity'] = interp_curve(ent.get('opacity', [1.0]), u)
        st['blur'] = interp_curve(ent.get('blur_em', [0.0]), u) * sans_px
    elif ent['type'] == 'glitch':
        st['hold'] = int(ent.get('footage_hold_frames', 1))
        decay = 1.0 - u
        split = ent.get('rgb_split_em', 0.0) * sans_px * decay
        if split >= 0.5:
            st['rgb'] = (int(round(split)), 0, -int(round(split)))
        n = int(ent.get('slice_count', 0))
        if n > 0 and ent.get('slice_shift_em', 0) > 0:
            rng = np.random.default_rng(seed * 1000 + k)
            st['slices'] = (n, rng.uniform(-1, 1, n) * ent['slice_shift_em'] * sans_px * decay)
    return st


def unit_alpha(p, st):
    a = p.r.alpha
    if st['blur'] > 0:
        a = gaussian(a, st['blur'])
    if st['slices'] is not None:
        n, offs = st['slices']
        a = a.copy()
        y0, y1 = p.r.ink[1], p.r.ink[3]
        edges = np.linspace(y0, y1, n + 1).astype(int)
        for j in range(n):
            a[edges[j]:edges[j + 1]] = shift(a[edges[j]:edges[j + 1]], int(round(offs[j])))
    return a * st['opacity']


# ----------------------------------------------------------------------------------------- sheets
def label_font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def contact_sheet(thumbs, path, cols=5):
    if not thumbs:
        return
    tw, th = thumbs[0][0].size
    lh = 44
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw + (cols + 1) * 8, rows * (th + lh) + (rows + 1) * 8), (22, 22, 22))
    f1, f2 = label_font(17), label_font(15)
    d = ImageDraw.Draw(sheet)
    for n, (im, t, f, words) in enumerate(thumbs):
        r, c = divmod(n, cols)
        x, y = 8 + c * (tw + 8), 8 + r * (th + lh + 8)
        d.text((x, y), f'{t:5.2f}s  f{f:03d}', fill=(255, 210, 0), font=f1)
        d.text((x, y + 22), (' / '.join(words) or '(none)')[:34], fill=(230, 230, 230), font=f2)
        sheet.paste(im, (x, y + lh))
    sheet.save(path)


# -------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--style', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--masks', help='folder of per-frame subject masks (white = subject) for behind_subject words')
    ap.add_argument('--start', type=float, default=0.0, help='render from this time (s)')
    ap.add_argument('--end', type=float, default=None, help='render up to this time (s)')
    ap.add_argument('--layout', help='layout variant from the style (default: plan["layout"] or by aspect ratio)')
    ap.add_argument('--contact-sheet', help='write a contact sheet PNG of the output (~3 frames per second)')
    ap.add_argument('--contact-per-second', type=float, default=3.0)
    ap.add_argument('--report', help='per-frame JSON report (default: OUT with .render.json)')
    args = ap.parse_args(argv)

    t_start = time.time()
    style = load_json(args.style)
    style_dir = os.path.dirname(os.path.abspath(args.style))
    plan = load_plan(args.plan)
    info = probe(args.video)
    W, H, fps = info['width'], info['height'], info['fps']
    fr = float(fps)
    t0 = max(0.0, args.start)
    t1 = min(info['duration'], args.end) if args.end else info['duration']
    t0_frame = int(round(t0 * fr))
    n_frames = max(0, min(info['frames'], int(round(t1 * fr))) - t0_frame)
    layout_name, layout = resolve_layout(style, plan, args, W, H)
    log(f'[render] {os.path.basename(args.video)} {W}x{H} @ {fr:.3f} fps, frames {t0_frame}..{t0_frame + n_frames - 1}, '
        f'style {style.get("name")!r} layout {layout_name!r}')

    units = [dict(u) for u in plan['words']]
    for u in units:
        u.setdefault('class', 'sans')
        u.setdefault('blend', 'difference')
        u.setdefault('entrance', 'auto' if u.get('stack') else 'cut')   # stacked lines take the layout's entrances
        u.setdefault('behind_subject', False)
    masks = Masks(args.masks, style.get('subject_mask')) if args.masks and os.path.isdir(args.masks) else None
    if any(u['behind_subject'] for u in units) and masks is None:
        log('[render] warning: behind_subject words but no --masks folder; they will be drawn in front')
    spans = frame_spans(units, fps, info['frames'], 0, style.get('timing', {}))
    ts = Typesetter(style, style_dir)
    placed, sans_px = place_units(units, spans, ts, style, layout, W, H, masks)
    zooms = zoom_curve(units, spans, style, n_frames, t0_frame, fps)
    origin = style.get('zoom', {}).get('origin', [0.5, 0.5])
    blends = style['blends']
    entrances = style.get('entrances', {})
    for i, u in enumerate(units):
        if u['blend'] not in blends:
            sys.exit(f'unknown blend preset {u["blend"]!r} for {u["text"]!r}')
        if u['entrance'] not in entrances:
            sys.exit(f'unknown entrance preset {u["entrance"]!r} for {u["text"]!r}')
        p = placed[i]
        log(f'  {u["text"]:>12s} [{u["class"]:6s}] frames {spans[i][0]:3d}-{spans[i][1] - 1:3d} blend={u["blend"]} '
            f'entrance={u["entrance"]} behind={u["behind_subject"]} zoom={u.get("zoom")} '
            f'ink {p.r.ink_w}px at x={p.x + p.r.ink[0]} baseline y={p.y + p.r.baseline:.0f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    dur = n_frames / fr
    dec = open_decoder(args.video, t0, dur, info, style.get('render', {}).get('color_matrix'))
    enc = open_encoder(args.out, args.video, t0, dur, info, style.get('render', {}))
    fsize = W * H * 3
    thumbs = []
    step = max(1, int(round(fr / args.contact_per_second))) if args.contact_sheet else 0
    report = []
    history = {}
    for k in range(n_frames):
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            log(f'[render] decoder ended early at frame {k}')
            break
        f = t0_frame + k
        history[f] = buf
        history.pop(f - 8, None)
        visible = [i for i in range(len(units)) if spans[i][0] <= f < spans[i][1]]
        states = {i: entrance_state(entrances[units[i]['entrance']], f - spans[i][0], fps, sans_px, i) for i in visible}
        hold = max([states[i]['hold'] for i in visible] + [1])
        src_f = f
        if hold > 1:
            first = min(spans[i][0] for i in visible if states[i]['hold'] == hold)
            src_f = max(first + ((f - first) // hold) * hold, t0_frame)
            if src_f not in history:
                src_f = f
        frame = np.frombuffer(history[src_f], np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        rec = dict(frame=f, t=round(f / fr, 4), zoom=round(float(zooms[k]), 5), source_frame=src_f, words=[])
        for i in sorted(visible, key=lambda i: (units[i].get('behind_subject') is not True, units[i]['start'])):
            u, p, st = units[i], placed[i], states[i]
            a = unit_alpha(p, st)
            if u['behind_subject'] and masks is not None:
                m = masks.get(src_f, (W, H))
                if m is not None:
                    hh, ww = a.shape
                    sub = np.zeros_like(a)
                    xs0, ys0 = max(p.x, 0), max(p.y, 0)
                    xs1, ys1 = min(p.x + ww, W), min(p.y + hh, H)
                    if xs0 < xs1 and ys0 < ys1:
                        sub[ys0 - p.y:ys1 - p.y, xs0 - p.x:xs1 - p.x] = m[ys0:ys1, xs0:xs1]
                    a = a * (1 - sub)
            b = blends[u['blend']]
            colour = hex_rgb(b.get('fill', '#FFFFFF'))
            a = a * b.get('opacity', 1.0)
            if st['rgb'] is not None:
                for c, dx in enumerate(st['rgb']):
                    sub = frame[..., c:c + 1].copy()
                    blend_region(sub, shift(a, dx), p.x, p.y, colour[c:c + 1], b['mode'])
                    frame[..., c] = sub[..., 0]
            else:
                blend_region(frame, a, p.x, p.y, colour, b['mode'])
            rec['words'].append(dict(id=u.get('id', i), text=u['text'], k=f - spans[i][0], opacity=round(st['opacity'], 3),
                                     box=[p.x + p.r.ink[0], p.y + p.r.ink[1], p.x + p.r.ink[2], p.y + p.r.ink[3]]))
        out = Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8))
        out = apply_zoom(out, float(zooms[k]), origin)
        enc.stdin.write(out.tobytes())
        report.append(rec)
        if step and k % step == 0:
            thumbs.append((out.resize(thumb_size(W, H)), f / fr, f, [w['text'] for w in rec['words']]))
        if k % 12 == 0 or k == n_frames - 1:
            el = time.time() - t_start
            log(f'[render] frame {k + 1}/{n_frames}  zoom {zooms[k]:.3f}  words {[w["text"] for w in rec["words"]]}  {el:.1f}s')
    dec.stdout.close()
    dec.wait()
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit('[render] ffmpeg encoder failed')
    rep_path = args.report or os.path.splitext(args.out)[0] + '.render.json'
    with open(rep_path, 'w') as fh:
        json.dump(dict(video=args.video, out=args.out, style=args.style, plan=args.plan, layout=layout_name,
                       width=W, height=H, fps=fr, first_frame=t0_frame,
                       units=[dict(text=u['text'], cls=u['class'], blend=u['blend'], entrance=u['entrance'],
                                   behind_subject=u['behind_subject'], zoom=u.get('zoom'), first=spans[i][0],
                                   last=spans[i][1] - 1, subject_top=u.get('_subject_top'))
                              for i, u in enumerate(units)],
                       frames=report), fh, indent=1)
    if args.contact_sheet:
        contact_sheet(thumbs, args.contact_sheet)
    log(f'[render] wrote {args.out} ({n_frames} frames) in {time.time() - t_start:.1f}s; report {rep_path}')
    return 0


def thumb_size(W, H, long_side=480):
    s = long_side / max(W, H)
    return int(round(W * s)), int(round(H * s))


if __name__ == '__main__':
    sys.exit(main())
