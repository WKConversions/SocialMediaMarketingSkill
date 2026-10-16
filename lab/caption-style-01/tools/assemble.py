"""Assemble all measurements into analysis/reference_spec.json (+ cache/words_final.json)."""
import json, numpy as np
from words import WORDS
LAB = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01'
C = LAB + '/analysis/cache'
W, H, FPS = 1916, 1078, 24
reg = json.load(open(f'{C}/registration.json'))
colour = json.load(open(f'{C}/colourfit.json'))
blend = json.load(open(f'{C}/blendtest.json'))
edge = json.load(open(f'{C}/edgeprofile_all.json'))
zoom = json.load(open(f'{C}/zoom_inc.json'))
prime = json.load(open(f'{C}/prime_fit.json'))
theyc = json.load(open(f'{C}/they_edgeprofile.json'))
CUTS = [1, 9, 11, 17, 23, 37, 41, 50, 53, 70, 86, 100, 106, 111, 122, 132, 137, 148]
SUBCUTS = {3: 'LED colour jump, same locked camera setup (footage on 2s, continuous digital zoom)',
           5: 'LED colour jump, same setup', 7: 'LED colour jump, same setup',
           55: 'desk lighting change (blue->cyan), same setup', 57: 'desk lighting change (cyan->red), same setup',
           60: 'desk lights switch off, same camera setup; zoom direction reverses here (new clip of the same set-up)'}
def shot_of(f):
    return max(i for i, c in enumerate(CUTS) if c <= f) + 1
def zoom_ratio(f0, f1):
    r = 1.0
    for f in range(f0, f1):
        if str(f) in zoom: r *= zoom[str(f)][0]
        elif f == 52: r *= 1.0283
    return r

FONT = {'w06': 'script', 'w10': 'script', 'w15': 'script+sans', 'w20': 'script', 'w28': 'script'}
BLEND = {'w20': 'normal', 'w21': 'normal', 'w05': 'difference_pink', 'w06': 'difference_pink'}
MANUAL_BBOX_FIRST = {'w06': [622, 396, 1375, 668]}
words = []
for wid, txt, f0, f1, rep, box, model in WORDS:
    if wid == 'w15': continue
    r = reg[wid]
    frs = {f['frame']: f for f in r['frames']}
    bb0 = MANUAL_BBOX_FIRST.get(wid, frs[f0]['bbox'])
    bb1 = frs[f1]['bbox']
    if wid == 'w06':
        s = frs[f1]['scale']/frs[f0]['scale']; cx, cy = 958, 539
        bb1 = [round(cx + s*(bb0[0]-cx), 1), round(cy + s*(bb0[1]-cy), 1), round(cx + s*(bb0[2]-cx), 1), round(cy + s*(bb0[3]-cy), 1)]
    fc = FONT.get(wid, 'sans'); bk = BLEND.get(wid, 'difference')
    cuts_inside = [c for c in CUTS if f0 < c <= f1]
    subcuts_inside = [c for c in SUBCUTS if f0 < c <= f1]
    w = dict(id=wid, text=txt, font_class=fc, first_frame=f0, last_frame=f1, n_frames=f1-f0+1,
             t_start_s=round((f0-1)/FPS, 3), t_end_s=round(f1/FPS, 3), shot=shot_of(f0),
             entrance='cut', exit='cut' if f1 < 163 else 'end_of_video (hard end)',
             blend_key=bk,
             blend={'difference': 'difference', 'normal': 'normal', 'difference_pink': 'difference'}[bk],
             fill_rgb={'difference': [255, 255, 255], 'normal': [255, 255, 255], 'difference_pink': [226, 139, 139]}[bk],
             opacity=1.0,
             ink_bbox_first_frame=[round(float(v), 1) for v in bb0],
             ink_bbox_last_frame=[round(float(v), 1) for v in bb1],
             center_x_frac=round((bb0[0]+bb0[2])/2/W, 4), center_y_frac=round((bb0[1]+bb0[3])/2/H, 4),
             height_px=round(bb0[3]-bb0[1], 1), width_px=round(bb0[2]-bb0[0], 1),
             scale_last_over_first=round(frs[f1]['scale']/frs[f0]['scale'], 4),
             scale_track=[round(frs[f]['scale']/frs[f0]['scale'], 4) for f in range(f0, f1+1)],
             registration_score_mean=round(float(np.mean([frs[f]['score'] for f in range(f0, f1+1)])), 3),
             occluded_by_foreground=False,
             spans_cut=len(cuts_inside) > 0, cuts_inside=cuts_inside, lighting_jumps_inside=subcuts_inside)
    if wid in colour: w['effective_fill_rgb_measured'] = colour[wid]['C_rgb255']
    if wid in blend:
        o = blend[wid]['overall']
        w['blend_test'] = dict(edge_pair_fit_I_in_eq_a_plus_b_I_out=o['lin_a_b_rgb'], rmse_difference=o['rmse_diff'], rmse_normal=o['rmse_normal'],
                               alpha_if_difference=o['alpha_diff'], alpha_if_normal=o['alpha_normal'], corr_inside_vs_outside_luma=o['corr_in_out_luma'],
                               bg_luma_p5_p95=o['out_luma_range'])
    if wid in edge and edge[wid]:
        al = [v['alpha'] for v in edge[wid].values()]; sg = [v['sigma'] for v in edge[wid].values()]
        w['edge_profile'] = dict(alpha_per_frame_min_max=[round(min(al), 3), round(max(al), 3)], blur_sigma_px_median=round(float(np.median(sg)), 2))
    words.append(w)
# ---- They / don't / wanna block
Z = zoom_ratio(86, 98)
def norm(b):
    return [round(958 + (b[0]-958)/Z, 1), round(539 + (b[1]-539)/Z, 1), round(958 + (b[2]-958)/Z, 1), round(539 + (b[3]-539)/Z, 1)]
lay = {'They': dict(bbox_f098=[308, 352, 770, 656]), "don't": dict(bbox_f098=[303, 585, 557, 673]), 'wanna': dict(bbox_f098=[364, 683, 708, 747])}
for k in lay: lay[k]['bbox_at_entrance_zoom'] = norm(lay[k]['bbox_f098'])
json.dump(lay, open(f'{C}/they_layout.json', 'w'), indent=1)
def curve(name):
    c = theyc[name]; final = c['98']['alpha']; base = c['98']['sigma_px']
    out = {}
    for f in range(86, 100):
        al = c[str(f)]['alpha']/final; sg = c[str(f)]['sigma_px']
        out[f] = dict(opacity=round(max(0.0, min(al, 1.0)), 2), extra_blur_sigma_px=(round(float(np.sqrt(max(sg**2 - base**2, 0))), 1) if al > 0.3 else None))
    return out
they_entries = []
for name, fc, f0, txt in [('They', 'script', 86, 'They'), ("don't", 'sans', 90, "don't"), ('wanna', 'sans', 95, 'wanna')]:
    b0 = lay[name]['bbox_at_entrance_zoom'] if name == 'They' else norm(lay[name]['bbox_f098'])
    # bbox at its own first frame: zoom from 86 to f0
    z0 = zoom_ratio(86, f0)
    bf = [round(958 + (b0[0]-958)*z0, 1), round(539 + (b0[1]-539)*z0, 1), round(958 + (b0[2]-958)*z0, 1), round(539 + (b0[3]-539)*z0, 1)]
    they_entries.append(dict(id={'They': 'w15', "don't": 'w16', 'wanna': 'w17'}[name], text=txt, font_class=fc, first_frame=f0, last_frame=99,
        n_frames=99-f0+1, t_start_s=round((f0-1)/FPS, 3), t_end_s=round(99/FPS, 3), shot=shot_of(f0),
        entrance=None, exit='cut (shot cut at f100)', blend_key='difference', blend='difference', fill_rgb=[255, 255, 255], opacity=1.0,
        ink_bbox_first_frame=bf, ink_bbox_f098=lay[name]['bbox_f098'],
        center_x_frac=round((bf[0]+bf[2])/2/W, 4), center_y_frac=round((bf[1]+bf[3])/2/H, 4), height_px=round(bf[3]-bf[1], 1), width_px=round(bf[2]-bf[0], 1),
        scale_last_over_first=round(zoom_ratio(f0, 99), 4), occluded_by_foreground=False, spans_cut=False, cuts_inside=[], lighting_jumps_inside=[],
        entrance_curve=curve(name), part_of_block='They/don\'t/wanna'))
they_entries[0]['entrance'] = 'blur+fade in, 7 frames (f086-f092): opacity ~0.20,0.29,0.41,0.56,0.76,0.80,0.93 then 1.0; extra gaussian blur sigma ~3.9->0 px'
they_entries[1]['entrance'] = "blur+fade in, ~6 frames (f090-f095): opacity ~0.08,0.18,0.52,0.80,0.94,0.93 then 1.0; extra blur sigma ~6.3->0 px"
they_entries[2]['entrance'] = 'blur+fade in, ~2-3 frames (f095-f096/97): opacity ~0.55 then 1.0; extra blur sigma ~3.4 -> 1.5 -> 0 px (a <=15% very blurred ghost may already exist f092-f094; unreliable)'
words = words[:14] + they_entries + words[14:]
words.sort(key=lambda w: (w['first_frame'], w['id']))
json.dump(words, open(f'{C}/words_final.json', 'w'), indent=1)
print(len(words), 'words')
for w in words:
    print(f"{w['id']} {w['text']:12s} {w['font_class']:7s} f{w['first_frame']:03d}-f{w['last_frame']:03d} {w['blend_key']:16s} c=({w['center_x_frac']:.3f},{w['center_y_frac']:.3f}) h={w['height_px']} cut={w['spans_cut']} zoom={w['scale_last_over_first']}")
