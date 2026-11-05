"""Féau's construction series (Musée Carnavalet album PH9114, CC0): four albumen prints of the tower rising, all shot
from the same Trocadéro tower. Cut each print out of its blue album mount, scale to one width and line them up on the
tower itself, so a cross-dissolve between them shows the tower growing in place:
  - horizontally the legs already match (within ~20 px at 1680 px wide);
  - vertically the prints were trimmed differently: the lower edge of the first platform's deck sits 687 px above the
    bottom in March 1888, 626 px in November 1888 and 610 px in March 1889 (measured: strongest dark-to-light edge in
    the centre columns), so the later prints move up by 61 and 77 px (December 1887 has no platform; its legs line up
    with March 1888 as cut).
Writes broll/prepared/feau_<date>.png, all the same size, and feau_registration.json.
Usage: python3 -I prep_feau.py
"""
import json

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
B = "/home/user/SocialMediaMarketingSkill/lab/edit-10/broll"
# print (inside the blue mount) as found by the warm-pixel test, and how far each moves up to match March 1888
PRINTS = {"feau_1887-12-12_four_legs": ((86, 83, 1800, 4186), 0),
          "feau_1888-03-15_first_platform": ((102, 61, 1838, 4193), 0),
          "feau_1888-11-14_above_second_platform": ((104, 74, 1819, 4168), 61),
          "feau_1889-03-12_top_section": ((92, 76, 1828, 4187), 77)}
W, INSET = 1680, 12


def main():
    crops = {}
    for p, (b, up) in PRINTS.items():
        im = Image.open(f"{B}/photos/{p}/source.jpg").convert("RGB")
        c = im.crop((b[0] + INSET, b[1] + INSET, b[2] - INSET, b[3] - INSET))
        crops[p] = (c.resize((W, round(c.height * W / c.width)), Image.LANCZOS), up)
    Hc = max(c.height for c, _ in crops.values()) + max(up for _, up in crops.values())
    tops = {p: Hc - c.height - up for p, (c, up) in crops.items()}            # bottom-aligned, then moved up
    y0 = max(tops.values())                                                    # common area: no empty borders
    y1 = min(tops[p] + c.height for p, (c, _) in crops.items())
    info = {}
    for p, (c, up) in crops.items():
        canvas = Image.new("RGB", (W, Hc))
        canvas.paste(c, (0, tops[p]))
        canvas.crop((0, y0, W, y1)).save(f"{B}/prepared/{p}.png")
        info[p] = dict(mount_box=PRINTS[p][0], moved_up_px=up)
    json.dump(dict(width=W, height=y1 - y0, prints=info), open(f"{B}/prepared/feau_registration.json", "w"), indent=1)
    print("registered", len(crops), "prints,", W, "x", y1 - y0)


if __name__ == "__main__":
    main()
