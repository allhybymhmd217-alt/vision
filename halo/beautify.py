#!/usr/bin/env python3
"""Art pass on anime.jpg in the style of the user's edit.

1. Depth: a gentle S-curve on brightness.
2. Background: the flat white turns into a twilight periwinkle that darkens
   away from the light on the left, with the edit's print dots in it. The
   background is the near-white area that touches the frame (the sky inside the
   halo included); its edge pixels darken in proportion to how white they are.
3. Colours: the edit's palette (same gradient map as recolor.py).
4. Light: a soft bloom on the highlights, the halo band glowing white-hot with
   a lavender halo around it, four-point sparkles and sparkle dust along the
   ring, floating light specks in the background, a catchlight in the eye,
   and a soft vignette at the far corners.

    python3 beautify.py anime.jpg edit.jpg source.jpg out.png
"""

import math
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import enlarge_halo as eh
import wings_halo as wh
from add_halo import screen_pattern, splat, star
from recolor import load, palette

ss = wh.ss
LIGHT = (120.0, 480.0)      # the light on the left
DEPTH = 0.42                # how far the background darkens away from it
HOLE_AT = (1000, 60)        # a pixel of the sky inside the halo
DOTS = 0.07                 # print-dot strength
GLOW = np.array([0.78, 0.82, 1.00])
STARS = [  # x, y, vertical ray, horizontal ray, gain; on dark spots so they read
    (1032.0, 112.0, 80.0, 56.0, 1.4),    # the sky inside the ring
    (770.0, 70.0, 46.0, 32.0, 1.0),      # over the hair at the ring's left end
    (1170.0, 332.0, 36.0, 26.0, 0.9),    # off its right end
    (958.0, 236.0, 20.0, 14.0, 0.7),
]
EYE = (547.0, 506.0)


def masks(L):
    """Background (near white touching the frame) and the halo band."""
    white = L > 0.975
    lab, n = ndimage.label(white)
    sizes = ndimage.sum(white, lab, range(1, n + 1))
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, [i for i in border if i > 0 and sizes[i - 1] > 2000])
    hole = lab == lab[HOLE_AT[1], HOLE_AT[0]]
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    band = (ndimage.distance_transform_edt(~hole) < 110) & ~bg & (L > 0.86) & (xx > 780) & (yy < 340)
    band = ndimage.binary_opening(band, np.ones((3, 3)))
    lb, nb = ndimage.label(band)
    band = lb == np.argmax(ndimage.sum(band, lb, range(1, nb + 1))) + 1
    return bg, band


def main():
    picture, edit, source, output = sys.argv[1:5]
    P = load(picture)
    H, W = P.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rng = np.random.default_rng(3)
    L0 = P @ wh.LW
    bg, band = masks(L0)

    # 1) depth
    L1 = np.interp(L0, [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.2, 0.47, 0.76, 1.0])

    # 2) twilight background
    r = np.hypot(xx - LIGHT[0], yy - LIGHT[1]) / W
    target = 0.98 - DEPTH * ss(0.25, 1.1, r)
    bgw = eh.gauss(ndimage.binary_dilation(bg).astype(np.float64), 1.0) * ss(0.80, 0.97, L0)
    L2 = L1 * (1.0 - bgw * (1.0 - target))

    # 3) colours, then the print dots where the background darkened and in the mid-tones
    rgb = wh.apply_map(palette(edit, source), L2)
    mid = np.exp(-((L2 - 0.45) / 0.18) ** 2)
    dots = DOTS * np.clip(bgw * ss(0.02, 0.25, 1.0 - target) + 0.35 * mid, 0.0, 1.0)
    rgb *= (1.0 + dots * (2.0 * screen_pattern(xx, yy) - 1.0))[..., None]

    # 4) light
    hi = ss(0.80, 0.97, L2) * (1.0 - bgw)
    bloom = 0.10 * eh.gauss(hi, 6.0) + 0.12 * eh.gauss(hi, 24.0)
    core = eh.gauss(band.astype(np.float64), 1.0)
    halo = 0.55 * eh.gauss(core, 4.0) + 0.45 * eh.gauss(core, 14.0) + 0.35 * eh.gauss(core, 40.0)
    light = np.clip(bloom + halo, 0.0, 1.0)[..., None] * GLOW
    light = eh.screen(light, (0.6 * core)[..., None] * np.ones(3))        # white-hot band

    sparkle = np.zeros((H, W))
    for x0, y0, lv, lh, g in STARS:
        sparkle += star(xx, yy, x0, y0, lv, lh, 0.9, g)
    # dust along the ring and light specks floating in the darkened background
    ring_d = ndimage.distance_transform_edt(~band)
    ys, xs = np.nonzero((ring_d > 2) & (ring_d < 45) & ~band)
    pick = rng.choice(len(xs), 260, replace=False)
    w_ = rng.uniform(0.3, 1.0, 260) * np.exp(-ring_d[ys[pick], xs[pick]] / 25.0)
    dust = splat(np.zeros((H, W)), xs[pick] + rng.uniform(-.5, .5, 260), ys[pick] + rng.uniform(-.5, .5, 260), w_)
    far = bg & (ring_d > 60) & (target < 0.85)
    ys, xs = np.nonzero(far)
    pick = rng.choice(len(xs), 90, replace=False)
    specks = splat(np.zeros((H, W)), xs[pick].astype(float), ys[pick].astype(float), rng.uniform(0.2, 0.8, 90))
    sp = 1.4 * eh.gauss(dust, 0.7) / max(eh.gauss(dust, 0.7).max(), 1e-6) + 0.5 * eh.gauss(specks, 1.2) / max(eh.gauss(specks, 1.2).max(), 1e-6)
    sparkle = sparkle + sp + 0.6 * eh.gauss(sp, 3.0)
    light = eh.screen(light, np.clip(sparkle, 0.0, 1.0)[..., None] * (0.55 + 0.45 * GLOW))

    eye = np.exp(-((xx - EYE[0]) ** 2 + (yy - EYE[1]) ** 2) / (2 * 1.6 ** 2))
    light = eh.screen(light, (0.75 * eye + 0.12 * eh.gauss(eye, 5.0) / eh.gauss(eye, 5.0).max())[..., None] * np.ones(3))

    # a soft vignette at the far corners, away from the light, so the face leads
    vig = 1.0 - 0.18 * ss(0.55, 1.05, np.hypot((xx - 560.0) / W, (yy - 520.0) / H) * 1.6) * ss(0.3, 0.8, r)
    out = eh.screen(np.clip(rgb * vig[..., None], 0.0, 1.0), light)
    out += (rng.normal(0.0, 0.012, (H, W)))[..., None]                 # fine grain
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(output)
    print(f"saved {output} ({W}x{H})")


if __name__ == "__main__":
    main()
