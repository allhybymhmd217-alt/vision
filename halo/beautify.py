#!/usr/bin/env python3
"""Art pass on anime.jpg in the style of the user's edit, rendered at 2x.

1. Resolution: the picture is enlarged 2x (Lanczos, mild unsharp mask) and
   everything below is drawn natively at that size; masks are found at 1x.
2. Hair colour and highlights on the left: the long hair there was backlit to
   white, with no shading, only drawn lines. It becomes dark hair like the crown:
   - where: wherever the picture has strand texture (gaps inside closed up, so
     wide clumps count), inside HAIR_POLY, sharp along the hand's drawn outline;
   - volume: each clump darkens towards its drawn lines and is lit through its
     middle; a little lighter towards the light and the hair's outer edge;
   - detail: the drawn lines redrawn crisp, fine hair fibres following the way
     the strands run (line integral convolution over a smooth structure-tensor
     flow), soft glossy streaks, and a rim light on the outer strands;
   - weighted by how light the hair is locally, so it blends into the dark hair.
   Elsewhere the highlights there were half pure white. Its
   strands are still in the original's top tones (and in its blue channel,
   which clips far less), so those tones are rolled off from LEFT_KNEE up to
   LEFT_TOP with a shoulder that spreads them out instead of flattening them,
   fading in from the left; bloom, bokeh and specks stay off that side.
3. Depth: a gentle S-curve, and local contrast on the mid-tones.
4. Background: the flat white turns into a twilight periwinkle, soft lavender
   near the light on the left and darker away from it, with the edit's print
   dots. Background = near-white areas touching the frame (the sky inside the
   halo included).
5. Colours: the edit's palette (same gradient map as recolor.py).
6. Light: a light bloom (weakest on the left), the halo band glowing with a
   lavender glow and made of fine strands like the user's halo (thin bright
   strands along the fitted ellipse, faint gaps between them, a few stray
   strands outside it), four-point stars, sparkle dust along the ring,
   soft bokeh orbs and light specks behind the figure, a glint on the cuff,
   a catchlight in the eye, a soft corner vignette and fine grain.

    python3 beautify.py anime.jpg edit.jpg source.jpg out.png
"""

import math
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import enlarge_halo as eh
import wings_halo as wh
from add_halo import ellipse_distance, screen_pattern, splat, star, vm
from recolor import load, palette

ss = wh.ss
S = 2                       # render scale
# all coordinates below are in 1x pixels of anime.jpg
LIGHT = (120.0, 480.0)      # the light on the left
BG_NEAR, BG_FAR = 0.72, 0.46  # background brightness near the light and far from it
LEFT_KNEE, LEFT_TOP = 0.60, 0.80  # the left's highlights roll off from the knee up to this
HOLE_AT = (1000, 60)        # a pixel of the sky inside the halo
RING = (818.7, -168.1, 513.3, 124.2, 51.8)   # ellipse fitted to the halo band's centre line
RING_HALF = 26.0            # half width of the band
DOTS = 0.05                 # print-dot strength
GLOW = np.array([0.78, 0.82, 1.00])
STARS = [  # x, y, vertical ray, horizontal ray, gain; on dark spots so they read
    (1032.0, 112.0, 80.0, 56.0, 1.4),    # the sky inside the ring
    (770.0, 70.0, 46.0, 32.0, 1.0),      # over the hair at the ring's left end
    (1170.0, 332.0, 36.0, 26.0, 0.9),    # off its right end
    (958.0, 236.0, 20.0, 14.0, 0.7),
    (884.0, 962.0, 22.0, 30.0, 0.8),     # glint on the cuff's top corner
]
EYE = (547.0, 506.0)
# the light, backlit hair on the left (1x outline; the face, hand and shirt collar
# stay outside it). Inside, the hair takes the dark hair's tones: HAIR_K * L^HAIR_G
HAIR_POLY = [(300, 0), (0, 0), (0, 1080), (130, 1075), (240, 975), (330, 990), (385, 975),
             (398, 940), (398, 850), (402, 828), (420, 808), (445, 785), (455, 740),     # the hand's outline
             (430, 690), (432, 620), (422, 540), (405, 470),                 # the face's left side
             (360, 380), (310, 250), (282, 120)]
# the drawn outline of the hand: the hair edge is sharp only along it
HAND_LINE = [(560, 722), (500, 740), (470, 760), (445, 785), (420, 808), (402, 828), (398, 850), (398, 925)]
# the hair itself is where the picture has strand texture (edge density), so it ends
# where the strands end and the sky begins, with no drawn edge there
HAIR_DENSITY = 0.012       # edge density that counts as strands
HAIR_CLOSE = 40             # px: wide strands with no lines inside the hair are hair too
SKY_MIN = 8000              # px: smaller gaps left between the strands are hair as well
# backlit dark hair: tone deep inside the hair / at its lit outer edge and towards the
# light (before the S-curve), rim light on the outer strands and its reach (px)
HAIR_BASE = (0.21, 0.30)
RIM, RIM_REACH = 0.55, 22.0
CLUMP_PX = 10.0             # px over which a clump darkens towards its drawn lines
CLUMP_DARK = 0.35           # line density that counts as fully in shadow
LINE_INK = 0.55             # how dark the drawn strand lines read
FIBRE, SHEEN = 0.06, 0.12   # fine hair texture, and its glossy streaks


def masks(L):
    """Background (near white touching the frame) and the halo band, at 1x."""
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


def up(img, size):
    """Lanczos enlargement of a float image (H, W) or (H, W, C)."""
    chans = img if img.ndim == 3 else img[..., None]
    out = np.stack([np.asarray(Image.fromarray(chans[..., c].astype(np.float32), "F").resize(size, Image.LANCZOS))
                    for c in range(chans.shape[-1])], -1).astype(np.float64)
    return out if img.ndim == 3 else out[..., 0]


def norm(a):
    return a / max(a.max(), 1e-9)


def halo_strands(shape, rng):
    """Strands along the halo ellipse: (inside the band, stray ones outside it)."""
    H, W = shape
    cx, cy, a, b, tilt = RING
    y1, x0 = int(360 * S), int(560 * S)                              # the halo's corner only
    yy, xx = np.mgrid[0:y1, x0:W].astype(np.float64)
    d, t, _, _ = ellipse_distance(xx, yy, cx * S, cy * S, a * S, b * S, math.radians(tilt))
    half = RING_HALF * S
    inner = np.zeros_like(d)
    for k in range(14):
        base = (-0.95 + 1.9 * (k + rng.uniform(0.2, 0.8)) / 14) * half
        off = base + rng.uniform(-0.18, 0.18) * half * np.sin(rng.uniform(2, 6) * t + rng.uniform(0, 6.3))
        wid = S * rng.uniform(0.6, 1.4)
        env = 0.35 + 0.65 * vm(t, rng.uniform(-1.0, 1.4), rng.uniform(0.5, 3.0))
        inner += rng.uniform(0.6, 1.0) * env * np.exp(-2.0 * ((d - off) / (0.5 * wid)) ** 2)
    stray = np.zeros_like(d)
    for k in range(6):
        side = 1.0 if k % 2 else -1.0
        base = side * rng.uniform(1.25, 2.1) * half
        off = base + rng.uniform(-0.3, 0.3) * half * np.sin(rng.uniform(1.5, 4) * t + rng.uniform(0, 6.3))
        wid = S * rng.uniform(0.6, 1.1)
        env = vm(t, rng.uniform(-0.8, 1.2), rng.uniform(18.0, 45.0))      # short arcs
        stray += rng.uniform(0.5, 0.9) * env * np.exp(-2.0 * ((d - off) / (0.5 * wid)) ** 2)
    full = lambda a_: np.pad(a_, ((0, H - y1), (x0, 0)))
    return full(np.clip(inner, 0, 1)), full(np.clip(stray, 0, 1))


def hair_fibres(Ld, region, rng, steps=30, sigma=90.0):
    """Fine hair texture: noise smeared along the direction the strands run (line
    integral convolution over the structure-tensor flow), zero mean, unit spread."""
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = Ld[y0:y1, x0:x1]
    g = ndimage.gaussian_filter(img, 1.5)
    ix, iy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
    jxx, jxy, jyy = (ndimage.gaussian_filter(a_, sigma) for a_ in (ix * ix, ix * iy, iy * iy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + 0.5 * np.pi   # along the strands
    vx, vy = np.cos(theta), np.sin(theta)
    noise = ndimage.gaussian_filter(rng.uniform(-1.0, 1.0, img.shape), 0.7)
    hh, ww = img.shape
    acc = noise.copy()
    for sign in (1.0, -1.0):
        py, px = np.mgrid[0:hh, 0:ww].astype(np.float64)
        dx, dy = sign * vx, sign * vy
        for _ in range(steps):
            px, py = np.clip(px + dx, 0, ww - 1), np.clip(py + dy, 0, hh - 1)
            nx = ndimage.map_coordinates(vx, [py, px], order=1)
            ny = ndimage.map_coordinates(vy, [py, px], order=1)
            flip = np.where(nx * dx + ny * dy < 0, -1.0, 1.0)           # keep heading the same way
            dx, dy = nx * flip, ny * flip
            acc += ndimage.map_coordinates(noise, [py, px], order=1)
    acc -= ndimage.gaussian_filter(acc, 6.0)                         # keep the fine fibres only
    acc /= acc.std() + 1e-9
    out = np.zeros_like(Ld)
    out[y0:y1, x0:x1] = acc
    return out


def main():
    picture, edit, source, output = sys.argv[1:5]
    P1 = load(picture)
    H1, W1 = P1.shape[:2]
    H, W = H1 * S, W1 * S
    rng = np.random.default_rng(3)
    bg1, band1 = masks(P1 @ wh.LW)

    # 1) resolution
    P = up(P1, (W, H))
    P = np.clip(P + 0.45 * (P - eh.gauss(P, 1.2)), 0.0, 1.0)
    L0 = P @ wh.LW
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    X, Y = xx / S, yy / S                                            # 1x coordinates
    hair_poly = Image.new("L", (W1, H1), 0)
    ImageDraw.Draw(hair_poly).polygon(HAIR_POLY, fill=255)
    hair_poly = up(np.asarray(hair_poly, np.float64) / 255.0, (W, H))
    gy, gx = np.mgrid[0:H, 0:W] / float(S)
    # sharp along the hand's drawn outline, soft everywhere else (strands cross the cheek,
    # and the hair thins out over the collar)
    line = Image.new("L", (W, H), 0)
    ImageDraw.Draw(line).line([(x * S, y * S) for x, y in HAND_LINE], fill=255, width=2)
    near = ss(28.0 * S, 10.0 * S, ndimage.distance_transform_edt(np.asarray(line) == 0))
    hair_poly = (1.0 - near) * eh.gauss(hair_poly, 12.0 * S) + near * eh.gauss(hair_poly, 1.5 * S)
    L1x = P1 @ wh.LW
    grad = np.hypot(ndimage.sobel(ndimage.gaussian_filter(L1x, 0.8), 0), ndimage.sobel(ndimage.gaussian_filter(L1x, 0.8), 1))
    present = ndimage.gaussian_filter(ndimage.gaussian_filter(grad, 14.0), 10.0) > HAIR_DENSITY
    r_ = HAIR_CLOSE
    disk = np.hypot(*np.mgrid[-r_:r_ + 1, -r_:r_ + 1]) <= r_
    present = ndimage.binary_closing(np.pad(present, r_ + 5, mode="edge"), structure=disk)[r_ + 5:-r_ - 5, r_ + 5:-r_ - 5]
    sky, n_sky = ndimage.label(~present)
    sizes = ndimage.sum(np.ones_like(L1x), sky, range(1, n_sky + 1))
    present = ~np.isin(sky, 1 + np.nonzero(sizes >= SKY_MIN)[0])
    present = ndimage.gaussian_filter(present.astype(np.float64), 6.0)   # a soft edge to the sky
    hair_poly = hair_poly * np.clip(up(present, (W, H)), 0.0, 1.0)
    bg_raw = up(bg1.astype(np.float64), (W, H)) > 0.5
    bg = bg_raw & (hair_poly < 0.5)                                  # blown-out hair is not sky
    # how much the sky treatment applies: soft, fading out wherever the hair fades in
    bgw = (eh.gauss(ndimage.binary_dilation(bg_raw, iterations=S).astype(np.float64), S)
           * ss(0.80, 0.97, L0) * (1.0 - hair_poly))
    band = up(band1.astype(np.float64), (W, H)) > 0.5

    # 2) highlights on the left: roll the top tones off with a spreading shoulder
    left = ss(0.62 * W1, 0.28 * W1, X)
    Ld = 0.45 * L0 + 0.55 * P[..., 2]                                # blue keeps the strands
    src = L0 + ss(LEFT_KNEE, 0.95, L0) * (Ld - L0)
    span = LEFT_TOP - LEFT_KNEE
    sh = np.where(src > LEFT_KNEE, LEFT_KNEE + span * (np.clip(src - LEFT_KNEE, 0, None) / (1.0 - LEFT_KNEE)) ** 1.25, src)
    notbg = 1.0 - bgw
    Lr = L0 + left * notbg * (sh - L0)

    # the light hair becomes backlit dark hair. Strand structure: the picture's own,
    # normalised locally so every strand reads. Tone: like the crown, a little lighter
    # towards the lit outer edge, with a rim light on the outer strands and gloss on
    # the brightest ones. How much applies depends on how light the hair is around each
    # pixel (local mean), so it blends into the dark hair smoothly
    strand = ss(0.55, 1.0, Ld)                                       # fill 1, lines 0
    # volume: each clump of hair between two drawn lines is rounded, dark by its lines
    # and lit through its middle; wide clumps catch more light than narrow ones
    # (a continuous field: the faint strokes are dashed, a distance to them is not)
    near_lines = eh.gauss(ss(0.02, 0.15, 1.0 - Ld), CLUMP_PX * S / 2.0)
    clump = 1.0 - ss(0.0, CLUMP_DARK, near_lines)
    edge_in = ndimage.distance_transform_edt(hair_poly > 0.5) / S     # px from the hair's edge
    lit = np.maximum(np.exp(-edge_in / 50.0), ss(0.42, 0.12, np.hypot(X - LIGHT[0], Y - LIGHT[1]) / W1))
    base = HAIR_BASE[0] + (HAIR_BASE[1] - HAIR_BASE[0]) * lit
    Lh = base * (0.45 + 0.80 * clump) * (0.55 + 0.45 * strand)
    # the drawn strand lines, crisp, and fine hair fibres with glossy streaks where a
    # clump catches the light
    lines = ss(0.03, 0.18, 1.0 - Ld)
    fib = hair_fibres(Ld, hair_poly > 0.3, rng)
    Lh = Lh * (1.0 - LINE_INK * lines) * (1.0 + FIBRE * fib)
    gloss = SHEEN * ss(0.55, 1.0, clump) * ss(0.6, 2.2, fib) * (1.0 - lines)
    Lh = 1.0 - (1.0 - Lh) * (1.0 - gloss)
    rim = RIM * strand ** 4 * np.exp(-edge_in / RIM_REACH)
    Lh = 1.0 - (1.0 - Lh) * (1.0 - rim)
    hair_w = hair_poly * ss(0.45, 0.75, eh.gauss(L0, 12.0 * S))
    Lr = Lr * (1.0 - hair_w) + hair_w * Lh

    # 3) depth and local contrast
    L1 = np.interp(Lr, [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.2, 0.47, 0.76, 1.0])
    mid = np.exp(-((L1 - 0.45) / 0.22) ** 2)
    L1 = np.clip(L1 + 0.18 * mid * (L1 - eh.gauss(L1, 8.0 * S)), 0.0, 1.0)

    # 4) twilight background
    r = np.hypot(X - LIGHT[0], Y - LIGHT[1]) / W1
    target = BG_NEAR - (BG_NEAR - BG_FAR) * ss(0.25, 1.1, r)
    L2 = L1 * (1.0 - bgw) + bgw * np.minimum(L1, target)

    # 5) colours, then the print dots in the darkened background and the mid-tones
    rgb = wh.apply_map(palette(edit, source), L2)
    midt = np.exp(-((L2 - 0.45) / 0.18) ** 2)
    # only where the background is darker: on light lavender the dots read as a checkerboard
    dots = DOTS * np.clip(bgw * ss(0.18, 0.42, 0.98 - target) + 0.35 * midt * (1.0 - bgw), 0.0, 1.0)
    rgb *= (1.0 + dots * (2.0 * screen_pattern(X, Y) - 1.0))[..., None]

    # 6) light
    hi = ss(0.80, 0.97, L2) * (1.0 - bgw) * (1.0 - 0.95 * left)
    bloom = 0.06 * eh.gauss(hi, 6.0 * S) + 0.06 * eh.gauss(hi, 24.0 * S)

    inner, stray = halo_strands((H, W), rng)
    core = eh.gauss(band.astype(np.float64), S)
    band_soft = eh.gauss(band.astype(np.float64), 2.0 * S)
    rgb *= (1.0 - 0.16 * band_soft * (1.0 - inner))[..., None]        # gaps between strands
    seen = np.clip(band_soft + bgw, 0.0, 1.0)                        # the hair covers the halo
    strands = np.clip(0.75 * inner * band_soft + 0.9 * stray * seen, 0.0, 1.0)

    halo = 0.5 * eh.gauss(core, 4.0 * S) + 0.42 * eh.gauss(core, 14.0 * S) + 0.32 * eh.gauss(core, 40.0 * S)
    halo *= 1.0 - 0.45 * band_soft                                   # the glow is mostly around the band
    halo += 0.5 * eh.gauss(strands, 2.5 * S)
    light = np.clip(bloom + halo, 0.0, 1.0)[..., None] * GLOW
    light = eh.screen(light, (0.30 * core)[..., None] * np.ones(3))     # white-hot band
    light = eh.screen(light, strands[..., None] * (0.7 + 0.3 * GLOW))

    sparkle = np.zeros((H, W))
    for x0, y0, lv, lh, g in STARS:
        sparkle += star(xx, yy, x0 * S, y0 * S, lv * S, lh * S, 0.9 * S, g)
    # sparkle dust along the ring and light specks in the darkened background
    ring_d = ndimage.distance_transform_edt(~band) / S
    ys, xs = np.nonzero((ring_d > 2) & (ring_d < 45) & ~band)
    n = 420
    pick = rng.choice(len(xs), n, replace=False)
    w_ = rng.uniform(0.3, 1.0, n) * np.exp(-ring_d[ys[pick], xs[pick]] / 25.0)
    dust = splat(np.zeros((H, W)), xs[pick] + rng.uniform(-1, 1, n), ys[pick] + rng.uniform(-1, 1, n), w_)
    ys, xs = np.nonzero(bg & (ring_d > 60) & (r > 0.42))           # away from the light only
    m = 110
    pick = rng.choice(len(xs), m, replace=False)
    specks = splat(np.zeros((H, W)), xs[pick].astype(float), ys[pick].astype(float), rng.uniform(0.2, 0.8, m))
    sp = 1.4 * norm(eh.gauss(dust, 0.7 * S)) + 0.5 * norm(eh.gauss(specks, 1.2 * S))
    sparkle = sparkle + sp + 0.6 * eh.gauss(sp, 3.0 * S)
    light = eh.screen(light, np.clip(sparkle, 0.0, 1.0)[..., None] * (0.55 + 0.45 * GLOW))

    # bokeh: soft orbs of light far behind the figure, in the darker background
    ok = bg & (ring_d > 50) & (r > 0.55) & (X > 40) & (Y > 30)
    ys, xs = np.nonzero(ok[::8, ::8])
    bok = np.zeros((H, W))
    placed = []
    for i in rng.permutation(len(xs)):
        bx, by = xs[i] * 8.0, ys[i] * 8.0
        rad = rng.uniform(9.0, 30.0) * S
        if any(math.hypot(bx - px, by - py) < rad + pr + 10 * S for px, py, pr in placed):
            continue
        placed.append((bx, by, rad))
        dist = np.hypot(xx - bx, yy - by)
        disc = ss(rad + 1.5 * S, rad - 1.5 * S, dist) * (0.7 + 0.3 * ss(0.6 * rad, rad, dist))
        bok += rng.uniform(0.07, 0.17) * disc
        if len(placed) >= 16:
            break
    bok = eh.gauss(bok, 0.8 * S) * eh.gauss(bg.astype(np.float64), 3.0 * S)
    light = eh.screen(light, np.clip(bok, 0.0, 1.0)[..., None] * GLOW)

    eye = np.exp(-((X - EYE[0]) ** 2 + (Y - EYE[1]) ** 2) / (2 * 1.6 ** 2))
    light = eh.screen(light, (0.75 * eye + 0.12 * norm(eh.gauss(eye, 5.0 * S)))[..., None] * np.ones(3))

    # a soft vignette at the far corners, away from the light, so the face leads
    vig = 1.0 - 0.18 * ss(0.55, 1.05, np.hypot((X - 560.0) / W1, (Y - 520.0) / H1) * 1.6) * ss(0.3, 0.8, r)
    out = eh.screen(np.clip(rgb * vig[..., None], 0.0, 1.0), light)
    out += rng.normal(0.0, 0.010, (H, W))[..., None]                 # fine grain
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(output)
    print(f"saved {output} ({W}x{H}), bokeh orbs {len(placed)}")


if __name__ == "__main__":
    main()
