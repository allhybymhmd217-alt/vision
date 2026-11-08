#!/usr/bin/env python3
"""Paint a floating "angel / dead" halo above the character's head.

Two looks are available:

* ``clean`` (default): a thin glowing tube with a violet / ice-blue chromatic
  split and a navy halftone edge;
* ``energy``: a large ring of light in the spirit of the reference
  edit the user picked - a bright band made of several luminous strands that
  bunch up at the front-left and the right end and fly off as wisps at the
  tips, a strong white-to-periwinkle bloom, four-point sparkle stars and
  glitter specks.

Both are built to sit inside the artwork: the ellipse is flattened by
perspective and tilted with the head (crown leaning to the upper left), the
colours come from the picture's own grading (navy, lavender, neutral white),
and the widest glow is printed with the picture's own halftone screen
(45-degree dot lattice, 0.1801/0.1802 cycles/px), anchored to the original
picture so the new dots land on the existing grid.

Only numpy and Pillow are needed:

    python3 add_halo.py extended.png final.png [--scale 2] [--style energy]
"""

import argparse
import math

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Halo geometry (pixels, for the 1200x1184 source)
# --------------------------------------------------------------------------
# Defaults are for the zoomed-out canvas made by zoom_out.py (1440x1456), where
# the original picture sits at (120, 236). Geometry is in canvas pixels at 1x:
# centre x, centre y, half width, half height, tilt (negative = right end
# higher, following the head) and band width.
STYLE_DEFAULTS = {
    # ~20% larger than the reference edit's halo; its front-left arc sits where
    # the reference has it, just above the lace crown
    "energy": dict(cx=563.0, cy=351.0, a=360.0, b=70.0, tilt=-17.0, width=24.0),
    "clean": dict(cx=575.0, cy=210.0, a=285.0, b=56.0, tilt=-12.0, width=9.5),
}
SCREEN_ORIGIN = (120.0, 236.0)  # original picture's top-left on the canvas

# sparkle stars of the energy look, placed relative to the ring:
# (ellipse angle in degrees, outward offset px, vertical ray, horizontal ray, gain)
STARS = [
    (212.0, 38.0, 106.0, 78.0, 2.30),  # big star over the back-left arc
    (150.0, 52.0, 42.0, 32.0, 1.45),   # small star under the front-left arc
    (-38.0, 0.0, 124.0, 30.0, 1.70),   # long vertical flare where the far arc turns
    (52.0, 30.0, 26.0, 19.0, 1.05),    # tiny glint off the front-right
]

# --------------------------------------------------------------------------
# Halftone screen measured from the source: two cosine gratings at +/-45 deg.
# Frequencies and phases were fitted on the dotted hair areas; the phase is
# consistent across the whole picture, so it is a single global screen.
# --------------------------------------------------------------------------
SCREEN_F1 = (0.1801, 0.1802)
SCREEN_F2 = (0.1801, -0.1802)
SCREEN_PH1 = math.radians(50.8)
SCREEN_PH2 = math.radians(-2.1)

# --------------------------------------------------------------------------
# Palette (sRGB 0..1), built from colours sampled in the picture
# (navy ~ (72, 77, 110), lavender ~ (173, 177, 200), neutral white highlights)
# --------------------------------------------------------------------------
WHITE = np.array([1.00, 1.00, 1.00])
TUBE_EDGE = np.array([0.80, 0.84, 1.00])
INNER = np.array([0.86, 0.90, 1.00])
MID = np.array([0.66, 0.72, 1.00])
HAZE = np.array([0.50, 0.56, 0.92])
VIOLET = np.array([0.70, 0.60, 1.00])
ICE = np.array([0.62, 0.90, 1.00])
NAVY = np.array([0.20, 0.23, 0.42])
GLOW_1 = np.array([0.90, 0.93, 1.00])   # tight bloom, almost white
GLOW_2 = np.array([0.74, 0.80, 1.00])   # pale periwinkle
GLOW_3 = np.array([0.56, 0.63, 0.98])   # lavender-blue
GLOW_4 = np.array([0.42, 0.48, 0.88])   # wide haze, printed as dots
NAVY_DEEP = np.array([0.17, 0.19, 0.36])  # backdrop shade behind the energy ring


def gaussian_blur(img, sigma):
    """Gaussian blur of a 2-D float array through the FFT, with reflect padding."""
    if sigma <= 0:
        return img.copy()
    pad = int(math.ceil(3 * sigma))
    p = np.pad(img, pad, mode="reflect")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    kernel = np.exp(-2.0 * math.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(p) * kernel, s=p.shape)
    return out[pad:pad + img.shape[0], pad:pad + img.shape[1]]


def ellipse_distance(xx, yy, cx, cy, a, b, tilt, seeds=48, iters=6):
    """Exact signed distance to a rotated ellipse, plus the curve parameter t
    of the nearest point (coarse search, then Newton refinement)."""
    c, s = math.cos(tilt), math.sin(tilt)
    dx, dy = xx - cx, yy - cy
    px = dx * c + dy * s          # along the major axis
    py = -dx * s + dy * c         # along the minor axis (+ = front of the ring)

    best_t = np.zeros_like(px)
    best_d = np.full_like(px, np.inf)
    for t0 in np.linspace(-math.pi, math.pi, seeds, endpoint=False):
        d2 = (px - a * math.cos(t0)) ** 2 + (py - b * math.sin(t0)) ** 2
        closer = d2 < best_d
        best_d = np.where(closer, d2, best_d)
        best_t = np.where(closer, t0, best_t)

    t = best_t
    k = a * a - b * b
    for _ in range(iters):
        ct, st = np.cos(t), np.sin(t)
        f = k * ct * st - a * px * st + b * py * ct
        fp = k * (ct * ct - st * st) - a * px * ct - b * py * st
        fp = np.where(np.abs(fp) < 1e-6, 1e-6, fp)
        t = t - np.clip(f / fp, -0.2, 0.2)

    dist = np.hypot(px - a * np.cos(t), py - b * np.sin(t))
    inside = (px / a) ** 2 + (py / b) ** 2 < 1.0
    return np.where(inside, -dist, dist), t, px, py


def screen_pattern(xx, yy):
    """The picture's halftone screen: 1 at its light-dot centres, 0 at its ink-dot centres."""
    return 0.5 + 0.25 * (
        np.cos(2 * math.pi * (SCREEN_F1[0] * xx + SCREEN_F1[1] * yy) + SCREEN_PH1)
        + np.cos(2 * math.pi * (SCREEN_F2[0] * xx + SCREEN_F2[1] * yy) + SCREEN_PH2)
    )


def halftone(g, pattern, soft=0.3):
    """Turn a smooth 0..1 intensity into dot coverage on the given screen
    (0 stays empty, 1 is solid, the dots grow in between)."""
    g = np.clip(g, 0.0, 1.0)
    return np.clip((pattern - 1.0 + g * (1.0 + soft)) / soft, 0.0, 1.0)


def norm(x):
    m = x.max()
    return x / m if m > 0 else x


def screen_blend(base, color, alpha):
    return 1.0 - (1.0 - base) * (1.0 - color[None, None, :] * alpha[..., None])


def multiply_toward(base, color, alpha):
    return base * (1.0 - alpha[..., None] + alpha[..., None] * color[None, None, :])


def over(base, color, alpha):
    """Normal blend; color may be a single colour or a per-pixel image."""
    if color.ndim == 1:
        color = color[None, None, :]
    return base * (1.0 - alpha[..., None]) + color * alpha[..., None]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ring_coverage(dist, width):
    """Anti-aliased coverage of a line of the given width centred on dist == 0."""
    return np.clip(width / 2.0 - np.abs(dist) + 0.5, 0.0, 1.0)


def render_clean(src, cx, cy, a, b, tilt_deg, core_w, screen_origin=(0.0, 0.0), scale=1.0, seed=7):
    """Paint the halo on `src`. Geometry is given in original-resolution pixels;
    `scale` renders it natively on an upscaled canvas (crisper ring)."""
    H, W = src.shape[:2]
    S = float(scale)
    # geometry in output pixels (pixel centres map as (x + 0.5) * S - 0.5)
    cx, cy = (cx + 0.5) * S - 0.5, (cy + 0.5) * S - 0.5
    a, b, core_w = a * S, b * S, core_w * S

    # work only on the region the halo can reach
    reach = 3 * 42.0 * S
    x0, x1 = int(max(cx - a - reach, 0)), int(min(cx + a + reach, W))
    y0, y1 = int(max(cy - a - reach, 0)), int(min(cy + 9 * b + reach, H))
    full = src
    src = src[y0:y1, x0:x1]
    h, w = src.shape[:2]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)

    tilt = math.radians(tilt_deg)
    vx, vy = -math.sin(tilt), math.cos(tilt)  # unit vector towards the front arc

    dist, t, px, py = ellipse_distance(xx, yy, cx, cy, a, b, tilt)
    front = np.sin(t)                          # +1 front centre, -1 back centre
    depth = 0.5 + 0.5 * front                  # 0 far arc .. 1 near arc
    width = core_w * (0.82 + 0.30 * depth)     # nearer arc reads a touch thicker
    # slow shimmer along the ring so the light does not look like a vector stroke
    shimmer = 1.0 + 0.06 * np.sin(3.0 * t + 1.3) + 0.04 * np.sin(7.0 * t + 0.4)
    bright = np.clip((0.86 + 0.14 * depth) * shimmer, 0.0, 1.0)

    core = ring_coverage(dist, width)
    core_lit = core * bright

    # chromatic split: violet copy nudged up, ice-blue copy nudged down
    shift = 0.22 * core_w
    d_up, _, _, _ = ellipse_distance(xx, yy, cx - vx * shift, cy - vy * shift, a, b, tilt)
    d_dn, _, _, _ = ellipse_distance(xx, yy, cx + vx * shift, cy + vy * shift, a, b, tilt)
    fringe_v = ring_coverage(d_up, width) * (1.0 - core)
    fringe_c = ring_coverage(d_dn, width) * (1.0 - core)

    # glows from the lit core
    g_tight = norm(gaussian_blur(core_lit, 2.2 * S))
    g_inner = norm(gaussian_blur(core_lit, 6.0 * S))
    g_mid = norm(gaussian_blur(core_lit, 16.0 * S))
    g_haze = norm(gaussian_blur(core_lit, 42.0 * S))

    # the dot screen is anchored to the original picture, wherever it sits on the canvas
    pattern = screen_pattern((xx + 0.5) / S - 0.5 - screen_origin[0],
                             (yy + 0.5) / S - 0.5 - screen_origin[1])
    ink_pattern = 1.0 - pattern

    # background brightness decides how much contrast edge the ring needs
    lum = src @ np.array([0.2126, 0.7152, 0.0722])
    bg = gaussian_blur(lum, 10.0 * S)
    on_bright = smoothstep(0.55, 0.85, bg)

    # light spill on the crown just below the ring
    sx = px / (a * 1.05)
    sy = (py - b * 2.2) / (b * 3.2)
    spill = np.exp(-2.2 * (sx * sx + sy * sy))

    out = src.copy()

    # 1) navy halftone edge hugging the ring, only where the background is bright:
    #    this is how the picture itself separates white shapes from white areas
    band = norm(gaussian_blur(core, 0.62 * core_w)) * (1.0 - core)
    band = np.clip(band * 1.3, 0.0, 1.0) * on_bright
    out = multiply_toward(out, NAVY, band * 0.30 + halftone(band * 0.85, ink_pattern) * 0.60)

    # 2) light spill + wide haze, printed as light dots
    out = screen_blend(out, HAZE, halftone(spill * 0.30 + g_haze * 0.45, pattern) * 0.60)

    # 3) mid glow: half smooth, half dotted
    out = screen_blend(out, MID, g_mid * 0.35 + halftone(g_mid * 0.65, pattern) * 0.40)

    # 4) inner glow and tight bloom
    out = screen_blend(out, INNER, g_inner * 0.85)
    out = screen_blend(out, INNER, g_tight * 0.80)

    # 5) chromatic split on the outer edges of the ring
    out = over(out, VIOLET, fringe_v * 0.55)
    out = over(out, ICE, fringe_c * 0.55)

    # 6) the ring itself, shaded like a glowing tube
    profile = np.clip(1.0 - (np.abs(dist) / (width / 2.0)) ** 2, 0.0, 1.0)
    mix = (0.45 + 0.55 * profile) * bright
    tube = TUBE_EDGE[None, None, :] + (WHITE - TUBE_EDGE)[None, None, :] * mix[..., None]
    tube = tube * (0.93 + 0.07 * depth)[..., None]
    out = over(out, tube, core)

    # 7) grain only where the halo touched the picture, sized like the picture's own
    touched = np.clip(g_mid * 1.4, 0.0, 1.0) * (1.0 - 0.6 * core)   # keep the ring itself clean
    rng = np.random.default_rng(seed)
    grain = gaussian_blur(rng.normal(0.0, 1.0, (h, w)), 0.45 * S)
    grain *= 0.022 / (grain.std() + 1e-9)
    out = out + (grain * touched)[..., None]

    result = full.copy()
    result[y0:y1, x0:x1] = np.clip(out, 0.0, 1.0)
    return result


def vm(t, mu, kappa):
    """Smooth bump along the ring (von Mises): 1 at angle mu."""
    return np.exp(kappa * (np.cos(t - mu) - 1.0))


def ellipse_point(cx, cy, a, b, tilt, t):
    """Point on the rotated ellipse at parameter t, with its outward normal."""
    c, s = math.cos(tilt), math.sin(tilt)
    ex, ey = a * np.cos(t), b * np.sin(t)
    px = cx + ex * c - ey * s
    py = cy + ex * s + ey * c
    nlx, nly = b * np.cos(t), a * np.sin(t)
    n = np.hypot(nlx, nly)
    nlx, nly = nlx / n, nly / n
    return px, py, nlx * c - nly * s, nlx * s + nly * c


def splat(img, x, y, w):
    """Bilinear splat of point weights into a 2-D image."""
    h, wd = img.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < wd) & (yi >= 0) & (yi < h)
        np.add.at(img, (yi[ok], xi[ok]), (w * wt)[ok])
    return img


def star(xx, yy, x0, y0, len_v, len_h, width, gain):
    """Four-point sparkle: thin vertical and horizontal rays and a hot centre."""
    dx, dy = xx - x0, yy - y0
    ray_h = np.exp(-(dy / width) ** 2) * np.exp(-np.abs(dx) / (0.42 * len_h))
    ray_v = np.exp(-(dx / width) ** 2) * np.exp(-np.abs(dy) / (0.42 * len_v))
    # rays are wider and hotter close to the centre, then taper to a hairline
    ray_h += 0.6 * np.exp(-(dy / (2.6 * width)) ** 2) * np.exp(-np.abs(dx) / (0.13 * len_h))
    ray_v += 0.6 * np.exp(-(dx / (2.6 * width)) ** 2) * np.exp(-np.abs(dy) / (0.13 * len_v))
    r2 = dx * dx + dy * dy
    core = np.exp(-r2 / (2.0 * (1.1 * width) ** 2))
    halo = np.exp(-r2 / (2.0 * (0.16 * len_v) ** 2))
    return gain * (ray_h + ray_v + 1.2 * core + 0.35 * halo)


def render_energy(src, cx, cy, a, b, tilt_deg, core_w, screen_origin=(0.0, 0.0), scale=1.0, seed=11):
    """The large ring of light (see module docstring). Geometry in 1x canvas px."""
    H, W = src.shape[:2]
    S = float(scale)
    cx, cy = (cx + 0.5) * S - 0.5, (cy + 0.5) * S - 0.5
    a, b = a * S, b * S
    tilt = math.radians(tilt_deg)

    reach = 3 * 55.0 * S
    span_y = a * abs(math.sin(tilt)) + b
    x0, x1 = int(max(cx - a - reach, 0)), int(min(cx + a + reach, W))
    y0, y1 = int(max(cy - span_y - reach, 0)), int(min(cy + span_y + 4 * b + reach, H))
    full = src
    src = src[y0:y1, x0:x1]
    h, w = src.shape[:2]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    rng = np.random.default_rng(seed)

    dist, t, px, py = ellipse_distance(xx, yy, cx, cy, a, b, tilt)
    front = np.sin(t)                                  # +1 near arc, -1 far arc
    hot_fl = vm(t, math.radians(150.0), 2.2)           # front-left, nearest the face
    hot_rt = vm(t, math.radians(-25.0), 3.0)           # right end, where the ring turns away
    far = vm(t, math.radians(-95.0), 1.5)

    # 1) the ring is a bundle of bright light strands: they bunch into one white-hot
    #    band at the front-left (and the right end) and fan apart towards the tips,
    #    where some fly off as wisps
    tip_l, tip_r = vm(t, math.pi, 4.0), vm(t, 0.0, 4.0)
    spread = (1.0 - 0.72 * hot_fl - 0.35 * hot_rt) * (1.0 + 0.7 * tip_l + 0.45 * tip_r)
    # local radius of curvature: inner strands must not fold over at the tight tips
    rho = (a * a * np.sin(t) ** 2 + b * b * np.cos(t) ** 2) ** 1.5 / (a * b)
    strands = np.zeros_like(dist)
    n_str = 12
    for k in range(n_str):
        base = -8.0 + 22.0 * (k + rng.uniform(0.15, 0.85)) / n_str
        wave = rng.uniform(-7.0, 7.0)          # strands weave across each other
        phase = rng.uniform(0.0, 2 * math.pi)
        fly_l = rng.uniform(8.0, 26.0) if k % 3 == 0 else rng.uniform(0.0, 6.0)
        fly_r = rng.uniform(5.0, 18.0) if k % 4 == 1 else 0.0
        off = S * ((base + wave * np.sin(t - phase)) * spread + fly_l * tip_l + fly_r * tip_r)
        off = np.maximum(off, -0.7 * rho)
        width = S * rng.uniform(1.5, 2.8)
        env = 0.35 + 0.65 * vm(t, rng.uniform(-math.pi, math.pi), rng.uniform(0.3, 1.2))
        shim = 1.0 + 0.25 * np.sin(rng.integers(3, 10) * t + rng.uniform(0.0, 6.3))
        noise = np.convolve(rng.normal(0.0, 1.0, 720), np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2), "same")
        noise /= noise.std() + 1e-9
        shim = shim * np.clip(1.0 + 0.35 * np.interp(t, np.linspace(-math.pi, math.pi, 720), noise), 0.2, 2.0)
        strands += rng.uniform(0.85, 1.1) * env * shim * np.exp(-2.0 * ((dist - off) / (0.5 * width)) ** 2)
    strands *= 0.55 + 0.45 * (1.0 - far)              # the far arc is quieter

    # the merged white-hot band where the strands pile up
    pile = np.clip(0.95 * hot_fl + 0.75 * hot_rt, 0.0, 1.0)
    band_w = S * core_w * (0.5 + 1.0 * hot_fl + 0.55 * hot_rt)
    band = np.exp(-2.0 * (dist / (0.5 * band_w)) ** 2) * (0.3 + 0.7 * pile)
    hot = np.exp(-2.0 * (dist / (0.34 * band_w)) ** 2) * pile

    # 3) glitter specks around the ring, densest where the light is
    n = int(150 * a / (335.0 * S))
    tt = rng.uniform(-math.pi, math.pi, n * 4)
    wgt = 0.35 + vm(tt, math.radians(150.0), 2.2) + 0.8 * vm(tt, math.radians(-25.0), 3.0)
    tt = tt[rng.uniform(0.0, wgt.max(), tt.size) < wgt][:n]
    gx, gy, gnx, gny = ellipse_point(cx, cy, a, b, tilt, tt)
    off = S * rng.normal(4.0, 11.0, tt.size)
    slide = S * rng.normal(0.0, 6.0, tt.size)
    glitter = splat(np.zeros((h, w)), gx + gnx * off - gny * slide - x0,
                    gy + gny * off + gnx * slide - y0, rng.uniform(0.35, 1.0, tt.size))
    # a second, denser sprinkle riding on the band itself, where the light piles up
    m = int(240 * a / (335.0 * S))
    tb = rng.uniform(-math.pi, math.pi, m * 4)
    wb = 0.3 + vm(tb, math.radians(150.0), 2.2) + 0.8 * vm(tb, math.radians(-25.0), 3.0)
    tb = tb[rng.uniform(0.0, wb.max(), tb.size) < wb][:m]
    bx, by, bnx, bny = ellipse_point(cx, cy, a, b, tilt, tb)
    boff = S * rng.normal(0.0, 4.0, tb.size)
    bsl = S * rng.normal(0.0, 3.0, tb.size)
    splat(glitter, bx + bnx * boff - bny * bsl - x0, by + bny * boff + bnx * bsl - y0,
          rng.uniform(0.5, 1.0, tb.size))
    glitter = np.clip(1.3 * norm(gaussian_blur(glitter, 0.7 * S)), 0.0, 1.0)

    # 4) sparkle stars
    stars = np.zeros((h, w))
    for td, offp, lv, lh, gain in STARS:
        sx, sy, snx, sny = ellipse_point(cx, cy, a, b, tilt, np.array([math.radians(td)]))
        stars += star(xx, yy, sx[0] + snx[0] * offp * S, sy[0] + sny[0] * offp * S,
                      lv * S, lh * S, 0.9 * S, gain)

    emission = band + strands + 0.9 * glitter + stars

    # 5) bloom, from white-hot to the picture's periwinkle haze; each layer is
    #    scaled against the ring itself so the star centres cannot dim it
    feed = np.clip(emission, 0.0, 1.3)

    def bloom(sigma):
        ref = gaussian_blur(band, sigma).max() + 1e-9
        return np.clip(gaussian_blur(feed, sigma) / ref, 0.0, 1.25)

    b1, b2, b3, b4 = bloom(2.5 * S), bloom(8.0 * S), bloom(22.0 * S), bloom(55.0 * S)

    pattern = screen_pattern((xx + 0.5) / S - 0.5 - screen_origin[0],
                             (yy + 0.5) / S - 0.5 - screen_origin[1])
    spill = np.exp(-2.0 * ((px / a) ** 2 + ((py - 1.6 * b) / (2.6 * b)) ** 2))

    out = src.copy()

    # a dark, dotted backdrop inside and around the ring (as in the reference edit),
    # so the light reads against shadow rather than against the white lace
    lum = src @ np.array([0.2126, 0.7152, 0.0722])
    on_bright = smoothstep(0.35, 0.80, gaussian_blur(lum, 6.0 * S))
    reach_r = np.sqrt((px / (a * 1.05)) ** 2 + (py / (b * 2.3)) ** 2)
    shade = smoothstep(1.70, 0.70, reach_r) * on_bright
    out = multiply_toward(out, NAVY_DEEP, shade * 0.35 + halftone(shade * 0.8, 1.0 - pattern) * 0.50)

    # the wide glow stays mostly outside the ring, so its inside keeps the deep navy
    ring_r = np.sqrt((px / a) ** 2 + (py / b) ** 2)
    outside = 1.0 - 0.65 * smoothstep(0.92, 0.55, ring_r)
    out = screen_blend(out, GLOW_4, (halftone(b4 * 0.6 + spill * 0.3, pattern) * 0.55 + b4 * 0.22) * outside)
    out = screen_blend(out, GLOW_3, np.clip(b3 * 0.55, 0.0, 1.0) * outside)
    out = screen_blend(out, GLOW_2, np.clip(b2 * 0.60, 0.0, 1.0))
    out = screen_blend(out, GLOW_1, np.clip(b1 * 0.45, 0.0, 1.0))
    hot_bloom = gaussian_blur(hot, 5.0 * S)
    out = screen_blend(out, WHITE, np.clip(0.75 * hot_bloom / (hot_bloom.max() + 1e-9), 0.0, 1.0))
    out = screen_blend(out, WHITE, np.clip(1.15 * emission, 0.0, 1.0))
    out = over(out, WHITE, np.clip(hot * 1.15, 0.0, 1.0))

    # grain in the glow only, sized like the picture's own; the light itself stays clean
    grain = gaussian_blur(rng.normal(0.0, 1.0, (h, w)), 0.45 * S)
    grain *= 0.02 / (grain.std() + 1e-9)
    out = out + (grain * np.clip(b3 * 1.2, 0.0, 1.0) * (1.0 - np.clip(emission, 0.0, 1.0)))[..., None]

    result = full.copy()
    result[y0:y1, x0:x1] = np.clip(out, 0.0, 1.0)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--style", choices=sorted(STYLE_DEFAULTS), default="clean")
    for key in ("cx", "cy", "a", "b", "tilt", "width"):
        ap.add_argument(f"--{key}", type=float, default=None)
    ap.add_argument("--screen-origin", type=float, nargs=2, default=SCREEN_ORIGIN, metavar=("X", "Y"),
                    help="where the original picture's top-left corner sits on this canvas")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="canvas resolution relative to the geometry above (2 for the 2x render)")
    args = ap.parse_args()

    source = Image.open(args.input)
    icc = source.info.get("icc_profile")  # keep the colour profile so tones match exactly
    src = np.asarray(source.convert("RGB")).astype(np.float64) / 255.0
    geo = dict(STYLE_DEFAULTS[args.style])
    for key in geo:
        if getattr(args, key) is not None:
            geo[key] = getattr(args, key)
    render = render_energy if args.style == "energy" else render_clean
    out = render(src, geo["cx"], geo["cy"], geo["a"], geo["b"], geo["tilt"], geo["width"],
                 screen_origin=tuple(args.screen_origin), scale=args.scale)
    img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))
    extra = {"icc_profile": icc} if icc else {}
    if args.output.lower().endswith((".jpg", ".jpeg")):
        img.save(args.output, quality=95, subsampling=0, **extra)
    else:
        img.save(args.output, **extra)


if __name__ == "__main__":
    main()
