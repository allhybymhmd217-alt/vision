#!/usr/bin/env python3
"""Paint a floating "angel / dead" halo above the character's head.

The halo is built to sit *inside* the artwork instead of on top of it:

* a clean elliptical ring with strong perspective flattening (b/a ~ 0.2),
  floating a little above the crown, like the reference sketches;
* tilted to follow the head, whose crown leans towards the upper left;
* shaded like a glowing tube: white-hot centre line, cold lavender edges,
  the near (lower) arc slightly thicker and brighter than the far arc;
* a lavender / periwinkle glow whose colours are sampled from the picture's
  own grading (navy shadows, lavender mids, neutral white highlights);
* the outer glow, the light spill on the hair and the contrast edge are
  printed with the picture's own halftone screen (45-degree dot lattice,
  0.1802 cycles/px), phase-matched so the new dots land on the existing grid;
* a violet / ice-blue chromatic split along the ring edges;
* matching grain.

Only numpy and Pillow are needed:

    python3 add_halo.py source.jpg output.png
"""

import argparse
import math

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Halo geometry (pixels, for the 1200x1184 source)
# --------------------------------------------------------------------------
CENTER = (430.0, 140.0)  # ring centre (x, y)
SEMI_MAJOR = 265.0       # half width of the ring
SEMI_MINOR = 52.0        # half height (perspective flattening)
TILT_DEG = -14.0         # negative = right end higher, follows the head tilt
CORE_WIDTH = 9.5         # line thickness at the sides of the ring

# --------------------------------------------------------------------------
# Halftone screen measured from the source: two cosine gratings at +/-45 deg.
# Frequencies and phases were fitted on the dotted hair areas; the phase is
# consistent across the whole picture, so it is a single global screen.
# --------------------------------------------------------------------------
SCREEN_F1 = (0.1802, 0.1802)
SCREEN_F2 = (0.1801, -0.1802)
SCREEN_PH1 = math.radians(38.0)
SCREEN_PH2 = math.radians(-5.0)

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


def render(src, cx, cy, a, b, tilt_deg, core_w, seed=7):
    H, W = src.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
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
    g_tight = norm(gaussian_blur(core_lit, 2.2))
    g_inner = norm(gaussian_blur(core_lit, 6.0))
    g_mid = norm(gaussian_blur(core_lit, 16.0))
    g_haze = norm(gaussian_blur(core_lit, 42.0))

    pattern = screen_pattern(xx, yy)
    ink_pattern = 1.0 - pattern

    # background brightness decides how much contrast edge the ring needs
    lum = src @ np.array([0.2126, 0.7152, 0.0722])
    bg = gaussian_blur(lum, 10.0)
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

    # 7) grain only where the halo touched the picture
    touched = np.clip(g_mid * 1.4 + core, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, 0.022, (H, W))
    out = out + (grain * touched)[..., None]

    return np.clip(out, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cx", type=float, default=CENTER[0])
    ap.add_argument("--cy", type=float, default=CENTER[1])
    ap.add_argument("--a", type=float, default=SEMI_MAJOR)
    ap.add_argument("--b", type=float, default=SEMI_MINOR)
    ap.add_argument("--tilt", type=float, default=TILT_DEG)
    ap.add_argument("--width", type=float, default=CORE_WIDTH)
    args = ap.parse_args()

    source = Image.open(args.input)
    icc = source.info.get("icc_profile")  # keep the colour profile so tones match exactly
    src = np.asarray(source.convert("RGB")).astype(np.float64) / 255.0
    out = render(src, args.cx, args.cy, args.a, args.b, args.tilt, args.width)
    img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))
    extra = {"icc_profile": icc} if icc else {}
    if args.output.lower().endswith((".jpg", ".jpeg")):
        img.save(args.output, quality=95, subsampling=0, **extra)
    else:
        img.save(args.output, **extra)


if __name__ == "__main__":
    main()
