#!/usr/bin/env python3
"""Enlarge the halo of the user's edit (edit.jpg), tilt it slightly, keep the frame.

edit.jpg is the original picture (source.jpg) after an enhancer pass (which
turned the print dots into a fine maze texture) with a glowing halo painted on
top. The halo is enlarged SCALE times and turned ROTATE degrees (positive =
flatter, the right end comes down) around a pivot on the ring's minor axis,
PIVOT of the way from its centre to the middle of its front arc. That choice
keeps the bigger ring inside the same 1254 px frame, keeps the old ring fully
under the new one, and moves the front arc only a few px on the head. The halo
is made of three parts, and each is moved with that same transform:

1. The ring, its darkened inside and its glow out to D0 px past the ring are
   magnified straight from the edit's own pixels (cubic spline), so the halo
   keeps exactly the look of the edit; a mild unsharp mask offsets the
   softening that the enlargement brings.
2. The lights outside that zone (sparkle stars, the vertical flare, glitter
   and the bloom) are lifted off the edit as a light layer, magnified and
   screened back on. The background they sat on is recovered by undoing the
   screen blend (bloom) and inpainting (stars, flare, glitter). The bloom is
   measured against source.jpg, aligned to the edit (scale, anisotropy,
   rotation and shift fitted to 0.986 correlation, under 1 px residual).
3. The frame stays as it is. The edit's last 3 columns on the right carry a
   green fringe; their colour is taken from the column next to them.
   (--margins L T R B grows the canvas instead, quilting the new border from
   the edit's own background, never from the face, the sword or the halo.)

Everything outside the halo's reach keeps the edit's exact pixels.

    python3 enlarge_halo.py edit.jpg source.jpg out.png [--scale 1.27] [--rotate 5]
"""

import argparse
import math
import time

import numpy as np
from PIL import Image
from scipy import ndimage

import zoom_out as zo
from add_halo import ellipse_distance

# ring of the edit (ellipse fitted to its bright band): cx, cy, a, b, tilt deg
RING = (389.9, 144.0, 287.5, 54.5, -18.1)
SCALE = 1.27
ROTATE = 5.0       # degrees; positive = flatter (the right end comes down)
PIVOT = 0.4        # 0 = ring centre, 1 = middle of the front arc
D0 = 25.0          # px past the ring centre line that move as pixels (band + glow)
FEATHER = 24.0     # px, soft hand-over from the magnified pixels to the background
MARGINS = (0, 0, 0, 0)       # canvas growth (left, top, right, bottom); the zoomed-out
                             # variant used (64, 80, 48, 32)
FRINGE_COLS = 3              # green fringe on the edit's right edge
RESHARPEN = 0.35             # unsharp amount that offsets the softening of the enlargement

# source.jpg -> edit.jpg geometry: scale, extra y scale, rotation (rad), shift x, y
ALIGN = (1.02971815, 1.00459745, 0.00472007546, 15.8556122, -0.988414132)

# light-detail detection (stars, flare, glitter)
DETAIL_MIN_LUMA = 0.50
FADE_BANDS = (8, 24, 6, 30)  # edge strips rebuilt by the zoom out (l, t, r, b): the edit
                             # darkens towards its top/bottom edges, and its last 3
                             # columns on the right carry a green fringe

FACE_ELLIPSE = (560.0, 670.0, 330.0, 250.0)  # in edit coordinates
SWORD_RECT = (300, 800, 1060, 1010)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gauss(a, s):
    return ndimage.gaussian_filter(a, (s, s, 0)) if a.ndim == 3 else ndimage.gaussian_filter(a, s)


def masked_gauss(a, m, s):
    """Normalised convolution: smooth a using only the pixels where m is set."""
    m = m.astype(np.float64)
    num = gauss(a * (m[..., None] if a.ndim == 3 else m), s)
    den = ndimage.gaussian_filter(m, s)
    return num / np.maximum(den[..., None] if a.ndim == 3 else den, 1e-6), den


def screen(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


def unscreen(r, e, cap=0.7):
    """Background that, screened with light e, gives r."""
    e = np.minimum(e, cap)
    return np.clip(1.0 - (1.0 - r) / (1.0 - e), 0.0, 1.0)


def warp_to_edit(img, shape):
    """Resample a source.jpg-shaped image into edit.jpg geometry (cubic)."""
    s, sy_, rot, tx, ty = ALIGN
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    c, sn = math.cos(rot), math.sin(rot)
    X, Y = xx - tx, yy - ty
    sx = (c * X + sn * Y) / s
    sy = (-sn * X + c * Y) / (s * sy_)
    return np.stack([ndimage.map_coordinates(img[..., k], [sy, sx], order=3, mode="nearest")
                     for k in range(img.shape[-1])], -1)


def tone_match(src, ref, free):
    """Per-channel tone curve taking src to ref, fitted on the free area."""
    out = np.empty_like(src)
    sl, rl = gauss(src, 2.5), gauss(ref, 2.5)
    for c in range(3):
        x, y = sl[..., c][free], rl[..., c][free]
        idx = np.clip(np.digitize(x, np.linspace(0, 1, 65)) - 1, 0, 63)
        cx_, cy_ = [], []
        for i in range(64):
            m = idx == i
            if m.sum() > 80:
                cx_.append(x[m].mean())
                cy_.append(np.median(y[m]))
        cx_, cy_ = np.array(cx_), np.maximum.accumulate(np.array(cy_))
        out[..., c] = np.interp(src[..., c], cx_, cy_)
    return out


def inpaint(img, mask, rng):
    """Fill masked pixels: smooth fill plus fine texture borrowed from nearby."""
    smooth = zo.push_pull_fill(img, ~mask)
    smooth[mask] = gauss(smooth, 1.5)[mask]
    hp = img - gauss(img, 1.2)
    tex = np.zeros_like(img)
    cnt = np.zeros(mask.shape)
    for dy, dx in ((0, 14), (0, -14), (14, 0), (-14, 0), (10, 10), (-10, -10)):
        sh_hp = np.roll(hp, (dy, dx), (0, 1))
        sh_ok = ~np.roll(mask, (dy, dx), (0, 1))
        tex += sh_hp * sh_ok[..., None]
        cnt += sh_ok
    tex /= np.maximum(cnt, 1)[..., None]
    out = img.copy()
    out[mask] = np.clip(smooth + 0.9 * tex, 0, 1)[mask]
    return out


def fix_right_fringe(img, n=FRINGE_COLS):
    """Give the last n columns the colour of the column next to them, keeping their
    brightness (the edit has a thin green line there)."""
    lw = np.array([0.2126, 0.7152, 0.0722])
    ref = img[:, -n - 2]
    chroma = ndimage.gaussian_filter1d(ref - (ref @ lw)[:, None], 2.0, axis=0)
    out = img.copy()
    for j in range(1, n + 1):
        out[:, -j] = np.clip((img[:, -j] @ lw)[:, None] + chroma, 0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edit")
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--scale", type=float, default=SCALE)
    ap.add_argument("--rotate", type=float, default=ROTATE)
    ap.add_argument("--pivot", type=float, default=PIVOT)
    ap.add_argument("--margins", type=int, nargs=4, default=MARGINS, metavar=("L", "T", "R", "B"))
    ap.add_argument("--debug", default=None, help="folder for intermediate images")
    args = ap.parse_args()
    k = args.scale
    rng = np.random.default_rng(5)

    R = np.asarray(Image.open(args.edit).convert("RGB")).astype(np.float64) / 255.0
    H, W = R.shape[:2]
    S = np.asarray(Image.open(args.source).convert("RGB")).astype(np.float64) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # ---- geometry of the old ring and the enlargement ---------------------------
    cx, cy, a, b, tilt_deg = RING
    tilt = math.radians(tilt_deg)
    anchor = (cx - b * math.sin(tilt), cy + b * math.cos(tilt))   # middle of the front arc
    d_old, t_old, _, _ = ellipse_distance(xx, yy, cx, cy, a, b, tilt)
    pvx, pvy = cx + args.pivot * (anchor[0] - cx), cy + args.pivot * (anchor[1] - cy)
    log(f"pivot {pvx:.1f},{pvy:.1f}  scale {k}  rotate {args.rotate}")

    # ---- the original picture in the edit's geometry and tones ------------------
    free = np.ones((H, W), bool)
    free[:520, :900] = False
    free[:8, :] = free[-8:, :] = free[:, :8] = free[:, -8:] = False
    s_clean, _ = zo.descreen(S)
    B_s = tone_match(warp_to_edit(s_clean, (H, W)), R, free)

    # ---- light details outside the band: stars, flare, glitter ------------------
    lw = np.array([0.2126, 0.7152, 0.0722])
    luma = R @ lw
    tophat = luma - ndimage.grey_opening(luma, size=(7, 7))
    # light the edit added: brighter than anything the original has within 2 px
    added = luma - ndimage.maximum_filter(B_s @ lw, size=5)
    tdeg = np.degrees(t_old)
    head = (d_old > 0) & (tdeg > 30) & (tdeg < 150) & (xx > 235)   # hair/lace under the front arc
    zone = (d_old > D0 - 8) & (d_old < 150) & ~head
    det = zone & (luma > DETAIL_MIN_LUMA) & (((tophat > 0.12) & (added > 0.15)) | (added > 0.32))
    det = ndimage.binary_opening(det, np.ones((1, 2))) | ndimage.binary_opening(det, np.ones((2, 1)))
    # grow each detail over its own soft glow (where the added light is still clear)
    for _ in range(4):
        det |= ndimage.binary_dilation(det, np.ones((3, 3))) & (added > 0.06) & zone
    det = ndimage.binary_dilation(det, np.ones((5, 5)))
    log(f"detail pixels {int(det.sum())}")

    # ---- bloom outside the band: compare with the original picture ---------------
    Rs = gauss(R, 2.0)
    Bs = gauss(B_s, 2.0)
    e_raw = 1.0 - (1.0 - Rs) / np.maximum(1.0 - Bs, 0.12)
    usable = zone & ~det & (Bs.max(-1) < 0.82)
    e_bloom, wsum = masked_gauss(np.clip(e_raw, -0.3, 1.0), usable, 9.0)
    e_bloom = np.clip(e_bloom, 0.0, 0.85) * (wsum > 0.05)[..., None]
    e_bloom *= zo.smoothstep(150.0, 100.0, d_old)[..., None]     # nothing far from the ring
    e_bloom *= zo.smoothstep(D0 - 12.0, D0 - 2.0, d_old)[..., None]
    e_bloom[head] = 0.0
    e_bloom = gauss(e_bloom, 2.0)

    # ---- the edit without its outer halo light -----------------------------------
    clean = R.copy()
    ext = d_old > D0 - 12.0
    clean[ext] = unscreen(R, e_bloom)[ext]
    clean = inpaint(clean, det, rng)
    lit_clean = screen(clean, e_bloom)
    e_det = np.where(det[..., None], np.clip(1.0 - (1.0 - R) / np.maximum(1.0 - lit_clean, 1e-3), 0, 1), 0.0)
    e_ext = screen(e_bloom, e_det)
    L, T, Rm, Bm = args.margins
    if T > 0 or L > 0:
        # the flare and bloom were cut by the old frame: let them taper there
        e_ext *= zo.smoothstep(0.0, 30.0, yy)[..., None]
        e_ext *= zo.smoothstep(0.0, 8.0, xx)[..., None]

    if args.debug:
        dbg = lambda name, im: Image.fromarray((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)).save(f"{args.debug}/{name}.png")
        dbg("clean", clean)
        dbg("e_ext", e_ext)
        dbg("det", det.astype(float))

    if (L, T, Rm, Bm) == (0, 0, 0, 0):
        bg = fix_right_fringe(clean)
        inner = np.ones((H, W), bool)
        inner[:, W - FRINGE_COLS:] = False
    else:
        bg, inner = zoom_out_background(clean, d_old, det, xx, yy, lw, (L, T, Rm, Bm), rng)
    place_halo(R, bg, inner, e_ext, clean, ext, d_old, (cx, cy, a, b, tilt), (pvx, pvy), k,
               math.radians(args.rotate), (L, T, Rm, Bm), args.output)


def zoom_out_background(clean, d_old, det, xx, yy, lw, margins, rng):
    """Grow the canvas: quilt the new border from the clean background."""
    H, W = clean.shape[:2]
    L, T, Rm, Bm = margins
    CH, CW = H + T + Bm, W + L + Rm
    fl, ft, fr, fb = FADE_BANDS
    trusted = np.zeros((H, W), bool)
    trusted[ft:H - fb, fl:W - fr] = True
    fcx, fcy, frx, fry = FACE_ELLIPSE
    face = ((xx - fcx) / frx) ** 2 + ((yy - fcy) / fry) ** 2 < 1.0
    x0, y0, x1, y1 = SWORD_RECT
    sword = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    allowed = trusted & ~face & ~sword & (d_old > D0 + 45.0) & ~ndimage.binary_dilation(det, np.ones((9, 9)))

    original = np.zeros((CH, CW), bool)
    original[T + ft:T + H - fb, L + fl:L + W - fr] = True
    # the old ring is still in `clean` (the enlarged one covers it later); fill it with
    # its surroundings so the new border is grown from background, not from the ring
    ring_zone = d_old < D0 + 40.0
    dark_bg = ~ring_zone & ((clean @ lw) < 0.55)       # the navy backdrop around the halo
    base = clean.copy()
    base[ring_zone] = zo.push_pull_fill(clean, dark_bg)[ring_zone]
    allowed &= ~ring_zone
    canvas = np.zeros((CH, CW, 3))
    canvas[T:T + H, L:L + W] = base
    filled = original.copy()
    low = np.zeros((CH, CW, 3))
    low[T:T + H, L:L + W] = gauss(base, zo.GUIDE_SIGMA)
    low = zo.push_pull_fill(low, original)
    wout = (~original).astype(np.float64)
    num = gauss(low * wout[..., None], zo.GUIDE_SIGMA * 2)
    den = ndimage.gaussian_filter(wout, zo.GUIDE_SIGMA * 2)[..., None]
    low[~original] = (num / np.maximum(den, 1e-6))[~original]
    zo.BLOCK, zo.OVERLAP = 56, 18
    zo.STEP = zo.BLOCK - zo.OVERLAP
    zo.REPLACE_DEPTH = 10
    zo.LOCAL_WINDOW = (300, 700)
    canvas = zo.quilt(canvas, filled, original, low, base, base, gauss(base, zo.GUIDE_SIGMA),
                      allowed, rng, offset=(T, L))
    bg = np.clip(canvas, 0.0, 1.0)
    inner = original & (ndimage.distance_transform_edt(original) > zo.REPLACE_DEPTH)
    edit_on_canvas = np.zeros((CH, CW, 3))
    edit_on_canvas[T:T + H, L:L + W] = clean
    bg[inner] = edit_on_canvas[inner]
    return bg, inner[T:T + H, L:L + W]


def place_halo(R, bg, inner_edit, e_ext, clean, ext, d_old, ring, pivot, k, rot, margins, output):
    """Paint the transformed halo on the background and save."""
    H, W = R.shape[:2]
    L, T, Rm, Bm = margins
    CH, CW = H + T + Bm, W + L + Rm
    cx, cy, a, b, tilt = ring
    px, py = pivot
    cr, sr = math.cos(rot), math.sin(rot)
    CY_, CX_ = np.mgrid[0:CH, 0:CW].astype(np.float64)
    ex, ey = CX_ - L - px, CY_ - T - py            # edit coordinates, relative to the pivot
    sx = px + (cr * ex + sr * ey) / k              # where each pixel comes from before the
    sy = py + (-sr * ex + cr * ey) / k             # enlargement and the turn
    vx, vy = cx - px, cy - py
    ncx, ncy = px + k * (cr * vx - sr * vy) + L, py + k * (sr * vx + cr * vy) + T
    d_new, _, _, _ = ellipse_distance(CX_, CY_, ncx, ncy, k * a, k * b, tilt + rot)
    inside_edit = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
    w = (1.0 - zo.smoothstep(k * D0, k * D0 + FEATHER, d_new)) * inside_edit
    patch = np.stack([ndimage.map_coordinates(R[..., c], [sy, sx], order=3, mode="nearest")
                      for c in range(3)], -1)
    light = np.stack([ndimage.map_coordinates(e_ext[..., c], [sy, sx], order=3, mode="constant", cval=0.0)
                      for c in range(3)], -1)
    # enlarging softens: give back the crispness of the rest of the picture
    amt = RESHARPEN * min(max(k - 1.0, 0.0) / 0.3, 1.0)
    if amt > 0:
        patch = np.clip(patch + amt * (patch - gauss(patch, 1.0)), 0.0, 1.0)
        light = light + amt * (light - gauss(light, 1.0))
    light = np.clip(light, 0.0, 1.0)
    out = w[..., None] * patch + (1.0 - w[..., None]) * screen(bg, light)
    out = np.clip(out, 0.0, 1.0)

    # untouched pixels stay bit-exact
    unchanged = inner_edit & ~(ext & (np.abs(clean - R).max(-1) > 0))
    keep = (np.abs(w) < 1e-6) & (light.max(-1) < 1e-4) & np.pad(unchanged, ((T, Bm), (L, Rm)))
    exact = np.zeros((CH, CW, 3))
    exact[T:T + H, L:L + W] = R
    out[keep] = exact[keep]

    Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(output)
    log(f"saved {output} ({CW}x{CH})")


if __name__ == "__main__":
    main()
