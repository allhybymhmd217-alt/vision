#!/usr/bin/env python3
"""Give a black-and-white picture the colours and the halo of the user's edit.

1. Colours: a gradient map measured on edit.jpg away from its halo. For every
   brightness level it takes the edit's median colour, and the picture keeps
   its own brightness exactly: only the hue changes (navy blacks, periwinkle
   mids, lavender-white highlights).
2. Halo: lifted off edit.jpg as light (screened on) and the shadow the edit put
   behind the ring (multiplied on, before the colouring).
   - Background under the ring: source.jpg aligned to the edit, times that
     shadow (edit / original away from the light, inside and outside the ring).
   - Light: what screens that background into the edit, per pixel. It is exact
     where the background was dark; there it is kept as it is (the median error
     against the edit is 0).
   - Over the bright lace and hair it is not exact. There the ring is read in
     ring coordinates (arc length along it, distance across it), where the
     halo's light runs in long lines and leaking background is short blobs: a
     directional opening and closing keeps the lines. Off the band, the light
     may not exceed the glow measured where it is exact and must fade with
     distance, which stops lily and lace edges coming through.
   - Kept by hand from measurements: the back arc over the lace top (carried
     along the ring from both sides), the thin outer arc round the right end,
     the folds inside the two tight ends (raw light), the two stars (exact
     light with their glow), the vertical flare (top-hat along its line), and
     sparkle dust (bright specks the line filter takes out).
3. The halo is mirrored (this head leans the other way), turned to a 10 degree
   tilt (parallel to the eye and brow line) and placed over the head. The two
   stars and the flare stay upright and ride along with their point on the
   ring. Pixels away from the halo keep the plain recolour.

    python3 wings_halo.py wings.jpg edit.jpg source.jpg out.png [--cx X --cy Y --scale S --rotate R]
"""

import argparse
import math

import numpy as np
from PIL import Image
from scipy import ndimage

import enlarge_halo as eh
import zoom_out as zo
from add_halo import ellipse_point

LW = np.array([0.2126, 0.7152, 0.0722])

# where the halo goes on the picture: ring centre, size relative to the edit's
# ring, extra turn (degrees, applied after mirroring; the mirrored ring alone
# tilts +18.1 degrees, right end down)
CENTER = (915.0, 125.0)   # right above the crown (x about 916)
SCALE = 0.84               # keeps the ring's end 17 px inside the right edge
ROTATE = -8.0      # -> 10 degrees: parallel to the eye and brow line (measured 6-8
                   # degrees, right side lower), with a touch more for the hair's sweep
MIRROR = True

# the edit's two four-point stars; they and the vertical flare stay upright when
# the ring turns, and ride along with their point on the ring
STARS = ((184.0, 134.0), (153.0, 296.0))

# the back arc crosses the top of the lace headpiece (ring angles in degrees). Its
# lace, lit by the halo, is as bright as the ring's own lines there, so across that
# stretch the back arc's lines are carried over from both ends of it, along the ring
LACE_T = (-101.0, -74.0)
LINE_SLOPES = (0.0, 0.05, -0.05, 0.1, -0.1, 0.18, -0.18, 0.3, -0.3)
# the thin outer arc around the ring's right end: ring angles and distance (px)
OUTER_ARC_T = (2.0, 32.0)
OUTER_ARC_D = (34.0, 52.0)
FLARE_X = (585, 612)      # columns that hold the edit's vertical flare
FLARE_ROWS = (0, 300)


def ss(e0, e1, x):
    return zo.smoothstep(e0, e1, x)


def gradient_map(R, mask):
    """Edit colour per brightness: 256 entries, each with the luma of its index."""
    y = R[mask] @ LW
    idx = np.clip((y * 255 + 0.5).astype(int), 0, 255)
    lut = np.full((256, 3), np.nan)
    for i in range(256):
        sel = idx == i
        if sel.sum() >= 30:
            lut[i] = np.median(R[mask][sel], 0)
    ok = ~np.isnan(lut[:, 0])
    grid = np.arange(256)
    chroma = lut - (lut @ LW)[:, None]              # colour offset at each level
    chroma = np.stack([np.interp(grid, grid[ok], chroma[ok, c]) for c in range(3)], 1)
    chroma = ndimage.gaussian_filter1d(chroma, 2.0, axis=0)
    return np.clip(grid[:, None] / 255.0 + chroma, 0.0, 1.0)


def apply_map(lut, luma):
    x = np.clip(luma, 0.0, 1.0) * 255.0
    return np.stack([np.interp(x, np.arange(256), lut[:, c]) for c in range(3)], -1)


def line_footprints(length, slopes):
    """Flat line structuring elements of the given length at the given slopes."""
    fps = []
    for m in slopes:
        h = int(math.ceil(abs(m) * length / 2))
        fp = np.zeros((2 * h + 1, length), bool)
        for j in range(length):
            fp[h + int(round(m * (j - length // 2))), j] = True
        fps.append(fp)
    return fps


class RingCoords:
    """Arc length s along the ring's centre line and distance d across it."""

    def __init__(self, ring, dmin=-60, dmax=100):
        cx, cy, a, b, tilt = ring
        self.args = (cx, cy, a, b, math.radians(tilt))
        self.tt = np.linspace(-math.pi, math.pi, 40001)
        px, py, _, _ = ellipse_point(*self.args, self.tt)
        self.sgrid = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        self.perim = self.sgrid[-1]
        self.ns = int(round(self.perim))
        self.t_u = np.interp(np.arange(self.ns) * self.perim / self.ns, self.sgrid, self.tt)
        self.dmin, self.dmax = dmin, dmax
        self.d_u = np.arange(dmin, dmax + 1.0)
        px, py, nx, ny = ellipse_point(*self.args, self.t_u)
        self.X = px[None] + self.d_u[:, None] * nx[None]
        self.Y = py[None] + self.d_u[:, None] * ny[None]

    def unroll(self, img):
        return ndimage.map_coordinates(img, [self.Y, self.X], order=1, mode="nearest")

    def reroll(self, U, d, t, pad=8):
        """Read an unrolled array back at image pixels with ring coordinates (d, t)."""
        col = np.interp(t, self.tt, self.sgrid) * self.ns / self.perim + pad
        row = np.clip(d - self.dmin, 0, U.shape[0] - 1)
        out = ndimage.map_coordinates(np.pad(U, ((0, 0), (pad, pad)), mode="wrap"), [row, col],
                                      order=1, mode="nearest")
        return out * ((d >= self.dmin) & (d <= self.dmax))

    def s_of_t(self, t_deg):
        return float(np.interp(math.radians(t_deg), self.tt, self.sgrid) * self.ns / self.perim)


def halo_light(R, S):
    """The edit's halo in edit coordinates: its whole light (RGB), the same light
    split into the ring and the upright parts (stars, flare) for when the ring is
    turned, and the shadow behind the ring (luma factor)."""
    hl = eh.halo_layers(R, S, np.random.default_rng(5))
    d, t, det, e_det, B_s = hl["d_old"], hl["t_old"], hl["det"], hl["e_det"], hl["B_s"]
    quiet = ~ndimage.binary_dilation(det, np.ones((15, 15)))

    # shadow: edit / original away from the light, filled in separately on each side
    ratio = np.clip(eh.gauss(R, 4.0) / np.maximum(eh.gauss(B_s, 4.0), 0.03), 0.2, 1.3)
    D = np.ones_like(R)
    for side, known in ((d < 0, (d < -32) & quiet), (d >= 0, ((d > 40) & quiet) | (d >= 160))):
        Dm = np.where(known[..., None], ratio, 1.0)
        Dm[d >= 160] = 1.0
        D[side] = eh.gauss(zo.push_pull_fill(Dm, known), 5.0)[side]
    bg = np.clip(D * B_s, 0.0, 1.0)

    # the light, pixel by pixel: what screens that background into the edit. It is
    # exact where the background was dark (w = 1); over bright lace and hair it is not
    E = np.clip(1.0 - (1.0 - R) / np.maximum(1.0 - bg, 1e-3), 0.0, 1.0)
    El = E @ LW
    w = ss(0.62, 0.42, eh.gauss(bg @ LW, 3.0))

    # in ring coordinates the halo's light runs in long lines along the ring, while
    # background texture leaking in is short blobs: keep what a long line fits in,
    # then close short dark holes. Raw light is kept up to that, so lines stay crisp
    rc = RingCoords(eh.RING)
    Elu = rc.unroll(El)
    pad = 40
    P = np.pad(Elu, ((0, 0), (pad, pad)), mode="wrap")
    op = np.max([ndimage.grey_opening(P, footprint=f, mode="nearest") for f in line_footprints(31, LINE_SLOPES)], 0)
    oc = np.min([ndimage.grey_closing(op, footprint=f, mode="nearest") for f in line_footprints(21, LINE_SLOPES)], 0)
    Lf = oc[:, pad:-pad]
    def keep_lines(raw, filt):
        dil = ndimage.grey_dilation(filt, size=(3, 3))
        return np.minimum(raw, dil + 0.04 * ss(0.05, 0.2, dil))
    lines_u = keep_lines(Elu, Lf)
    # the back arc over the lace top: its lace is as bright as the ring's own lines,
    # so that stretch is carried over from both ends of it, along the ring
    s0, s1 = rc.s_of_t(LACE_T[0] - 3.0), rc.s_of_t(LACE_T[1] + 3.0)
    i0, i1 = int(math.floor(s0)), int(math.ceil(s1))
    lam = np.clip((np.arange(rc.ns) - i0) / (i1 - i0), 0.0, 1.0)
    carried = lines_u.copy()
    carried[:, i0:i1 + 1] = (1.0 - lam[i0:i1 + 1]) * lines_u[:, i0:i0 + 1] + lam[i0:i1 + 1] * lines_u[:, i1:i1 + 1]
    lines = keep_lines(El, rc.reroll(Lf, d, t))
    tdeg = np.degrees(t)
    lace = ss(LACE_T[0] - 3.0, LACE_T[0], tdeg) * ss(LACE_T[1] + 3.0, LACE_T[1], tdeg) * ss(70.0, 55.0, np.abs(d))
    lines = (1.0 - lace) * lines + lace * rc.reroll(carried, d, t)

    # where the light is unknown, the ring's glow off the band stays within what it
    # is where it is exact (30th to 90th percentile at that distance): this keeps
    # the outer glow over bright background and stops hair strands leaking in
    rel = (w > 0.9) & ~ndimage.binary_dilation(det, np.ones((9, 9))) & (lace < 0.01)
    grid = np.arange(-60.0, 102.0, 2.0)
    E_low = eh.gauss(E, 2.5)

    def profile(q_):
        v = np.array([np.percentile((E_low @ LW)[rel & (np.abs(d - g_) < 1.5)], q_)
                      if (rel & (np.abs(d - g_) < 1.5)).sum() > 50 else 0.0 for g_ in grid])
        return np.interp(d, grid, ndimage.gaussian_filter1d(v, 1.0))
    Lr = rc.reroll(Lf, d, t)
    # near the two ends the ring curves so tightly (radius about 10 px) that ring
    # coordinates fold over inside it: there the light is kept as measured
    cx_, cy_, a_, b_, _ = eh.RING
    rho = (a_ ** 2 * np.sin(t) ** 2 + b_ ** 2 * np.cos(t) ** 2) ** 1.5 / (a_ * b_)
    fold = ss(-0.5 * rho, -0.9 * rho, d) * ss(-45.0, -35.0, d)
    lines = fold * El + (1.0 - fold) * lines
    # faint glow is smooth light: the print dots of the edit's background showing
    # through it are not part of the halo
    lines = (1.0 - ss(0.25, 0.5, Lr)) * eh.gauss(np.minimum(lines, Lr + 0.02), 1.2) + ss(0.25, 0.5, Lr) * lines
    # where the light is unknown, keep only the bright line-like strands and lay
    # them over the ring's usual glow at that distance (its median where exact)
    strands = Lr * ss(-34.0, -24.0, d) * ss(40.0, 30.0, d)
    usual = profile(50) * ss(-60.0, -45.0, d) * ss(80.0, 60.0, d)
    usual = usual * (1.0 - lace) + lace * lines
    unknown = 1.0 - w
    lines = w * lines + unknown * np.maximum(strands, usual)
    # off the band the halo is only glow: nothing there may be brighter than the
    # brightest glow measured at that distance (lily and lace edges are long and
    # straight like the strands, so the line filter alone lets them through)
    off_band = np.maximum(ss(-30.0, -40.0, d), ss(34.0, 44.0, d))
    cap = profile(95) + 0.05
    lines = lines - off_band * np.maximum(lines - cap, 0.0)
    # and a glow only fades with distance: along every line across the ring, light
    # off the band can not rise again further out (that is background showing through)
    Lu = rc.unroll(lines)
    rows = rc.d_u
    out_i, in_i = np.searchsorted(rows, 34.0), np.searchsorted(rows, -30.0)
    env = Lu.copy()
    env[out_i:] = np.minimum.accumulate(Lu[out_i:], axis=0)
    env[:in_i + 1] = np.minimum.accumulate(Lu[:in_i + 1][::-1], axis=0)[::-1]
    env = ndimage.uniform_filter1d(ndimage.maximum_filter1d(env, 5, axis=1), 5, axis=1)
    lines = np.minimum(lines, rc.reroll(env, d, t) + 0.01 + (1.0 - off_band))
    # the thin outer arc around the right end is the halo's own (bright, crisp)
    arc = (ss(OUTER_ARC_T[0] - 3.0, OUTER_ARC_T[0], tdeg) * ss(OUTER_ARC_T[1] + 3.0, OUTER_ARC_T[1], tdeg)
           * ss(OUTER_ARC_D[0] - 4.0, OUTER_ARC_D[0], d) * ss(OUTER_ARC_D[1] + 4.0, OUTER_ARC_D[1], d))
    lines = np.maximum(lines, arc * El * ss(0.38, 0.62, El))

    # colour: exact where the light is exact; elsewhere the halo's colour at that
    # brightness (lavender when faint, white when strong), measured where it is exact
    sel = rel & (El > 0.03) & (np.abs(d) < 60)
    lev = np.clip((El[sel] * 64).astype(int), 0, 63)
    cm = np.full((64, 3), np.nan)
    for i in range(64):
        m = lev == i
        if m.sum() > 40:
            cm[i] = np.median(E[sel][m] / np.maximum(El[sel][m, None], 1e-3), 0)
    ok = ~np.isnan(cm[:, 0])
    cm = np.stack([np.interp(np.arange(64), np.arange(64)[ok], cm[ok, c]) for c in range(3)], 1)
    cm = ndimage.gaussian_filter1d(cm, 1.5, axis=0)
    cm /= (cm @ LW)[:, None]

    def colour(lv):
        return np.clip(lv[..., None] * cm[np.clip((lv * 63).astype(int), 0, 63)], 0.0, 1.0)

    exact = w[..., None] * (1.0 - lace[..., None])
    kept = E * np.minimum(lines / np.maximum(El, 1e-4), 1.0)[..., None]
    rgb_exact = eh.screen(kept, colour(np.maximum(lines - El, 0.0)))
    rgb = exact * rgb_exact + (1.0 - exact) * colour(lines)

    # sparkle dust: small bright specks the line filter took out, where the light is
    # exact (print dots of the background are fainter and are left out)
    dust_l = np.maximum(El - lines, 0.0)
    dust = E * (ss(0.40, 0.60, dust_l) * ss(0.5, 0.8, w) * ss(60.0, 45.0, np.abs(d)) * (1.0 - lace))[..., None]
    rgb = eh.screen(rgb, dust)

    # stars and glitter: components that peak like sparkles (lace and lily bits peak lower)
    Dk = ss(0.30, 0.55, e_det @ LW)
    lab, n = ndimage.label(ndimage.binary_dilation(Dk > 0.02, np.ones((3, 3))))
    peak = ndimage.maximum(e_det @ LW, lab, np.arange(1, n + 1))
    Dk = Dk * np.concatenate([[False], peak >= 0.85])[lab] * (1.0 - lace)
    glitter = E * Dk[..., None]

    # the two stars keep their light exactly as it is, glow included
    lab, _ = ndimage.label(Dk > 0.02, np.ones((3, 3)))
    star_zones = []
    for sx_, sy_ in STARS:
        near = lab[int(sy_) - 12:int(sy_) + 13, int(sx_) - 12:int(sx_) + 13]
        m = np.isin(lab, np.unique(near[near > 0]))
        zone = eh.gauss(ndimage.binary_dilation(m, iterations=35).astype(np.float64), 4.0)
        star_zones.append(zone * ss(18.0, 28.0, d))                  # outside the band only
        glitter = glitter * (1.0 - m)[..., None]
    stars = [np.clip(E * z[..., None], 0.0, 1.0) for z in star_zones]
    star_mask = np.max(star_zones, 0)

    # vertical flare: brightness above its neighbours on both sides, along its line
    Rl = R @ LW
    x0, x1 = FLARE_X
    r0, r1 = FLARE_ROWS
    strip = Rl[r0:r1, x0 - 10:x1 + 10]
    side = np.maximum(ndimage.median_filter(strip, footprint=np.r_[np.ones(6), np.zeros(15)][None, :]),
                      ndimage.median_filter(strip, footprint=np.r_[np.zeros(15), np.ones(6)][None, :]))
    th = np.clip((strip - side) / np.maximum(1.0 - side, 0.05), 0.0, 1.0)[:, 10:-10]
    rows = np.arange(r0, r1)
    cols = np.argmax(th, 1) + x0
    good = th.max(1) > 0.15
    fit = np.polyfit(rows[good], cols[good], 1)                       # x of the line per row
    good &= np.abs(np.polyval(fit, rows) - cols) < 2.0                # refit without outliers
    fit = np.polyfit(rows[good], cols[good], 1)
    xc = np.polyval(fit, rows)
    on = np.clip(np.round(xc).astype(int) - x0, 1, x1 - x0 - 2)
    line = np.max([th[np.arange(len(rows)), on + k] for k in (-1, 0, 1)], 0)
    line = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(line, 9), 6.0)   # smooth envelope
    line *= ss(0.08, 0.16, line)
    F = np.zeros(R.shape[:2])
    xs = np.arange(x0 - 10, x1 + 10)
    F[r0:r1, x0 - 10:x1 + 10] = line[:, None] * np.exp(-0.5 * ((xs[None, :] - xc[:, None]) / 0.9) ** 2)
    near_white = 0.6 + 0.4 * cm[40]                                  # the flare is nearly white
    flare = np.clip(F[..., None] * near_white / (near_white @ LW), 0.0, 1.0)

    ring = np.clip(eh.screen(rgb * (1.0 - star_mask)[..., None], glitter), 0.0, 1.0)
    light = ring
    for st in stars:
        light = eh.screen(light, st)
    light = np.clip(eh.screen(light, flare), 0.0, 1.0)

    fr = np.arange(0, 150)                                           # flare meets the back arc
    fy = fr[np.argmin(np.abs(d[fr, np.clip(np.round(np.polyval(fit, fr)).astype(int), 0, R.shape[1] - 1)]))]
    parts = [(st, xy) for st, xy in zip(stars, STARS)] + [(flare, (float(np.polyval(fit, fy)), float(fy)))]
    shadow = np.clip(D @ LW, 0.0, 1.0)
    return light, ring, parts, shadow, hl, dict(colour_map=cm, flare=fit)


def forward(q, src_pt, dst_pt, scale, rotate, mirror):
    """Where edit point q lands on the picture."""
    u, v = (q[0] - src_pt[0]) * (-1.0 if mirror else 1.0), q[1] - src_pt[1]
    c, s = math.cos(math.radians(rotate)), math.sin(math.radians(rotate))
    return dst_pt[0] + scale * (c * u - s * v), dst_pt[1] + scale * (s * u + c * v)


def place(layer, fill, shape, src_pt, center, scale, rotate, mirror):
    """Move a layer from edit coordinates onto the picture: mirror (optional), scale
    and turn around src_pt, and put src_pt at `center`."""
    H, W = shape
    cx, cy = src_pt
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    c, s = math.cos(math.radians(rotate)), math.sin(math.radians(rotate))
    u, v = xx - center[0], yy - center[1]
    u, v = (c * u + s * v) / scale, (-s * u + c * v) / scale        # undo turn and scale
    if mirror:
        u = -u
    sx, sy = cx + u, cy + v
    chans = layer if layer.ndim == 3 else layer[..., None]
    if scale < 1.0:                                                  # shrinking: anti-alias first
        sig = 0.5 * math.sqrt(1.0 / scale ** 2 - 1.0)
        chans = ndimage.gaussian_filter(chans, (sig, sig, 0))
    out = np.stack([ndimage.map_coordinates(chans[..., k], [sy, sx], order=3, mode="constant", cval=fill)
                    for k in range(chans.shape[-1])], -1)
    return out if layer.ndim == 3 else out[..., 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture")
    ap.add_argument("edit")
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--cx", type=float, default=CENTER[0])
    ap.add_argument("--cy", type=float, default=CENTER[1])
    ap.add_argument("--scale", type=float, default=SCALE)
    ap.add_argument("--rotate", type=float, default=ROTATE)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--debug", default=None, help="save the halo layers to this folder")
    args = ap.parse_args()

    P = np.asarray(Image.open(args.picture).convert("RGB")).astype(np.float64) / 255.0
    R = np.asarray(Image.open(args.edit).convert("RGB")).astype(np.float64) / 255.0
    S = np.asarray(Image.open(args.source).convert("RGB")).astype(np.float64) / 255.0

    light, ring, upright, shadow, hl, info = halo_light(R, S)

    away = (hl["d_old"] > 160) & ~ndimage.binary_dilation(hl["det"], np.ones((15, 15)))
    away[:, -eh.FRINGE_COLS:] = False                                # green fringe
    lut = gradient_map(R, away)

    luma = P @ LW
    plain = apply_map(lut, luma)
    geo = (eh.RING[:2], (args.cx, args.cy), args.scale, args.rotate, args.mirror)
    D_t = place(shadow, 1.0, luma.shape, *geo)
    if args.rotate == 0.0:                                           # the halo exactly as it is
        L_t = place(light, 0.0, luma.shape, *geo)
    else:                                                            # stars and flare stay upright
        L_t = place(ring, 0.0, luma.shape, *geo)
        for part, anchor in upright:
            L_t = eh.screen(L_t, place(part, 0.0, luma.shape, anchor, forward(anchor, *geo),
                                       args.scale, 0.0, args.mirror))
    amt = eh.RESHARPEN * min(max(args.scale - 1.0, 0.0) / 0.3, 1.0)  # enlarging softens
    if amt > 0:
        L_t = L_t + amt * (L_t - eh.gauss(L_t, 1.0))
    L_t = np.clip(L_t, 0.0, 1.0)
    D_t = np.clip(D_t, 0.0, 1.0)
    out = eh.screen(apply_map(lut, luma * D_t), L_t)

    # pixels the halo does not reach keep the plain recolour exactly
    untouched = (L_t.max(-1) < 0.5 / 255) & (np.abs(D_t - 1.0) < 1e-4)
    out[untouched] = plain[untouched]

    # check: the map keeps the picture's brightness (up to rounding and clipping at white)
    y = np.linspace(0.0, 0.95, 96)
    assert np.abs(apply_map(lut, y) @ LW - y).max() < 2.0 / 255, "gradient map changes brightness"

    Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(args.output)
    print(f"saved {args.output} ({out.shape[1]}x{out.shape[0]}), halo touches {(~untouched).mean():.1%}")
    if args.debug:
        sv = lambda n, im: Image.fromarray((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)).save(f"{args.debug}/{n}.png")
        sv("light_edit", light)
        sv("shadow_edit", shadow)
        sv("plain", plain)
        sv("light_placed", L_t)


if __name__ == "__main__":
    main()
