#!/usr/bin/env python3
"""Give a black-and-white picture the colours and the halo of the user's edit.

1. Colours: a gradient map measured on edit.jpg away from its halo. For every
   brightness level it takes the edit's median colour, and the picture keeps
   its own brightness exactly: only the hue changes (navy blacks, periwinkle
   mids, lavender-white highlights).
2. Halo: lifted off edit.jpg as two layers, light (screened on) and the
   shadow the edit put behind the ring (multiplied on, before the colouring).
   - Under the ring the edit's background is source.jpg aligned to it, scaled
     by that shadow. The shadow is measured as edit / original away from the
     light, separately inside and outside the ring.
   - Ring and strands: the light above that background, kept where it is light
     the original does not have, or the band's white-hot core.
   - Stars and glitter: the detail mask of enlarge_halo.halo_layers, keeping
     only parts that peak like real sparkles (lace and lily bits peak lower).
   - The tall flare at the right end: a horizontal top-hat along its line.
   - Glow: sum of blurs of all of the above, fitted where the edit's
     background was dark (there it is measured exactly).
3. The halo is mirrored (this head leans the other way), turned to a 10 degree
   tilt (parallel to the eye and brow line) and placed over the head. The two
   stars and the flare stay upright and ride along with their point on the
   ring. Pixels away from the halo keep the plain recolour.

    python3 wings_halo.py wings.jpg edit.jpg source.jpg out.png [--cx X --cy Y --scale S]
"""

import argparse
import math

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

import enlarge_halo as eh
import zoom_out as zo

LW = np.array([0.2126, 0.7152, 0.0722])

# where the halo goes on the picture: ring centre, size relative to the edit's
# ring, extra turn (degrees, applied after mirroring; the mirrored ring alone
# tilts +18.1 degrees, right end down)
CENTER = (865.0, 122.0)
SCALE = 1.0
ROTATE = -8.0      # -> 10 degrees: parallel to the eye and brow line (measured 6-8
                   # degrees, right side lower), with a touch more for the hair's sweep
MIRROR = True

# the edit's two four-point stars; they and the vertical flare stay upright when
# the ring turns, and ride along with their point on the ring
STARS = ((184.0, 134.0), (153.0, 296.0))

# the top of the lace headpiece, lit by the halo in the edit: background, not halo
# (cx, cy, rx, ry in edit pixels); only the ring's core line is kept through it
LACE_TOP = (383.0, 72.0, 32.0, 30.0)
FLARE_X = (585, 612)      # columns that hold the edit's vertical flare
FLARE_ROWS = (0, 300)
GLOW_SIGMAS = (2.5, 7.0, 18.0, 45.0)


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


def halo_light(R, S):
    """Light (RGB) and shadow (luma factor) of the edit's halo, in edit coordinates."""
    hl = eh.halo_layers(R, S, np.random.default_rng(5))
    d, det, e_det, B_s = hl["d_old"], hl["det"], hl["e_det"], hl["B_s"]
    quiet = ~ndimage.binary_dilation(det, np.ones((15, 15)))

    # shadow: edit / original away from the light, filled in separately on each side
    ratio = np.clip(eh.gauss(R, 4.0) / np.maximum(eh.gauss(B_s, 4.0), 0.03), 0.2, 1.3)
    D = np.ones_like(R)
    for side, known in ((d < 0, (d < -32) & quiet), (d >= 0, ((d > 40) & quiet) | (d >= 160))):
        Dm = np.where(known[..., None], ratio, 1.0)
        Dm[d >= 160] = 1.0
        D[side] = eh.gauss(zo.push_pull_fill(Dm, known), 5.0)[side]
    bg = np.clip(D * B_s, 0.0, 1.0)
    E = np.clip(1.0 - (1.0 - R) / np.maximum(1.0 - bg, 1e-3), 0.0, 1.0)
    El, bgl, Rl = E @ LW, bg @ LW, R @ LW

    # ring and strands
    band = np.where(d < 0, ss(-30.0, -20.0, d), ss(36.0, 24.0, d))
    tau = 0.30 + 0.45 * ss(0.50, 0.85, bgl)
    added = Rl - ndimage.maximum_filter(B_s @ LW, size=5)      # light the original lacks
    core = ss(0.85, 0.95, El) * ss(14.0, 8.0, np.abs(d))
    keep = np.maximum(ss(0.04, 0.15, added), core)
    yy, xx = np.mgrid[0:R.shape[0], 0:R.shape[1]].astype(np.float64)
    lx, ly, lrx, lry = LACE_TOP
    lace = ss(1.0, 0.75, np.hypot((xx - lx) / lrx, (yy - ly) / lry)) * ss(7.0, 10.0, np.abs(d))
    C = E * (ss(tau, tau + 0.25, El) * band * keep * (1.0 - lace))[..., None]

    # stars and glitter: bright cores of components that peak like sparkles
    Dc = e_det * ss(0.30, 0.55, e_det @ LW)[..., None]
    lab, n = ndimage.label(ndimage.binary_dilation((Dc @ LW) > 0.02, np.ones((3, 3))))
    peak = ndimage.maximum(e_det @ LW, lab, np.arange(1, n + 1))
    Dc *= np.concatenate([[False], peak >= 0.85])[lab][..., None] * (1.0 - lace)[..., None]

    # vertical flare: brightness above its neighbours on both sides, along its line
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
    # strength along it: a smooth envelope (the texture under it makes it flicker)
    line = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(line, 9), 6.0)
    line *= ss(0.08, 0.16, line)
    F = np.zeros(R.shape[:2])
    xs = np.arange(x0 - 10, x1 + 10)
    F[r0:r1, x0 - 10:x1 + 10] = line[:, None] * np.exp(-0.5 * ((xs[None, :] - xc[:, None]) / 0.9) ** 2)

    # glow, fitted where the edit's background was dark
    src = np.clip(eh.screen(C, Dc) @ LW, 0.0, 1.0)
    src = 1.0 - (1.0 - src) * (1.0 - F)
    blurs = [eh.gauss(src, s) for s in GLOW_SIGMAS]
    sel = (bgl < 0.40) & (src < 0.05) & ~ndimage.binary_dilation(det, np.ones((9, 9))) & (d > -50) & (d < 140)
    sel &= np.random.default_rng(0).random(sel.shape) < 0.25
    alpha, _ = optimize.nnls(np.stack([bl[sel] for bl in blurs], 1), El[sel])
    glow_l = sum(a_ * bl for a_, bl in zip(alpha, blurs))
    gsel = sel & (glow_l > 0.08) & (El > 0.05)
    tint = (E[gsel] / np.maximum(El[gsel], 1e-3)[:, None]).mean(0)
    tint /= tint @ LW

    def with_glow(light):
        g = sum(a_ * eh.gauss(light @ LW, s_) for a_, s_ in zip(alpha, GLOW_SIGMAS))
        return np.clip(eh.screen(light, np.clip(g[..., None] * tint, 0.0, 1.0)), 0.0, 1.0)

    # split off the upright parts: each star component, and the flare
    lab, _ = ndimage.label((Dc @ LW) > 0.02, np.ones((3, 3)))
    stars = []
    for sx_, sy_ in STARS:
        near = lab[int(sy_) - 12:int(sy_) + 13, int(sx_) - 12:int(sx_) + 13]
        ids = np.unique(near[near > 0])
        m = np.isin(lab, ids)
        stars.append((with_glow(Dc * m[..., None]), (sx_, sy_)))
        Dc = Dc * ~m[..., None]
    fr = np.arange(0, 150)                                           # flare meets the back arc
    fy = fr[np.argmin(np.abs(d[fr, np.clip(np.round(np.polyval(fit, fr)).astype(int), 0, R.shape[1] - 1)]))]
    white_blue = 0.6 + 0.4 * tint                                    # flare: nearly white
    flare = (with_glow(F[..., None] * white_blue / (white_blue @ LW)), (float(np.polyval(fit, fy)), float(fy)))
    ring = with_glow(eh.screen(C, Dc))
    shadow = np.clip(D @ LW, 0.0, 1.0)
    return ring, stars + [flare], shadow, hl, dict(alpha=alpha, tint=tint, flare=fit)


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

    ring, upright, shadow, hl, info = halo_light(R, S)
    print("glow alphas", np.round(info["alpha"], 3), "tint", np.round(info["tint"], 3))

    away = (hl["d_old"] > 160) & ~ndimage.binary_dilation(hl["det"], np.ones((15, 15)))
    away[:, -eh.FRINGE_COLS:] = False                                # green fringe
    lut = gradient_map(R, away)

    luma = P @ LW
    plain = apply_map(lut, luma)
    geo = (eh.RING[:2], (args.cx, args.cy), args.scale, args.rotate, args.mirror)
    L_t = place(ring, 0.0, luma.shape, *geo)
    D_t = place(shadow, 1.0, luma.shape, *geo)
    for part, anchor in upright:                                     # stars and flare stay upright
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
        sv("light_edit", eh.screen(ring, np.max([p for p, _ in upright], 0)))
        sv("shadow_edit", shadow)
        sv("plain", plain)
        sv("light_placed", L_t)


if __name__ == "__main__":
    main()
