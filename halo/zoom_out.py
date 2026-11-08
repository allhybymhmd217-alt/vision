#!/usr/bin/env python3
"""Zoom out: extend the canvas around the picture and fill the new border with
detail taken from the picture itself.

The picture's interior stays bit-exact. Only its outermost strip changes: the
source darkens towards its own edges (12-26 px), so those strips are rebuilt to
avoid a dark line where the old edge was, and a seam may run up to 14 px inside
that wherever the new content matches.

1. The picture's halftone screen (two cosine gratings at +/-45 degrees,
   0.1801/0.1802 cycles/px) is separated from the image by local
   demodulation, giving a clean "descreened" image plus per-pixel dot
   amplitudes.
2. The border is grown from the inside out with image quilting: real 80x80
   blocks of the picture are chosen with FFT matching against what is already
   there and against a colour guide. Sources are limited to the neighbourhood
   of the nearest edge point, so hair continues hair, lilies continue lilies
   and armour continues armour; the face and the sword are never used.
   Each block is stitched in along a minimum-error graph-cut seam and its tone
   is matched to its neighbours with screened Poisson blending. Blocks are
   copied, not averaged, so the new area keeps the picture's sharpness.
3. The colour guide carries the edge colours outward and lets them sink
   (multiplicatively) into deep navy, so the frame darkens gently and leaves a
   calm, dark space above the head for the halo.
4. A lens-like softening grows with the distance from the picture, and the
   halftone is re-applied on the picture's own global screen, so the dots run
   across the seam without a break and stay crisp everywhere.

Only numpy, scipy and Pillow are needed:

    python3 zoom_out.py source.jpg extended.png
"""

import argparse
import math
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.sparse.linalg import factorized

# canvas growth in source pixels (left, top, right, bottom)
MARGINS = (120, 236, 120, 36)

# the source's halftone screen
SCREEN = [(0.1801, 0.1802), (0.1801, -0.1802)]
DEMOD_SIGMA = 2.5

# never used as a source (source coordinates)
FACE_ELLIPSE = (520.0, 650.0, 330.0, 250.0)   # cx, cy, rx, ry
SWORD_RECT = (280, 790, 1010, 975)            # x0, y0, x1, y1

BLOCK = 80
OVERLAP = 26
STEP = BLOCK - OVERLAP
REPLACE_DEPTH = 14        # a seam may wander this far into the picture, where it matches
FADE_BANDS = (12, 20, 12, 26)   # the source darkens towards its own edges (l, t, r, b);
                                # those strips are rebuilt so no dark line shows at the seam
DEFOCUS_MAX = 1.8         # gentle lens-like softening far from the picture (px sigma)
GUIDE_SIGMA = 8.0
GUIDE_WEIGHT = 0.35
TOLERANCE = 0.06          # pick randomly among blocks this close to the best one
TINT = np.array([0.40, 0.43, 0.60])   # what full darkening multiplies by (deep, blue-leaning)
DARKEN_DIST = (210.0, 150.0, 210.0, 60.0)   # left, top, right, bottom
DARKEN_MAX = (0.50, 0.72, 0.50, 0.35)
SCREENED_POISSON = 0.0007  # seam tone correction fades over ~40 px inside a block
LOCAL_WINDOW = (320, 420)   # sources come from this close (y, x) to the nearest edge point,
                            # so hair continues hair, lilies continue lilies, armour armour


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# halftone separation
# --------------------------------------------------------------------------
def carrier_phases(h, w, oy=0.0, ox=0.0):
    """Carrier phase of each screen grating, in source pixel coordinates."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy -= oy
    xx -= ox
    return [2.0 * math.pi * (f[0] * xx + f[1] * yy) for f in SCREEN]


def descreen(img, origin=(0.0, 0.0)):
    """Split img into a dot-free image and complex dot amplitudes (k, h, w, 3).
    `origin` is where the original picture's top-left sits in img (x, y).
    Normalised convolution keeps the estimate unbiased right up to the edges."""
    h, w, _ = img.shape
    ones = np.pad(np.ones((h, w)), 8)
    norm = ndimage.gaussian_filter(ones, DEMOD_SIGMA, mode="constant")[8:-8, 8:-8]
    amps = np.zeros((len(SCREEN), h, w, 3), np.complex128)
    halftone = np.zeros_like(img)
    for k, theta in enumerate(carrier_phases(h, w, origin[1], origin[0])):
        e = np.exp(-1j * theta)
        for c in range(3):
            v = np.pad(img[..., c] * e, 8)
            re = ndimage.gaussian_filter(v.real, DEMOD_SIGMA, mode="constant")[8:-8, 8:-8]
            im = ndimage.gaussian_filter(v.imag, DEMOD_SIGMA, mode="constant")[8:-8, 8:-8]
            amps[k, ..., c] = 2.0 * (re + 1j * im) / norm
        halftone += (amps[k] * np.exp(1j * theta)[..., None]).real
    return img - halftone, amps


def remodulate(amps, phases):
    out = np.zeros(amps.shape[1:], np.float64)
    for k, theta in enumerate(phases):
        out += (amps[k] * np.exp(1j * theta)[..., None]).real
    return out


# --------------------------------------------------------------------------
# smooth extrapolation (for the colour guide)
# --------------------------------------------------------------------------
def pool2(a):
    h, w = a.shape[:2]
    return a[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2, *a.shape[2:]).mean((1, 3))


def push_pull_fill(img, known):
    if known.all():
        return img
    wk = known.astype(np.float64)
    pyr = [(img * wk[..., None], wk)]
    while min(pyr[-1][1].shape) > 2 and (pyr[-1][1] == 0).any():
        n, k = pyr[-1]
        hh, ww = k.shape
        n = np.pad(n, ((0, hh % 2), (0, ww % 2), (0, 0)), mode="edge")
        k = np.pad(k, ((0, hh % 2), (0, ww % 2)), mode="edge")
        pyr.append((pool2(n) * 4, pool2(k) * 4))
    n, k = pyr[-1]
    cur = n / np.maximum(k, 1e-9)[..., None]
    for n, k in reversed(pyr[:-1]):
        hh, ww = k.shape
        up = np.repeat(np.repeat(cur, 2, 0), 2, 1)[:hh, :ww]
        up = ndimage.uniform_filter(up, size=(3, 3, 1), mode="nearest")
        alpha = np.clip(k, 0, 1)[..., None]
        cur = alpha * (n / np.maximum(k, 1e-9)[..., None]) + (1 - alpha) * up
    out = img.copy()
    out[~known] = cur[~known]
    return out


# --------------------------------------------------------------------------
# image quilting with FFT matching
# --------------------------------------------------------------------------
def fast_len(n):
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def box_sum(a, b):
    s = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return s[b:, b:] - s[:-b, b:] - s[b:, :-b] + s[:-b, :-b]


class Quilter:
    """Scores every source block against a partly known template in one FFT pass."""

    def __init__(self, feat, guide, allowed):
        self.h, self.w, self.C = feat.shape
        self.P = (fast_len(self.h + BLOCK), fast_len(self.w + BLOCK))
        rf = lambda a: np.fft.rfft2(a, self.P)
        self.FS = [rf(feat[..., c]) for c in range(self.C)]
        self.FS2 = rf((feat ** 2).sum(-1))
        self.FG = [rf(guide[..., c]) for c in range(3)]
        self.G2 = box_sum((guide ** 2).sum(-1), BLOCK)
        bad = box_sum((~allowed).astype(np.float64), BLOCK)
        self.valid = bad < 0.5
        self.oh, self.ow = self.G2.shape

    def cost(self, T, M, Gt):
        rf = lambda a: np.fft.rfft2(a, self.P)
        spec = self.FS2 * np.conj(rf(M))
        for c in range(self.C):
            spec -= 2.0 * self.FS[c] * np.conj(rf(M * T[..., c]))
        for c in range(3):
            spec -= 2.0 * GUIDE_WEIGHT * self.FG[c] * np.conj(rf(Gt[..., c]))
        E = np.fft.irfft2(spec, self.P)[: self.oh, : self.ow]
        E += (M[..., None] * T ** 2).sum() + GUIDE_WEIGHT * ((Gt ** 2).sum() + self.G2)
        E[~self.valid] = np.inf
        return E


def graphcut_new_mask(err, must_old, must_new):
    """Min-cut labelling of a block: True where the new patch should win.
    Edge cost between neighbours is the summed old/new mismatch (Kwatra 2003)."""
    bh, bw = err.shape
    n = bh * bw
    src, snk = n, n + 1
    idx = np.arange(n).reshape(bh, bw)
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    c, d = idx[:-1, :].ravel(), idx[1:, :].ravel()
    ch = (err[:, :-1] + err[:, 1:]).ravel()
    cv = (err[:-1, :] + err[1:, :]).ravel()
    u = [a, b, c, d]
    v = [b, a, d, c]
    cap = [ch, ch, cv, cv]
    cap = [np.maximum(1, (x * 1000.0).astype(np.int64)) for x in cap]
    big = 10 ** 8
    so, si = idx[must_old], idx[must_new]
    u += [np.full(so.size, src), si]
    v += [so, np.full(si.size, snk)]
    cap += [np.full(so.size, big), np.full(si.size, big)]
    G = csr_matrix((np.concatenate(cap).astype(np.int32),
                    (np.concatenate(u), np.concatenate(v))), shape=(n + 2, n + 2))
    flow = maximum_flow(G, src, snk).flow
    resid = (G - flow).tocsr()
    resid.data[resid.data < 0] = 0
    resid.eliminate_zeros()
    reach = breadth_first_order(resid, src, directed=True, return_predecessors=False)
    old = np.zeros(n + 2, bool)
    old[reach] = True
    return ~old[:n].reshape(bh, bw)


def poisson_correct(new, old, take, has_old):
    """Gradient-domain seam correction (screened Poisson blending): keep the new
    block's detail but add a smooth offset so it meets the old content along
    the whole cut, wherever the cut runs."""
    bh, bw = take.shape
    keep = has_old & ~take
    n = int(take.sum())
    if n == 0 or not keep.any():
        return new
    ids = np.full((bh, bw), -1)
    ids[take] = np.arange(n)
    ys, xs = np.nonzero(take)
    gap = old - new                     # Dirichlet values on the kept side of the cut
    rows, cols, vals = [], [], []
    rhs = np.zeros((n, new.shape[-1]))
    deg = np.full(n, SCREENED_POISSON)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        inb = (ny >= 0) & (ny < bh) & (nx >= 0) & (nx < bw)
        ny, nx = np.clip(ny, 0, bh - 1), np.clip(nx, 0, bw - 1)
        in_take = inb & take[ny, nx]
        in_keep = inb & keep[ny, nx]
        deg += in_take | in_keep
        i = np.nonzero(in_take)[0]
        rows.append(i)
        cols.append(ids[ny[i], nx[i]])
        vals.append(-np.ones(i.size))
        j = np.nonzero(in_keep)[0]
        np.add.at(rhs, j, gap[ny[j], nx[j]])
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(deg)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsc()
    solve = factorized(A)
    out = new.copy()
    for c in range(new.shape[-1]):
        out[ys, xs, c] += solve(rhs[:, c])
    return out


def quilt(canvas, filled, original, guide, src_feat, src_copy, src_guide, allowed, rng, offset):
    """Grow the unfilled part of `canvas` (H, W, C_copy) block by block.
    The first src_feat.shape[-1] channels of the canvas are the matching features.
    `offset` is the (top, left) position of the source inside the canvas."""
    H, W = filled.shape
    hs, ws = src_feat.shape[:2]
    q = Quilter(src_feat, src_guide, allowed)
    nf = src_feat.shape[-1]
    # the picture may only be replaced in its outermost REPLACE_DEPTH pixels
    locked = original & (ndimage.distance_transform_edt(original) > REPLACE_DEPTH)

    ys = sorted(set(list(range(0, H - BLOCK + 1, STEP)) + [H - BLOCK]))
    xs = sorted(set(list(range(0, W - BLOCK + 1, STEP)) + [W - BLOCK]))
    # fill from the inside out: order by the signed distance of each block's centre
    # to the picture, so blocks resting mostly on the picture are placed first
    sdist = ndimage.distance_transform_edt(~original) - ndimage.distance_transform_edt(original)
    blocks = []
    for y0 in ys:
        for x0 in xs:
            if (~filled[y0:y0 + BLOCK, x0:x0 + BLOCK]).any():
                blocks.append((sdist[y0 + BLOCK // 2, x0 + BLOCK // 2], y0, x0))
    blocks.sort()
    log(f"quilting {len(blocks)} blocks")

    used = []  # (canvas y, x, source y, x) of placed blocks
    for i, (_, y0, x0) in enumerate(blocks):
        sl = (slice(y0, y0 + BLOCK), slice(x0, x0 + BLOCK))
        unf = ~filled[sl]
        if not unf.any():
            continue
        M = (~unf).astype(np.float64)
        T = canvas[sl][..., :nf] * M[..., None]
        E = q.cost(T, M, guide[sl])
        # only borrow from the neighbourhood of the nearest point of the picture
        ry = int(np.clip(y0 + BLOCK // 2 - offset[0], 0, hs - 1)) - BLOCK // 2
        rx = int(np.clip(x0 + BLOCK // 2 - offset[1], 0, ws - 1)) - BLOCK // 2
        wy, wx = LOCAL_WINDOW
        E[:max(ry - wy, 0), :] = np.inf
        E[max(ry + wy, 0):, :] = np.inf
        E[:, :max(rx - wx, 0)] = np.inf
        E[:, max(rx + wx, 0):] = np.inf
        # keep neighbouring blocks from cloning the same source spot
        for cy, cx, sy, sx in used:
            if abs(cy - y0) < 3 * STEP and abs(cx - x0) < 3 * STEP:
                E[max(sy - 16, 0):sy + 17, max(sx - 16, 0):sx + 17] = np.inf
        best = E.min()
        cand = np.argwhere(E <= best + TOLERANCE * abs(best) + 1e-3 * BLOCK * BLOCK * 1e-2)
        sy, sx = cand[rng.integers(len(cand))]
        used.append((y0, x0, sy, sx))

        patch = src_copy[sy:sy + BLOCK, sx:sx + BLOCK]
        err = np.sqrt(((canvas[sl][..., :nf] - patch[..., :nf]) ** 2).sum(-1))
        must_new = unf.copy()
        must_old = locked[sl] & ~unf
        edge = np.zeros_like(unf)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        must_old |= edge & ~unf          # keep old content where it meets the block border
        take = graphcut_new_mask(err, must_old, must_new)
        patch = poisson_correct(patch, canvas[sl], take, ~unf)
        alpha = ndimage.gaussian_filter(take.astype(np.float64), 0.7)
        alpha[must_new] = 1.0
        alpha[must_old] = 0.0
        canvas[sl] = alpha[..., None] * patch + (1.0 - alpha[..., None]) * canvas[sl]
        filled[sl] = True
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(blocks)}")
    return canvas


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    source = Image.open(args.input)
    icc = source.info.get("icc_profile")
    img = np.asarray(source.convert("RGB")).astype(np.float64) / 255.0
    h, w, _ = img.shape
    L, T, R, B = MARGINS
    H, W = h + T + B, w + L + R
    log(f"source {w}x{h} -> canvas {W}x{H}")

    clean, amps = descreen(img)
    amp_ch = np.concatenate([np.stack([a.real, a.imag], -1).reshape(h, w, 6) for a in amps], -1)
    mag = np.sqrt((amp_ch ** 2).sum(-1, keepdims=True) / 6.0)

    # source stacks: matching features first, then everything that gets copied
    src_feat = np.concatenate([clean, mag], -1)
    src_copy = np.concatenate([clean, mag, amp_ch], -1)
    src_guide = ndimage.gaussian_filter(clean, (GUIDE_SIGMA, GUIDE_SIGMA, 0))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy, rx, ry = FACE_ELLIPSE
    face = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0
    x0, y0, x1, y1 = SWORD_RECT
    sword = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    fl, ft, fr, fb = FADE_BANDS
    faded = np.ones((h, w), bool)
    faded[ft:h - fb, fl:w - fr] = False
    allowed = ~(face | sword | faded)

    original = np.zeros((H, W), bool)
    original[T + ft:T + h - fb, L + fl:L + w - fr] = True   # the trusted part of the picture

    canvas = np.zeros((H, W, src_copy.shape[-1]))
    canvas[T:T + h, L:L + w] = src_copy
    filled = original.copy()

    # colour guide: edge colours carried outward, sinking into deep navy
    low = np.zeros((H, W, 3))
    low[T:T + h, L:L + w] = src_guide
    low = push_pull_fill(low, original)
    # smooth the extrapolation without letting the brighter interior leak into it
    wout = (~original).astype(np.float64)
    num = ndimage.gaussian_filter(low * wout[..., None], (GUIDE_SIGMA * 2, GUIDE_SIGMA * 2, 0))
    den = ndimage.gaussian_filter(wout, GUIDE_SIGMA * 2)[..., None]
    low[~original] = (num / np.maximum(den, 1e-6))[~original]
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
    side = np.stack([L - gx, T - gy, gx - (L + w - 1), gy - (T + h - 1)], 0)
    strength = np.zeros((H, W))
    for k in range(4):
        strength = np.maximum(strength, DARKEN_MAX[k] * smoothstep(0.0, DARKEN_DIST[k], side[k]))
    shade = 1.0 - strength[..., None] * (1.0 - TINT)   # multiplicative: darks stay dark
    guide = low * shade

    canvas = quilt(canvas, filled, original, guide, src_feat, src_copy, src_guide, allowed, rng,
                   offset=(T, L))

    # lens-like softening that grows with the distance from the picture; the dots
    # are added afterwards, so they stay as crisp as in the original print
    new_clean = canvas[..., :3]
    far = ndimage.distance_transform_edt(~original)
    sig = DEFOCUS_MAX * smoothstep(30.0, 200.0, far)
    levels = [0.0, 0.6, 1.2, DEFOCUS_MAX]
    stack = [new_clean] + [ndimage.gaussian_filter(new_clean, (s_, s_, 0)) for s_ in levels[1:]]
    soft = np.zeros_like(new_clean)
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        wgt = np.clip((sig - lo) / (hi - lo), 0, 1)
        inside = (sig >= lo) & (sig <= hi)
        soft[inside] = ((1 - wgt)[..., None] * stack[i] + wgt[..., None] * stack[i + 1])[inside]
    new_clean = soft
    a = canvas[..., 4:].reshape(H, W, len(SCREEN), 3, 2)
    new_amps = np.moveaxis(a[..., 0] + 1j * a[..., 1], 2, 0)
    out = new_clean + remodulate(new_amps, carrier_phases(H, W, T, L))

    # a final gentle darkening towards the outer edge, matching the guide
    out = out * (1.0 - 0.35 * strength[..., None] * (1.0 - TINT))

    out = np.clip(out, 0.0, 1.0)
    # inside the picture, anything deeper than the seam zone is the exact original
    inner = original & (ndimage.distance_transform_edt(original) > REPLACE_DEPTH)
    orig_canvas = np.zeros((H, W, 3))
    orig_canvas[T:T + h, L:L + w] = img
    out[inner] = orig_canvas[inner]
    res = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))
    extra = {"icc_profile": icc} if icc else {}
    res.save(args.output, **extra)
    log(f"saved {args.output}")


if __name__ == "__main__":
    main()
