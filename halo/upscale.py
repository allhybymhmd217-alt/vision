#!/usr/bin/env python3
"""Halftone-aware upscale for the zoomed-out picture.

A plain resize would blow the print dots up into soft blobs and enlarge the
JPEG blocks with them. Instead:

1. the dot screen is separated from the image (same demodulation as
   zoom_out.py),
2. the clean image is enlarged with Lanczos and given a gentle unsharp mask,
3. the dot amplitudes are enlarged smoothly, and the dots are re-drawn
   analytically at the new resolution on the picture's own global screen.

The result looks like the same print, just with more pixels and no
compression debris.

    python3 upscale.py extended.png extended_2x.png --scale 2 --screen-origin 120 236
"""

import argparse
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from zoom_out import SCREEN, descreen

SHARPEN_AMOUNT = 0.35
SHARPEN_SIGMA = 1.2   # in output pixels


def resize(channel, size, resample):
    return np.asarray(Image.fromarray(channel.astype(np.float32), mode="F").resize(size, resample),
                      dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--screen-origin", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    args = ap.parse_args()

    source = Image.open(args.input)
    icc = source.info.get("icc_profile")
    img = np.asarray(source.convert("RGB")).astype(np.float64) / 255.0
    h, w, _ = img.shape
    S = args.scale
    ox, oy = args.screen_origin
    H, W = h * S, w * S

    clean, amps = descreen(img, origin=(ox, oy))

    big = np.stack([resize(clean[..., c], (W, H), Image.LANCZOS) for c in range(3)], -1)
    big += SHARPEN_AMOUNT * (big - ndimage.gaussian_filter(big, (SHARPEN_SIGMA, SHARPEN_SIGMA, 0)))

    # source coordinates of every output pixel centre
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    sx = (xx + 0.5) / S - 0.5 - ox
    sy = (yy + 0.5) / S - 0.5 - oy
    for k, f in enumerate(SCREEN):
        carrier = np.exp(1j * 2.0 * math.pi * (f[0] * sx + f[1] * sy))
        for c in range(3):
            amp = (resize(amps[k, ..., c].real, (W, H), Image.BICUBIC)
                   + 1j * resize(amps[k, ..., c].imag, (W, H), Image.BICUBIC))
            big[..., c] += (amp * carrier).real

    out = Image.fromarray((np.clip(big, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    extra = {"icc_profile": icc} if icc else {}
    out.save(args.output, **extra)


if __name__ == "__main__":
    main()
