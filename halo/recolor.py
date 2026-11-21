#!/usr/bin/env python3
"""Recolour any picture with the palette of the user's edit, halo included.

The palette is the same gradient map wings_halo.py uses (edit.jpg's median
colour per brightness level, measured away from its halo). The picture keeps
its own brightness exactly; only the hue changes, so a halo already in the
picture turns white-lavender with periwinkle shading like the edit's.

    python3 recolor.py picture.jpg edit.jpg source.jpg out.png
"""

import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import enlarge_halo as eh
import wings_halo as wh


def main():
    picture, edit, source, output = sys.argv[1:5]
    load = lambda f: np.asarray(Image.open(f).convert("RGB")).astype(np.float64) / 255.0
    R, S, P = load(edit), load(source), load(picture)

    hl = eh.halo_layers(R, S, np.random.default_rng(5))
    away = (hl["d_old"] > 160) & ~ndimage.binary_dilation(hl["det"], np.ones((15, 15)))
    away[:, -eh.FRINGE_COLS:] = False                                # green fringe
    lut = wh.gradient_map(R, away)

    luma = P @ wh.LW
    out = wh.apply_map(lut, luma)
    keep = luma < 0.95                                               # near white clips
    assert np.abs(out @ wh.LW - luma)[keep].max() < 2.0 / 255, "brightness changed"
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(output)
    print(f"saved {output} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
