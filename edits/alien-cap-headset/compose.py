"""Put the cap and the drive-thru headset on the plush alien photo.

Usage:
    python3 compose.py [key=value ...]

Defaults reproduce result.png (headset on the image's right side).
Pass ear_side=left for the mirrored version (result-headset-left.png).
Inputs are read from src/ next to this script; output path via out=...
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
params = dict(a.split("=", 1) for a in sys.argv[1:])
P = lambda k, d: float(params.get(k, d))

HEAD_CX = 361.0                   # vertical axis of the head in the photo

# ---- cap ----
CAP_W = P("cap_w", 342)           # cap width in px (head is ~322 px wide above the glasses)
CAP_SQ = P("cap_sq", 0.92)        # vertical/horizontal aspect of the cap
BRIM_K = P("brim_k", 0.75)        # brim foreshortening (camera is near eye level)
CAP_CX = P("cap_cx", HEAD_CX)
CAP_BRIM_Y = P("cap_brim_y", 338) # brim lower edge, centre (glasses top is at ~346)
BLUR = P("blur", 2.2)             # matches the photo's optical blur (glasses edges)

# ---- headset ----
EAR_SIDE = params.get("ear_side", "right")  # side as seen by the viewer
EAR_SCALE = P("ear_scale", 1.2)
_right = EAR_SIDE == "right"
EAR_X = P("ear_x", 527 if _right else 2 * HEAD_CX - 527)
EAR_Y = P("ear_y", 404)
EAR_ROT = P("ear_rot", -8 if _right else 8)  # degrees, + = counter-clockwise
EAR_BLUR = P("ear_blur", 1.2)

OUT = params.get("out", str(HERE / ("result.png" if _right else "result-headset-left.png")))


def load_rgba(path):
    return np.array(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0


def premul(rgba):
    out = rgba.copy()
    out[..., :3] *= out[..., 3:4]
    return out


def unpremul(pre):
    a = pre[..., 3:4]
    return np.concatenate([np.where(a > 1e-4, pre[..., :3] / np.maximum(a, 1e-4), 0), a], -1)


def affine(scale_xy, rot_deg, src_pt, dst_pt):
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    A = R @ np.diag(scale_xy)
    t = np.asarray(dst_pt) - A @ np.asarray(src_pt)
    return np.hstack([A, t[:, None]]).astype(np.float32)


def place(rgba, M, size):
    """Warp a straight-alpha layer into photo space; returns premultiplied RGBA."""
    out = cv2.warpAffine(premul(rgba), M, size, flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return np.clip(out, 0, 1)


def grade(pre, gain, lift, sat):
    """Match the layer to the photo: slightly desaturated, warm ambient, lifted blacks."""
    rgb, a = unpremul(pre)[..., :3], pre[..., 3:4]
    lum = (rgb * np.array([0.299, 0.587, 0.114])).sum(-1, keepdims=True)
    rgb = np.clip(lum + (rgb - lum) * sat, 0, 1) * np.array(gain) + np.array(lift)
    out = pre.copy()
    out[..., :3] = np.clip(rgb, 0, 1) * a
    return out


def over(base_rgb, pre):
    return pre[..., :3] + base_rgb * (1 - pre[..., 3:4])


photo = cv2.cvtColor(cv2.imread(str(HERE / "src/alien.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
H, W = photo.shape[:2]

# ---------------- cap ----------------
cap = load_rgba(HERE / "src/cap.png")
ch, cw = cap.shape[:2]
CAP_SRC_CX, CAP_SRC_BRIM = 203.0, 282.0  # centre column / brim lower edge row in cap.png
if BRIM_K != 1.0:
    # compress rows below the crown/brim seam (~row 180) with a smooth transition
    y0, d = 180.0, 18.0
    ys = np.linspace(0, ch, 4 * ch + 1)
    slope = np.interp(ys, [0, y0 - d, y0 + d, ch], [1, 1, BRIM_K, BRIM_K])
    f = np.concatenate([[0], np.cumsum((slope[1:] + slope[:-1]) / 2 * np.diff(ys))])
    new_h = int(np.ceil(f[-1]))
    src_rows = np.interp(np.arange(new_h, dtype=np.float64), f, ys).astype(np.float32)
    map_x = np.tile(np.arange(cw, dtype=np.float32), (new_h, 1))
    map_y = np.tile(src_rows[:, None], (1, cw))
    pre = cv2.remap(premul(cap), map_x, map_y, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    cap = unpremul(np.clip(pre, 0, 1))
    CAP_SRC_BRIM = float(np.interp(CAP_SRC_BRIM, ys, f))

sx = CAP_W / cw
sy = sx * CAP_SQ
cap_pre = place(cap, affine((sx, sy), 0.0, (CAP_SRC_CX, CAP_SRC_BRIM), (CAP_CX, CAP_BRIM_Y)), (W, H))
cap_pre = grade(cap_pre, gain=(1.0, 0.97, 0.95), lift=(0.025, 0.018, 0.02), sat=0.85)
# scene is lit from above: crown top a little brighter than the brim
yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
top_y = CAP_BRIM_Y - CAP_SRC_BRIM * sy
cap_pre[..., :3] *= (1.0 + 0.10 * np.clip(1 - (yy - top_y) / (CAP_SRC_BRIM * sy), 0, 1))[..., None]
cap_pre = cv2.GaussianBlur(cap_pre, (0, 0), BLUR)

# ---------------- headset ----------------
ear = load_rgba(HERE / "src/earpiece.png")
eh, ew = ear.shape[:2]
ear_src = np.array([22.0, 52.0])  # point on the body that sits over the "ear"
if _right:  # the source boom points down-right; mirror it so it points to the mouth
    ear = ear[:, ::-1].copy()
    ear_src[0] = ew - 1 - ear_src[0]
ear_pre = place(ear, affine((EAR_SCALE, EAR_SCALE), EAR_ROT, ear_src, (EAR_X, EAR_Y)), (W, H))
ear_pre = grade(ear_pre, gain=(1.0, 0.98, 0.96), lift=(0.02, 0.015, 0.015), sat=0.9)
ear_pre = cv2.GaussianBlur(ear_pre, (0, 0), EAR_BLUR)

# ---------------- contact shadows (only on the character) ----------------
green = (photo[..., 1] > photo[..., 0] + 0.05) & (photo[..., 1] > photo[..., 2] + 0.05)
recv = np.clip(cv2.GaussianBlur(green.astype(np.float32), (0, 0), 3) * 1.5, 0, 1)
head_box = np.zeros((H, W), np.float32)
head_box[200:520, 160:560] = 1
glasses = ((photo.mean(-1) < 0.30) * head_box).astype(np.float32)
recv = np.clip(recv + cv2.GaussianBlur(glasses, (0, 0), 2), 0, 1)

cap_a = cap_pre[..., 3]
sh = cv2.warpAffine(cap_a, np.float32([[1, 0, 2], [0, 1, 12]]), (W, H))
cap_shadow = 0.65 * cv2.GaussianBlur(sh, (0, 0), 10) * (1 - cap_a) * recv

ear_a = ear_pre[..., 3]
sh2 = cv2.warpAffine(ear_a, np.float32([[1, 0, -3 if _right else 3], [0, 1, 6]]), (W, H))
ear_shadow = 0.45 * cv2.GaussianBlur(sh2, (0, 0), 5) * (1 - ear_a) * recv

base = photo * np.clip(1 - cap_shadow - ear_shadow, 0, 1)[..., None]

# ---------------- composite: headset sits under the brim corner ----------------
out = over(over(base, ear_pre), cap_pre)

# light grain on the pasted parts so they match the photo
rng = np.random.default_rng(7)
added = np.clip(cap_a + ear_a, 0, 1)[..., None]
out = np.clip(out + rng.normal(0, 0.008, (H, W, 1)).astype(np.float32) * added, 0, 1)

Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(OUT)
print("saved", OUT)
