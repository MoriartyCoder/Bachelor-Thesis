"""Section "sota" of the augmentation config: state-of-the-art copy-paste augmentations.

* ACB (Tong et al. 2024), described below.
* PGDAA (PGDA adapted; Wang, Li, Huang, Wu, Jie, Tao: "Prior-Guided Data Augmentation for Infrared
  Small Target Detection", IEEE JSTARS 15 (2022) 10027-10040, Sec. III, Algorithm 1), described
  further below. Its target library is built by data_handler/target_library.py.

PGDAA, paper -> implementation:

* T (every training target, cut by its pixel mask) -> the target library: pseudo masks from local
  contrast, all datasets, no tag filter. Missing: targets too faint to segment.
* A (Eq. 1, area ratios < 0.15 %) -> box area / frame area of the run's training boxes below
  `max_area_ratio`. The pasted target is scaled so that its BOX has the drawn ratio (box against box;
  Eq. 1 counts target pixels, which only the library subset has).
* S (Eq. 2 SCR, 20 %-80 %) -> the library's `scr` column (Eq. 2 with the pseudo mask, ring d = 20),
  between `scr_percentiles`.
* B_O (pure-background subfigures, Algorithm 1 line 1) -> a random training frame without a box.
  Subfigure split: random crop with side fraction U(`crop_scale`), resized to the frame size.
* Area ratio beta from A, +-10 % (`area_jitter`); each target gets its own beta (Fig. 5). The paper
  changes the background resolution; with a fixed input size the target is scaled instead, which
  gives the same area ratio.
* gt() / nt(): flips, affine / elastic, grid distortion, piecewise affine (`target_ops`, strengths ours).
* gp(): weights = Gaussian-blurred mask (`gaussian_weights`, `gaussian_sigma`).
* SCR perception (lines 8-16): R = background window (patch +- d) at the position, before pasting;
  scr = (avg(T) - avg(R)) / std(R), SIGNED (the paper's goal: the target is always warmer, so a
  target darker than R is outside S); if min(S) < scr < max(S) unchanged, else
  T <- ColorJitter(T + S_i * std(R) - avg(T) + avg(R)), clipped to [0, 255].
* Only bright library targets ("no cases where the temperature of the target is lower").
* Offline synthesis (synthesis ratio r) -> online: with probability p the sample is REPLACED by a
  synthetic image whose boxes are only the pasted targets. p = r / (1 + r), e.g. r = 1 -> p = 0.5.
* Ablation (Table V): `subfigure_split: false`, `scr_perception: false`, `gaussian_sigma: 0`.
* Not given in the paper (our defaults): N_t (`num_targets` 1, our frames have one target),
  `crop_scale`, `gaussian_sigma`, strengths of the target ops, ColorJitter (brightness/contrast 0.1),
  random non-overlapping positions.

`TRANSFORMS` (end of the file) is the registry of the names allowed in the `sota:` section of a
run config.

ACB: auto-augmented copy-paste bounding-box augmentation.

Tong, Leng, Yang, Wang: "Target-Focused Enhancement Network for Distant Infrared Dim
and Small Target Detection", IEEE TGRS 62 (2024), 4109711, Sec. III-D, Fig. 4.

Per pasted target, the paper's three steps:

1. Bounding-box copy: crop a target of the *training set* together with its
   surrounding context (a box, not a segmentation mask, because in IR the target's
   radiation also affects the pixels around it). ACB is a method for dim and *small*
   targets (3-5 px in area; Sec. IV-D sets it apart from Copy-Paste for "normal-sized
   targets"), so only boxes below ``max_area_ratio`` of the frame are copied (0.15 %, the
   small-target limit also used by PGDAA). Large targets already in a frame stay untouched.
2. Random augment: RandAugment (Cubuk et al. 2020, ref. [27] of the paper) on the
   patch: ``num_ops`` operations drawn uniformly with replacement from ``ops``, every
   drawn operation is applied, all with the single global magnitude ``magnitude`` on
   RandAugment's 0..10 scale. The paper does not give N, M or the search ("automatically
   learns the optimal strategy", Fig. 4); the abstract describes ACB as "mapping augment
   strategies to 2-D space", which is RandAugment's (N, M) space. So the "auto" part here is
   RandAugment's grid search over (``num_ops``, ``magnitude``): one run config per grid point.
3. Adaptive paste onto a random position of the current training image (the
   "random background"), with I the background patch of the same size as the
   augmented grayscale target patch T:

       T'(x, y) = T(x, y) / max_xy T(x, y)          Eq. (6)
       I'       = T' * T + (1 - T') * I             Eq. (7)

   ``fusion: paper`` (default) is Eq. (6) as written. It assumes the paper's semi-simulated
   data (Sec. IV-A: synthetic targets with a Gaussian grey-value profile), whose crops have a
   dark context (Fig. 4). On our frames the context of a small target gets a median weight of
   0.44, so part of the source background is pasted along and the patch rectangle stays
   visible. Variant, not in the paper: ``fusion: background`` takes Eq. (6) relative to the
   patch's background, T' = max(T - b, 0) / max(T - b), b = median grey value of the context
   (patch outside the box); for a dark context (b ~ 0) it equals Eq. (6).

The paper enriches the dataset once ("initially enriched", Sec. III-A, Fig. 2; Table V gives the
target share of the enriched set). With ``fixed: true`` (default) a frame gets the same pastes in
every epoch (RNG seeded from run seed + frame), i.e. that fixed enriched set, built online.

The paper is a segmentation method (a pasted target gets its mask label); here every pasted
target gets a box. All deviations: augmentations/ACB_deviations.md.

The paper leaves the following open. Each choice is configurable and documented:

* context around the box: ``context`` (fraction of the box side on each side,
  at least ``min_context_px``). Default 1.0 and 8 px: the mask crops of Fig. 4 show
  0.6-1.9x the target size per side for extended targets and ~5x for a point target;
* number of pasted targets per image: ``num_targets`` (n or [min, max], uniform).
  Default 1 with p = 1: in Tables V/VI ACB doubles the share of target pixels
  (x2.02 / x2.06), with pastes from the same small-target distribution. One paste per
  frame doubles our target count (x2.07) and, on CST (small targets as in the paper),
  also the pixel share (x2.02). Overall the pixel share rises only x1.18, because the
  large Anti-UAV410 targets (not copied) stay; 500 random training frames, one paste;
* magnitude: default M = 5. At RandAugment's ImageNet optimum M = 9 the brightness
  factors 0.19 / 1.81 make many small targets vanish or saturate;
* operations: the paper names Rotate, Flip, Color, Scale, Brightness, Sharpness and
  "...". Default = the named ones + RandAugment's Identity. Color (saturation) is the
  identity on single-channel IR and is left out of the default. The other RandAugment
  operations can be added; Translate is not offered because the paste position is
  random anyway. Magnitude ranges follow torchvision's RandAugment (max: rotate 30 deg,
  shear 0.3, enhance factors 1 +/- 0.9, posterize 8 -> 4 bits, solarize 255 -> 0);
  Scale (not part of RandAugment) uses 1 +/- 0.5;
* geometric ops enlarge the patch canvas instead of cutting the patch. The uncovered
  canvas is 0, so Eq. (6) gives it weight 0 and the background stays there;
* the box after rotate/shear is the bounding box of the transformed ellipse inscribed
  in the original box (tight for compact targets like drones; the bounding box of the
  rotated corners would make a square box 37 % wider at 30 deg);
* a patch never overlaps an existing box or another pasted patch, otherwise a
  labelled target could be painted over.

Ablations (paper's Tables V/VI):
  ``num_ops: 0, fusion: none``  -> box copy only (the "CutMix" row: box, no augment, no fusion)
  defaults                      -> full ACB (box, random augment, adaptive fusion = Eq. (6))
  ``fusion: none`` / ``num_ops: 0`` alone -> one component switched off (not in the paper)
  ``fusion: background``        -> full ACB with the background-relative weight (not in the paper)

Works on grayscale uint8 HxW images (see augmentations/uint8.py) and on any Albumentations
bbox format; the new boxes get ``class_label`` in every label field. Point operations run
as lookup tables and filters in OpenCV on uint8; only the fusion (Eq. 7) computes in float
once. Results are rounded (PIL/torchvision truncate, which darkens by ~0.5 grey values).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.core.type_definitions import Targets
from PIL import Image

from augmentations.uint8 import RAMP, check_gray_uint8, to_uint8
from config import LIBRARY_DIR
from data_handler.formats import FrameRecord
from log_setup import get_logger

logger = get_logger(__name__)

MAX_MAGNITUDE = 10.0  # RandAugment: "integer scale from 0 to 10"; larger values extrapolate
MIN_BOX_PX = 2.0  # Scale never shrinks a target below this size (in pixels)
FUSIONS = ("paper", "background", "none")
DEFAULT_OPS = ("identity", "rotate", "flip", "scale", "brightness", "sharpness")


@dataclass
class Patch:
    """A cropped target patch (grayscale) while it is being augmented."""

    img: np.ndarray  # uint8 HxW, 0 wherever `valid` is False
    valid: np.ndarray  # bool HxW, False where a geometric op uncovered the canvas
    box: tuple[float, float, float, float]  # target (cx, cy, w, h); pixel (0, 0) spans [0, 1) x [0, 1)


# ---------------------------------------------------------------------------
# Step 1: bounding-box copy
# ---------------------------------------------------------------------------
def crop_target(record: FrameRecord, context: float, min_context_px: float) -> Patch | None:
    """Target box of `record` plus `context` * box size (at least `min_context_px`) on every side."""
    img = cv2.imread(record.img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"ACB: cannot read {record.img_path}, source skipped.")
        return None
    H, W = img.shape
    x, y, w, h = record.bbox
    x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + w, W), min(y + h, H)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    pad_x = max(context * (x1 - x0), min_context_px)
    pad_y = max(context * (y1 - y0), min_context_px)
    cx0, cy0 = max(0, math.floor(x0 - pad_x)), max(0, math.floor(y0 - pad_y))
    cx1, cy1 = min(W, math.ceil(x1 + pad_x)), min(H, math.ceil(y1 + pad_y))
    patch = img[cy0:cy1, cx0:cx1].copy()
    box = ((x0 + x1) / 2 - cx0, (y0 + y1) / 2 - cy0, x1 - x0, y1 - y0)
    return Patch(patch, np.ones(patch.shape, dtype=bool), box)


# ---------------------------------------------------------------------------
# Step 2: random augment (RandAugment operations on one patch)
# `m` is the signed magnitude as a fraction of the maximum: magnitude / MAX_MAGNITUDE.
# ---------------------------------------------------------------------------
def _affine(p: Patch, lin: np.ndarray) -> Patch:
    """Apply the 2x2 linear map `lin`; the canvas grows so that no patch pixel is cut off."""
    h, w = p.img.shape
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64) @ lin.T
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    out_w = max(1, math.ceil(hi[0] - lo[0] - 1e-6))
    out_h = max(1, math.ceil(hi[1] - lo[1] - 1e-6))
    t = -lo
    # Our coordinates put pixel centres on .5, cv2 puts them on integers.
    mat = np.hstack([lin, (lin @ [0.5, 0.5] + t - 0.5)[:, None]])
    img = cv2.warpAffine(p.img, mat, (out_w, out_h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(p.valid.astype(np.uint8), mat, (out_w, out_h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    img[~valid] = 0
    cx, cy, bw, bh = p.box
    ncx, ncy = lin @ [cx, cy] + t
    a, b = bw / 2, bh / 2  # half axes of the ellipse inscribed in the box
    nw = 2 * math.hypot(lin[0, 0] * a, lin[0, 1] * b)
    nh = 2 * math.hypot(lin[1, 0] * a, lin[1, 1] * b)
    return Patch(img, valid, (float(ncx), float(ncy), nw, nh))


def _rotate(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    rad = math.radians(30.0 * m)
    c, s = math.cos(rad), math.sin(rad)
    return _affine(p, np.array([[c, -s], [s, c]]))


def _shear_x(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    return _affine(p, np.array([[1.0, 0.3 * m], [0.0, 1.0]]))


def _shear_y(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    return _affine(p, np.array([[1.0, 0.0], [0.3 * m, 1.0]]))


def _flip(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    """Horizontal or vertical flip (50/50); no magnitude."""
    h, w = p.img.shape
    cx, cy, bw, bh = p.box
    if rng.random() < 0.5:
        return Patch(np.ascontiguousarray(p.img[:, ::-1]), np.ascontiguousarray(p.valid[:, ::-1]), (w - cx, cy, bw, bh))
    return Patch(np.ascontiguousarray(p.img[::-1]), np.ascontiguousarray(p.valid[::-1]), (cx, h - cy, bw, bh))


def _scale(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    h, w = p.img.shape
    cx, cy, bw, bh = p.box
    factor = max(1.0 + 0.5 * m, min(1.0, MIN_BOX_PX / max(min(bw, bh), 1e-6)))
    new_w, new_h = max(1, round(w * factor)), max(1, round(h * factor))
    sx, sy = new_w / w, new_h / h
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    img = cv2.resize(p.img, (new_w, new_h), interpolation=interp)
    valid = cv2.resize(p.valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[~valid] = 0
    return Patch(img, valid, (cx * sx, cy * sy, bw * sx, bh * sy))


def _point_op(p: Patch, lut: np.ndarray) -> Patch:
    """y = f(x) as a lookup table over the 256 grey values (`lut` = f(RAMP), any float)."""
    img = cv2.LUT(p.img, to_uint8(lut))
    img[~p.valid] = 0
    return Patch(img, p.valid, p.box)


# PIL ImageEnhance / torchvision _blend: factor * img + (1 - factor) * degenerate
def _brightness(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    return _point_op(p, max(0.0, 1.0 + 0.9 * m) * RAMP)  # degenerate = 0


def _contrast(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    factor = max(0.0, 1.0 + 0.9 * m)
    mean = float(p.img[p.valid].mean())  # degenerate = mean grey value
    return _point_op(p, factor * RAMP + (1.0 - factor) * mean)


def _sharpness(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    if min(p.img.shape) <= 2:
        return p
    factor = max(0.0, 1.0 + 0.9 * m)
    kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.0
    degenerate = p.img.copy()  # border pixels stay unchanged, as in PIL/torchvision
    degenerate[1:-1, 1:-1] = cv2.filter2D(p.img, -1, kernel)[1:-1, 1:-1]  # uint8, like PIL's SMOOTH image
    img = cv2.addWeighted(p.img, factor, degenerate, 1.0 - factor, 0.0)  # rounded + saturated to uint8
    img[~p.valid] = 0
    return Patch(img, p.valid, p.box)


def _identity(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    return p


def _posterize(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    bits = int(np.clip(8 - round(4 * m), 1, 8))
    return Patch(p.img & np.uint8((0xFF << (8 - bits)) & 0xFF), p.valid, p.box)


def _solarize(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    threshold = max(0.0, 255.0 * (1.0 - m))
    img = np.where(p.img >= threshold, 255 - p.img, p.img)
    img[~p.valid] = 0
    return Patch(img, p.valid, p.box)


def _autocontrast(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    vals = p.img[p.valid]
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return p
    return _point_op(p, (RAMP - lo) * (255.0 / (hi - lo)))


def _equalize(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    """Histogram equalization over the valid pixels (same LUT as PIL / torchvision)."""
    hist = np.bincount(p.img[p.valid], minlength=256)
    step = hist[hist != 0][:-1].sum() // 255
    if step == 0:
        return p
    lut = (np.cumsum(hist) + step // 2) // step
    lut = np.clip(np.concatenate([[0], lut[:-1]]), 0, 255).astype(np.uint8)
    img = lut[p.img]
    img[~p.valid] = 0
    return Patch(img, p.valid, p.box)


# name -> (function(patch, m, rng), signed: magnitude gets a random sign)
OPS: dict[str, tuple[Callable[[Patch, float, np.random.Generator], Patch], bool]] = {
    "identity": (_identity, False),
    "color": (_identity, False),  # saturation does nothing on a single-channel IR patch
    "flip": (_flip, False),
    "rotate": (_rotate, True),
    "scale": (_scale, True),
    "shear_x": (_shear_x, True),
    "shear_y": (_shear_y, True),
    "brightness": (_brightness, True),
    "contrast": (_contrast, True),
    "sharpness": (_sharpness, True),
    "posterize": (_posterize, False),
    "solarize": (_solarize, False),
    "autocontrast": (_autocontrast, False),
    "equalize": (_equalize, False),
}


def rand_augment(p: Patch, ops: Sequence[str], num_ops: int, magnitude: float, rng: np.random.Generator) -> Patch:
    """RandAugment: `num_ops` ops drawn uniformly with replacement, each applied with `magnitude`."""
    if num_ops <= 0 or not ops:
        return p
    for i in rng.integers(0, len(ops), size=num_ops):
        fn, signed = OPS[ops[i]]
        m = magnitude / MAX_MAGNITUDE
        if signed and rng.random() < 0.5:
            m = -m
        p = fn(p, m, rng)
    return p


# ---------------------------------------------------------------------------
# Step 3: adaptive paste
# ---------------------------------------------------------------------------
def context_level(p: Patch) -> float:
    """Median grey value b of the patch around the target (valid pixels outside the box); 0 if there are none."""
    h, w = p.img.shape
    cx, cy, bw, bh = p.box
    inside = ((np.abs(np.arange(h) + 0.5 - cy) <= bh / 2)[:, None]
              & (np.abs(np.arange(w) + 0.5 - cx) <= bw / 2)[None, :])
    ctx = p.img[p.valid & ~inside]
    return float(np.median(ctx)) if ctx.size else 0.0


def paste_weights(p: Patch, fusion: str) -> np.ndarray | None:
    """Per-pixel weight T' of the target patch; None if nothing would be pasted.

    `paper`:      Eq. (6) as written, T / max(T).
    `background`: Eq. (6) relative to the background, max(T - b, 0) / max(T - b), b = context_level(p)
                  (not in the paper).
    `none`:       the patch is copied as is.
    """
    if fusion == "none":
        return p.valid.astype(np.float32)
    t = p.img.astype(np.float32)
    if fusion == "background":
        t = np.clip(t - context_level(p), 0.0, None)
    peak = float(t.max())  # invalid pixels are 0 (so also <= b): this is the max over the patch
    if peak <= 0:
        return None
    return t / peak


def fuse(region: np.ndarray, patch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Eq. (7): I' = T' * T + (1 - T') * I. uint8 region and patch, float32 weights -> uint8."""
    return to_uint8(weights * patch + (1.0 - weights) * region)


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def encoded_labels(transform: A.BasicTransform, data: dict[str, Any], class_label: float) -> list[float]:
    """`class_label` encoded for each label field, as Albumentations stores it in the bbox array.

    A frame without boxes reaches the transform without label columns, and Albumentations
    registers a label field only when it first sees a box. If this Compose has not seen one
    yet, the field is registered here, with the type of the caller's (empty) label input.
    """
    proc = transform.get_processor("bboxes")
    if proc is None:
        return []
    codes = []
    for field in proc.params.label_fields or []:
        like = data.get(field)  # still in `data` only when the frame has no boxes
        if isinstance(like, np.ndarray) and np.issubdtype(like.dtype, np.number):
            sample = np.full(1, class_label, dtype=like.dtype)
        else:
            sample = [class_label]
        codes.append(float(proc.label_manager.process_field("bboxes", field, sample)[0, 0]))
    return codes


# ---------------------------------------------------------------------------
# Albumentations transform
# ---------------------------------------------------------------------------
class ACB(A.DualTransform):
    """Auto-augmented copy-paste bounding-box (Tong et al. 2024). See the module docstring.

    Args:
        records: training FrameRecords; every record with a box below `max_area_ratio` is a possible
            source target.
        max_area_ratio: only boxes whose area / frame area is below this are copied (small targets,
            0.15 % as in PGDAA); None = every box.
        num_targets: number of targets pasted per image, int or [min, max] (uniform).
        num_ops: RandAugment N, operations per patch (0 = no random augment).
        magnitude: RandAugment M on the 0..10 scale, shared by all operations.
        ops: operation names, see OPS.
        context: context on each side of the box, as a fraction of the box width/height.
        min_context_px: minimum context on each side in pixels.
        fusion: "paper" (Eqs. 6-7 as written; default), "background" (Eq. 6 relative to the context
            median; not in the paper) or "none" (patch copied as is).
        max_tries: random positions tried per patch before it is skipped.
        class_label: label of the pasted boxes, in every label field (encoded like the input labels).
        fixed: True = a frame gets the same pastes in every epoch (the paper's "initially enriched"
            dataset): the transform's own RNG is seeded per call from the `frame_key` that AlbuStep
            passes ("<run seed>|<dataset>/<split>/<seq>/<frame>"). Calls without `frame_key`
            (notebooks, the config check) stay random. False = new pastes in every epoch.
    """

    _targets = (Targets.IMAGE, Targets.BBOXES)

    def __init__(
        self,
        records: Sequence[FrameRecord],
        max_area_ratio: float | None = 0.0015,
        num_targets: int | Sequence[int] = 1,
        num_ops: int = 2,
        magnitude: float = 5.0,
        ops: Sequence[str] = DEFAULT_OPS,
        context: float = 1.0,
        min_context_px: float = 8.0,
        fusion: str = "paper",
        max_tries: int = 20,
        class_label: float = 0,
        fixed: bool = True,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        lo, hi = (num_targets, num_targets) if isinstance(num_targets, int) else tuple(num_targets)
        if not 0 <= lo <= hi:
            raise ValueError(f"ACB: num_targets must be n or [min, max] with 0 <= min <= max, got {num_targets}")
        if not isinstance(fixed, bool):
            raise ValueError(f"ACB: fixed must be true or false, got {fixed!r}")
        unknown = [o for o in ops if o not in OPS]
        if unknown:
            raise ValueError(f"ACB: unknown ops {unknown}. Available: {sorted(OPS)}")
        if fusion not in FUSIONS:
            raise ValueError(f"ACB: fusion must be one of {FUSIONS}, got '{fusion}'")
        if num_ops < 0 or magnitude < 0 or context < 0 or min_context_px < 0 or max_tries < 1:
            raise ValueError("ACB: num_ops, magnitude, context, min_context_px must be >= 0 and max_tries >= 1")

        if max_area_ratio is not None and max_area_ratio <= 0:
            raise ValueError(f"ACB: max_area_ratio must be > 0 or None, got {max_area_ratio}")

        boxed = [r for r in records if r.bbox is not None]
        self.max_area_ratio = None if max_area_ratio is None else float(max_area_ratio)
        self.sources = boxed if self.max_area_ratio is None else [
            r for r in boxed if r.bbox[2] * r.bbox[3] / (r.img_w * r.img_h) < self.max_area_ratio]
        if not self.sources:
            raise ValueError("ACB: no training box" + (f" below max_area_ratio {self.max_area_ratio:g}"
                                                       if self.max_area_ratio is not None else "")
                             + ", nothing to copy.")
        self.num_targets = (int(lo), int(hi))
        self.num_ops = int(num_ops)
        self.magnitude = float(magnitude)
        self.ops = list(ops)
        self.context = float(context)
        self.min_context_px = float(min_context_px)
        self.fusion = fusion
        self.max_tries = int(max_tries)
        self.class_label = class_label
        self.fixed = fixed
        logger.info(f"ACB: {len(self.sources)} source targets of {len(boxed)} training boxes"
                    + (f" (area ratio < {self.max_area_ratio:g})" if self.max_area_ratio is not None else "")
                    + (", fixed per frame" if self.fixed else ", new pastes every epoch"))
        logger.debug(f"ACB: num_targets={self.num_targets}, N={self.num_ops}, M={self.magnitude}, "
                     f"ops={self.ops}, fusion={self.fusion}")

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Any:
        key = kwargs.get("frame_key")
        if self.fixed and key is not None:
            # Same frame -> same RNG state -> same decision (p), sources, ops and positions in every epoch.
            # The Compose gives every transform its own RNG objects, so the others stay random.
            self.set_random_seed(zlib.crc32(f"ACB|{key}".encode()))
        return super().__call__(*args, force_apply=force_apply, **kwargs)

    @property
    def targets_as_params(self) -> list[str]:
        return ["image", "bboxes"]  # without boxes the pasted targets would be unlabelled

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        # `records` is left out on purpose: repr/serialization would print every source frame.
        return ("max_area_ratio", "num_targets", "num_ops", "magnitude", "ops", "context", "min_context_px",
                "fusion", "max_tries", "class_label", "fixed")

    def _encoded_labels(self, data: dict[str, Any]) -> list[float]:
        return encoded_labels(self, data, self.class_label)

    def _make_patch(self) -> Patch | None:
        record = self.sources[self.py_random.randrange(len(self.sources))]
        patch = crop_target(record, self.context, self.min_context_px)
        if patch is None:
            return None
        return rand_augment(patch, self.ops, self.num_ops, self.magnitude, self.random_generator)

    def _find_position(self, ph: int, pw: int, H: int, W: int, occupied: list) -> tuple[int, int] | None:
        if ph > H or pw > W:
            return None
        for _ in range(self.max_tries):
            x, y = self.py_random.randint(0, W - pw), self.py_random.randint(0, H - ph)
            if not any(_overlaps((x, y, x + pw, y + ph), o) for o in occupied):
                return x, y
        return None

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        H, W = data["image"].shape[:2]
        boxes = np.asarray(data.get("bboxes", np.zeros((0, 4))), dtype=np.float64)  # normalized x_min, y_min, x_max, y_max
        occupied = [tuple(b[:4] * (W, H, W, H)) for b in boxes] if boxes.ndim == 2 else []
        pastes, new_boxes = [], []
        for _ in range(self.py_random.randint(*self.num_targets)):
            patch = self._make_patch()
            if patch is None:
                continue
            weights = paste_weights(patch, self.fusion)
            if weights is None:
                continue
            ph, pw = patch.img.shape
            pos = self._find_position(ph, pw, H, W, occupied)
            if pos is None:
                continue
            x, y = pos
            occupied.append((x, y, x + pw, y + ph))
            pastes.append((x, y, patch.img, weights))
            cx, cy, bw, bh = patch.box
            x0, y0 = np.clip([x + cx - bw / 2, y + cy - bh / 2], 0, [W, H])
            x1, y1 = np.clip([x + cx + bw / 2, y + cy + bh / 2], 0, [W, H])
            new_boxes.append([x0 / W, y0 / H, x1 / W, y1 / H])  # albumentations' internal format
        return {
            "pastes": pastes,
            "new_bboxes": np.asarray(new_boxes, dtype=np.float32).reshape(-1, 4),
            "new_label_codes": self._encoded_labels(data) if new_boxes else [],
        }

    def apply(self, img: np.ndarray, pastes: list, **params: Any) -> np.ndarray:
        check_gray_uint8(img)
        if not pastes:
            return img
        out = img.copy()
        for x, y, patch, weights in pastes:
            h, w = patch.shape
            out[y:y + h, x:x + w] = fuse(out[y:y + h, x:x + w], patch, weights)
        return out

    def apply_to_bboxes(self, bboxes: np.ndarray, new_bboxes: np.ndarray, new_label_codes: list[float],
                        **params: Any) -> np.ndarray:
        if len(new_bboxes) == 0:
            return bboxes
        # Rows: 4 coordinates, extra input columns (if any), then one encoded column per label field.
        # An empty input has no label columns yet, so its width says nothing about them.
        n_labels = len(new_label_codes)
        n_cols = bboxes.shape[1] if bboxes.size else 4 + n_labels
        added = np.zeros((len(new_bboxes), n_cols), dtype=np.float32)
        added[:, :4] = new_bboxes
        if n_labels:
            added[:, n_cols - n_labels:] = new_label_codes
        return np.concatenate([bboxes.reshape(-1, n_cols).astype(np.float32), added], axis=0)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError("ACB pastes box targets only; masks are not supported.")

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError("ACB pastes box targets only; keypoints are not supported.")


# ---------------------------------------------------------------------------
# PGDAA: target library (read) and Gaussian paste (Wang et al. 2022, "Gaussian paste")
# The library is built by data_handler/target_library.py.
# ---------------------------------------------------------------------------
CSV_FILE = "library.csv"


def read_cutout(path) -> tuple[np.ndarray, np.ndarray]:
    """(gray uint8, mask bool) of a library PNG."""
    with Image.open(path) as im:
        la = np.asarray(im.convert("LA"))
    return la[..., 0].copy(), la[..., 1] > 127


def load_library(lib_dir: Path = LIBRARY_DIR) -> pd.DataFrame:
    """The library table with an absolute `path` column per cutout."""
    lib_dir = Path(lib_dir)
    csv_path = lib_dir / CSV_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Build the library first (notebook 05, build_library).")
    df = pd.read_csv(csv_path)
    df["path"] = [str(lib_dir / f) for f in df.file]
    return df


def gaussian_weights(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Weight map of the target pixels: Gaussian-blurred mask, scaled to max 1, 0 outside the mask.

    Restricting it to the mask matters: the cutout is 0 outside the mask, so weights there would
    blend black into the background. sigma = 0 gives the hard mask.
    """
    m = mask.astype(np.float32)
    if sigma <= 0 or not mask.any():
        return m
    w = cv2.GaussianBlur(m, (0, 0), sigma) * m
    return w / w.max()


def gaussian_paste(image: np.ndarray, gray: np.ndarray, mask: np.ndarray, x: int, y: int,
                   sigma: float = 1.0, box=None) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Paste a cutout with its top-left corner at (x, y): I' = w * T + (1 - w) * I.

    `box` (x, y, w, h) in cutout coordinates is the label to carry over (default: tight mask box).
    Returns the new image and the pasted box in image coordinates.
    """
    h, w = gray.shape
    H, W = image.shape[:2]
    if not (0 <= x and 0 <= y and x + w <= W and y + h <= H):
        raise ValueError(f"Cutout {w}x{h} at ({x}, {y}) does not fit into {W}x{H}.")
    if box is None:
        ys, xs = np.nonzero(mask)
        box = (xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())
    out = image.copy()
    out[y:y + h, x:x + w] = fuse(out[y:y + h, x:x + w], gray, gaussian_weights(mask, sigma))
    bx, by, bw, bh = (float(v) for v in box)
    return out, (x + bx, y + by, bw, bh)


# ---------------------------------------------------------------------------
# PGDAA: PGDA adapted (Wang et al. 2022), see the module docstring
# ---------------------------------------------------------------------------
# Shape transforms of the target patch (image + mask + box). The paper names the operations,
# not their strength; these defaults are ours (moderate, for patches of ~10-100 px).
DEFAULT_TARGET_OPS: tuple[dict, ...] = (
    # gt(): geometric transforms (paper: horizontal/vertical flip, mirror, affine)
    {"name": "HorizontalFlip", "p": 0.5},
    {"name": "VerticalFlip", "p": 0.5},
    {"name": "Affine", "p": 0.5, "params": {"rotate": [-30, 30], "fit_output": True}},
    # nt(): non-rigid transforms (paper: elastic [43], grid distortion [44], piecewise affine [45])
    {"name": "ElasticTransform", "p": 0.5, "params": {"alpha": 10, "sigma": 3}},
    {"name": "GridDistortion", "p": 0.5, "params": {"num_steps": 3, "distort_limit": [-0.2, 0.2]}},
    {"name": "PiecewiseAffine", "p": 0.5, "params": {"scale": [0.02, 0.04], "nb_rows": 3, "nb_cols": 3}},
)
TARGET_OP_NAMES = ("HorizontalFlip", "VerticalFlip", "Affine", "ElasticTransform", "GridDistortion", "PiecewiseAffine")


@dataclass
class LibTarget:
    """One library target in memory: gray (0 outside the mask), mask, annotated box (x0, y0, x1, y1)."""

    gray: np.ndarray
    mask: np.ndarray
    box: tuple[float, float, float, float]


class PGDAA(A.DualTransform):
    """PGDA adapted (Wang et al. 2022, Sec. III, Algorithm 1). See the module docstring.

    With probability p the training sample is REPLACED by a synthetic image: a pure-background
    subfigure (B_O) with `num_targets` pasted library targets; its boxes are only the pasted ones.

    Args:
        records: training FrameRecords. Frames without a box are the pure backgrounds B_O; the boxes
            of the others give the scale prior A.
        library_dir: target library (built with `python src/main.py -l`), the set T and the prior S.
        num_targets: N_t, targets per synthetic image, int or [min, max] (uniform).
        subfigure_split: crop a random multiscale subfigure of the background and scale every target to
            an area ratio drawn from A (paper's subfigure-split module). False: full background frame,
            targets keep their library size (ablation, Table V).
        crop_scale: [min, max] side length of the subfigure as a fraction of the frame (uniform).
        max_area_ratio: A keeps only box area ratios below this (paper: 0.15 %); None = all.
        area_jitter: the drawn area ratio is multiplied by U(1 - j, 1 + j) (paper: +-10 %).
        scr_perception: Algorithm 1, lines 8-16 (False = ablation, Table V).
        scr_percentiles: S = library SCR values between these percentiles (paper: 20 %-80 %).
        d: radius of the SCR neighbourhood R around the paste position (paper: 20).
        sigma_floor: lower bound for std(R) (flat backgrounds; same as the library build).
        gaussian_sigma: sigma of the Gaussian paste in px (0 = hard mask, plain copy-paste).
        target_ops: shape transforms of the patch, entries {name, p, params}, names in TARGET_OP_NAMES.
        color_jitter: {brightness, contrast} factors of the ColorJitter after the SCR shift.
        max_tries: random positions tried per target before it is skipped.
        class_label: label of the pasted boxes.
    """

    _targets = (Targets.IMAGE, Targets.BBOXES)

    def __init__(
        self,
        records: Sequence[FrameRecord],
        library_dir: str | Path = LIBRARY_DIR,
        num_targets: int | Sequence[int] = 1,
        subfigure_split: bool = True,
        crop_scale: Sequence[float] = (0.5, 1.0),
        max_area_ratio: float | None = 0.0015,
        area_jitter: float = 0.1,
        scr_perception: bool = True,
        scr_percentiles: Sequence[float] = (20, 80),
        d: int = 20,
        sigma_floor: float = 1.0,
        gaussian_sigma: float = 1.0,
        target_ops: Sequence[dict] = DEFAULT_TARGET_OPS,
        color_jitter: dict | None = None,
        max_tries: int = 20,
        class_label: float = 0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        lo, hi = (num_targets, num_targets) if isinstance(num_targets, int) else tuple(num_targets)
        if not 0 <= lo <= hi:
            raise ValueError(f"PGDAA: num_targets must be n or [min, max] with 0 <= min <= max, got {num_targets}")
        c_lo, c_hi = (float(v) for v in crop_scale)
        if not 0.0 < c_lo <= c_hi <= 1.0:
            raise ValueError(f"PGDAA: crop_scale must be [min, max] with 0 < min <= max <= 1, got {crop_scale}")
        q_lo, q_hi = (float(v) for v in scr_percentiles)
        if not 0.0 <= q_lo < q_hi <= 100.0:
            raise ValueError(f"PGDAA: scr_percentiles must be [lo, hi] with 0 <= lo < hi <= 100, got {scr_percentiles}")
        if not 0.0 <= area_jitter < 1.0 or d < 1 or gaussian_sigma < 0 or max_tries < 1:
            raise ValueError("PGDAA: need 0 <= area_jitter < 1, d >= 1, gaussian_sigma >= 0, max_tries >= 1")
        jitter = {"brightness": 0.1, "contrast": 0.1, **(color_jitter or {})}
        unknown = set(jitter) - {"brightness", "contrast"}
        if unknown:
            raise ValueError(f"PGDAA: color_jitter only has brightness/contrast (grayscale), got {sorted(unknown)}")

        self.library_dir = str(library_dir)
        self.num_targets = (int(lo), int(hi))
        self.subfigure_split = bool(subfigure_split)
        self.crop_scale = (c_lo, c_hi)
        self.max_area_ratio = None if max_area_ratio is None else float(max_area_ratio)
        self.area_jitter = float(area_jitter)
        self.scr_perception = bool(scr_perception)
        self.scr_percentiles = (q_lo, q_hi)
        self.d = int(d)
        self.sigma_floor = float(sigma_floor)
        self.gaussian_sigma = float(gaussian_sigma)
        self.target_ops = [dict(e) for e in target_ops]
        self.color_jitter = {k: float(v) for k, v in jitter.items()}
        self.max_tries = int(max_tries)
        self.class_label = class_label

        # --- T and S from the library ---------------------------------------------------------
        try:
            lib = load_library(library_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"PGDAA needs the target library: {e}") from None
        scr = lib.scr.to_numpy(dtype=np.float64)
        s_lo, s_hi = np.percentile(scr, self.scr_percentiles)
        self.S = scr[(scr >= s_lo) & (scr <= s_hi)].astype(np.float32)
        bright = lib[lib.polarity == "bright"]  # Algorithm 1 makes every target warmer than its background
        self.lib_targets: list[LibTarget] = []
        for row in bright.itertuples(index=False):
            gray, mask = read_cutout(row.path)
            self.lib_targets.append(LibTarget(gray, mask, (float(row.box_x), float(row.box_y),
                                                       float(row.box_x + row.box_w), float(row.box_y + row.box_h))))
        if not self.lib_targets:
            raise ValueError(f"PGDAA: no bright target in {library_dir}.")

        # --- A and B_O from the training records -------------------------------------------------
        ratios = np.array([r.bbox[2] * r.bbox[3] / (r.img_w * r.img_h) for r in records if r.bbox is not None],
                          dtype=np.float64)
        ratios = ratios[ratios > 0]
        if self.max_area_ratio is not None:
            ratios = ratios[ratios < self.max_area_ratio]
        if self.subfigure_split and ratios.size == 0:
            raise ValueError("PGDAA: no training box below max_area_ratio, the scale prior A is empty.")
        self.A = ratios.astype(np.float32)
        self.backgrounds = [r for r in records if r.bbox is None]
        if not self.backgrounds:
            raise ValueError("PGDAA: no training frame without a target, the pure backgrounds B_O are empty.")

        # --- shape transforms of the patch (gt, nt) -----------------------------------------------
        ops = []
        for e in self.target_ops:
            if e.get("name") not in TARGET_OP_NAMES:
                raise ValueError(f"PGDAA: unknown target op {e.get('name')!r}. Available: {TARGET_OP_NAMES}")
            ops.append(getattr(A, e["name"])(p=float(e.get("p", 0.5)), **dict(e.get("params") or {})))
        self._patch_aug = A.Compose(
            ops, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], clip=True, min_visibility=0.0)
        ) if ops else None

        logger.info(f"PGDAA: T = {len(self.lib_targets)} bright library targets (of {len(lib)}), "
                    f"S = {len(self.S)} SCR values in [{s_lo:.2f}, {s_hi:.2f}] (percentiles {self.scr_percentiles}), "
                    f"A = {len(self.A)} area ratios"
                    + (f" < {self.max_area_ratio:g}" if self.max_area_ratio is not None else "")
                    + f", B_O = {len(self.backgrounds)} pure-background frames")

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        # `records` is left out on purpose: repr/serialization would print every frame.
        return ("library_dir", "num_targets", "subfigure_split", "crop_scale", "max_area_ratio", "area_jitter",
                "scr_perception", "scr_percentiles", "d", "sigma_floor", "gaussian_sigma", "target_ops",
                "color_jitter", "max_tries", "class_label")

    # --- B_O and subfigure split ------------------------------------------------------------------
    def _background(self, H: int, W: int) -> np.ndarray:
        """Random pure-background frame; with the subfigure split a random multiscale crop of it.
        Resized ("increased resolution by interpolation") to the size of the training frame."""
        r = self.backgrounds[self.py_random.randrange(len(self.backgrounds))]
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"PGDAA: cannot read background {r.img_path}")
        if self.subfigure_split:
            s = self.py_random.uniform(*self.crop_scale)
            ih, iw = img.shape
            ch, cw = max(1, round(ih * s)), max(1, round(iw * s))
            y0, x0 = self.py_random.randint(0, ih - ch), self.py_random.randint(0, iw - cw)
            img = img[y0:y0 + ch, x0:x0 + cw]
        if img.shape != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return img

    # --- target: gt(), nt(), area ratio from A ----------------------------------------------------
    def _make_target(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray, list[float]] | None:
        t = self.lib_targets[self.py_random.randrange(len(self.lib_targets))]
        # The cutout is 0 outside the mask. Interpolation (rotate, elastic, resize) would pull that
        # black into the edge pixels of the target, which stay inside the mask (cold rim). So the
        # outside is first filled from the nearest target pixels; it is set to 0 again at the end.
        gray = cv2.inpaint(t.gray, (~t.mask).astype(np.uint8), 2, cv2.INPAINT_TELEA)
        mask, box = t.mask.astype(np.uint8), list(t.box)
        if self._patch_aug is not None:
            self._patch_aug.set_random_seed(int(self.random_generator.integers(2**31)))
            res = self._patch_aug(image=gray, mask=mask, bboxes=[box], labels=[0])
            gray, mask = res["image"], res["mask"]
            box = [float(v) for v in res["bboxes"][0][:4]] if len(res["bboxes"]) else None
        mask = mask > 0
        if not mask.any():
            return None
        if box is None:  # the box was lost in a transform: tight box of the mask
            ys, xs = np.nonzero(mask)
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        if self.subfigure_split:
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw <= 0 or bh <= 0:
                return None
            beta = float(self.A[self.random_generator.integers(len(self.A))])
            beta *= self.py_random.uniform(1.0 - self.area_jitter, 1.0 + self.area_jitter)
            f = max(math.sqrt(beta * W * H / (bw * bh)), MIN_BOX_PX / min(bw, bh))
            h, w = mask.shape
            new_w, new_h = max(1, round(w * f)), max(1, round(h * f))
            if new_w > W or new_h > H:
                return None
            interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR
            gray = cv2.resize(gray, (new_w, new_h), interpolation=interp)
            mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
            if not mask.any():
                return None
            sx, sy = new_w / w, new_h / h
            box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
        gray = np.where(mask, gray, 0).astype(np.uint8)
        return gray, mask, box

    # --- SCR perception (Algorithm 1, lines 8-16) --------------------------------------------------
    def _scr_perception(self, gray: np.ndarray, mask: np.ndarray, bg: np.ndarray, x: int, y: int) -> np.ndarray:
        h, w = gray.shape
        H, W = bg.shape
        region = bg[max(y - self.d, 0):min(y + h + self.d, H), max(x - self.d, 0):min(x + w + self.d, W)]
        mu_r = float(region.mean())
        std_r = max(float(region.std()), self.sigma_floor)
        avg_t = float(gray[mask].mean())
        scr = (avg_t - mu_r) / std_r  # signed: a target darker than R is outside S and gets adjusted
        if float(self.S.min()) < scr < float(self.S.max()):
            return gray
        s_i = float(self.S[self.random_generator.integers(len(self.S))])
        diff = s_i * std_r - avg_t + mu_r
        t = gray.astype(np.float32) + diff
        # ColorJitter (grayscale: brightness, then contrast around the target mean)
        b = self.py_random.uniform(1.0 - self.color_jitter["brightness"], 1.0 + self.color_jitter["brightness"])
        c = self.py_random.uniform(1.0 - self.color_jitter["contrast"], 1.0 + self.color_jitter["contrast"])
        t *= b
        m = float(t[mask].mean())
        t = m + (t - m) * c
        out = to_uint8(t)  # clipped at 255: targets above it saturate
        out[~mask] = 0
        return out

    def _find_position(self, h: int, w: int, H: int, W: int, occupied: list) -> tuple[int, int] | None:
        if h > H or w > W:
            return None
        for _ in range(self.max_tries):
            x, y = self.py_random.randint(0, W - w), self.py_random.randint(0, H - h)
            if not any(_overlaps((x, y, x + w, y + h), o) for o in occupied):
                return x, y
        return None

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        H, W = data["image"].shape[:2]
        bg = self._background(H, W)  # B_o
        out = bg.copy()
        occupied, new_boxes = [], []
        for _ in range(self.py_random.randint(*self.num_targets)):
            target = self._make_target(H, W)
            if target is None:
                continue
            gray, mask, box = target
            h, w = gray.shape
            pos = self._find_position(h, w, H, W, occupied)
            if pos is None:
                continue
            x, y = pos
            weights = gaussian_weights(mask, self.gaussian_sigma)  # gp(): Gaussian paste
            if self.scr_perception:
                gray = self._scr_perception(gray, mask, bg, x, y)  # R from the background B_o
            out[y:y + h, x:x + w] = fuse(out[y:y + h, x:x + w], gray, weights)
            occupied.append((x, y, x + w, y + h))
            x0, y0 = np.clip([x + box[0], y + box[1]], 0, [W, H])
            x1, y1 = np.clip([x + box[2], y + box[3]], 0, [W, H])
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                new_boxes.append([x0 / W, y0 / H, x1 / W, y1 / H])  # albumentations' internal format
        return {
            "synthetic": out,
            "new_bboxes": np.asarray(new_boxes, dtype=np.float32).reshape(-1, 4),
            "new_label_codes": encoded_labels(self, data, self.class_label) if new_boxes else [],
        }

    def apply(self, img: np.ndarray, synthetic: np.ndarray, **params: Any) -> np.ndarray:
        check_gray_uint8(img)
        return synthetic

    def apply_to_bboxes(self, bboxes: np.ndarray, new_bboxes: np.ndarray, new_label_codes: list[float],
                        **params: Any) -> np.ndarray:
        # The sample is replaced: only the pasted boxes remain. Same column layout as the input.
        n_labels = len(new_label_codes)
        n_cols = bboxes.shape[1] if bboxes.size else 4 + n_labels
        out = np.zeros((len(new_bboxes), n_cols), dtype=np.float32)
        if len(new_bboxes):
            out[:, :4] = new_bboxes
            if n_labels:
                out[:, n_cols - n_labels:] = new_label_codes
        return out

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError("PGDAA pastes box targets only; masks are not supported.")

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError("PGDAA pastes box targets only; keypoints are not supported.")


# ---------------------------------------------------------------------------
# Registry of the `sota:` section
# ---------------------------------------------------------------------------
# Name in the `sota:` section -> factory(p=..., **params) -> transform
TRANSFORMS: dict[str, Callable[..., A.BasicTransform]] = {
    "ACB": ACB,
    "PGDAA": PGDAA,
}
# These factories get the training FrameRecords as first argument (ACB: copy sources;
# PGDAA: pure backgrounds B_O and the scale prior A).
NEEDS_TRAIN_RECORDS = {"ACB", "PGDAA"}
