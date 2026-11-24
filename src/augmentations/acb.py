"""ACB: auto-augmented copy-paste bounding-box augmentation.

Tong, Leng, Yang, Wang: "Target-Focused Enhancement Network for Distant Infrared Dim
and Small Target Detection", IEEE TGRS 62 (2024), 4109711, Sec. III-D, Fig. 4.

Per pasted target, the paper's three steps:

1. Bounding-box copy: crop a target of the *training set* together with its
   surrounding context (a box, not a segmentation mask, because in IR the target's
   radiation also affects the pixels around it).
2. Random augment: RandAugment (Cubuk et al. 2020, ref. [27] of the paper) on the
   patch: ``num_ops`` operations drawn uniformly with replacement from ``ops``, every
   drawn operation is applied, all with the single global magnitude ``magnitude`` on
   RandAugment's 0..10 scale. The "auto" part of the paper is the search over
   (``num_ops``, ``magnitude``), which RandAugment does by grid search: one run
   config per grid point.
3. Adaptive paste onto a random position of the current training image (the
   "random background"), with I the background patch of the same size as the
   augmented grayscale target patch T:

       T'(x, y) = T(x, y) / max_xy T(x, y)          Eq. (6)
       I'       = T' * T + (1 - T') * I             Eq. (7)

The paper leaves the following open. Each choice is configurable and documented:

* context around the box: ``context`` (fraction of the box side on each side,
  at least ``min_context_px``);
* number of pasted targets per image: ``num_targets`` (n or [min, max], uniform).
  Default 1 with p = 1: in Tables V/VI ACB doubles the share of target pixels
  (x2.02 / x2.06), and on Anti-UAV410 (one target per frame) one paste per frame
  gives x2.10 (measured with the defaults; three pastes give x3.24);
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
  defaults                      -> full ACB (box, random augment, adaptive fusion)
  ``fusion: none`` / ``num_ops: 0`` alone -> one component switched off (not in the paper)

Works on uint8 HxW or HxWxC images whose channels are identical (our IR frames) and on
any Albumentations bbox format; the new boxes get ``class_label`` in every label field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import albumentations as A
import cv2
import numpy as np
from albumentations.core.type_definitions import Targets

from data_handler.formats import FrameRecord
from log_setup import get_logger

logger = get_logger(__name__)

MAX_MAGNITUDE = 10.0  # RandAugment: "integer scale from 0 to 10"; larger values extrapolate
MIN_BOX_PX = 2.0  # Scale never shrinks a target below this size (in pixels)
FUSIONS = ("adaptive", "none")
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


def _blend(p: Patch, degenerate: np.ndarray | float, factor: float) -> Patch:
    """PIL ImageEnhance / torchvision _blend: factor * img + (1 - factor) * degenerate."""
    out = factor * p.img.astype(np.float32) + (1.0 - factor) * degenerate
    img = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    img[~p.valid] = 0
    return Patch(img, p.valid, p.box)


def _brightness(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    return _blend(p, 0.0, max(0.0, 1.0 + 0.9 * m))


def _contrast(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    return _blend(p, float(p.img[p.valid].mean()), max(0.0, 1.0 + 0.9 * m))


def _sharpness(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    if min(p.img.shape) <= 2:
        return p
    kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.0
    img = p.img.astype(np.float32)
    degenerate = img.copy()  # border pixels stay unchanged, as in PIL/torchvision
    degenerate[1:-1, 1:-1] = cv2.filter2D(img, -1, kernel)[1:-1, 1:-1]
    return _blend(p, degenerate, max(0.0, 1.0 + 0.9 * m))


def _identity(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    return p


def _posterize(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    bits = int(np.clip(8 - round(4 * m), 1, 8))
    return Patch(p.img & np.uint8((0xFF << (8 - bits)) & 0xFF), p.valid, p.box)


def _solarize(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    threshold = max(0.0, 255.0 * (1.0 - m))
    img = np.where(p.img >= threshold, 255 - p.img, p.img).astype(np.uint8)
    img[~p.valid] = 0
    return Patch(img, p.valid, p.box)


def _autocontrast(p: Patch, m: float, rng: np.random.Generator) -> Patch:
    vals = p.img[p.valid]
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return p
    img = np.clip((p.img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    img[~p.valid] = 0
    return Patch(img, p.valid, p.box)


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
def paste_weights(p: Patch, fusion: str) -> np.ndarray | None:
    """Per-pixel weight of the target patch. `adaptive` = Eq. (6), `none` = plain copy of the patch."""
    if fusion == "none":
        return p.valid.astype(np.float32)
    peak = float(p.img.max())  # invalid pixels are 0, so this is the max over the patch
    if peak <= 0:
        return None
    return p.img.astype(np.float32) / peak


def fuse(region: np.ndarray, patch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Eq. (7): I' = T' * T + (1 - T') * I, on every channel of `region` (uint8)."""
    t, w = patch.astype(np.float32), weights
    if region.ndim == 3:
        t, w = t[..., None], w[..., None]
    out = w * t + (1.0 - w) * region.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------------------
# Albumentations transform
# ---------------------------------------------------------------------------
class ACB(A.DualTransform):
    """Auto-augmented copy-paste bounding-box (Tong et al. 2024). See the module docstring.

    Args:
        records: training FrameRecords; every record with a box is a possible source target.
        num_targets: number of targets pasted per image, int or [min, max] (uniform).
        num_ops: RandAugment N, operations per patch (0 = no random augment).
        magnitude: RandAugment M on the 0..10 scale, shared by all operations.
        ops: operation names, see OPS.
        context: context on each side of the box, as a fraction of the box width/height.
        min_context_px: minimum context on each side in pixels.
        fusion: "adaptive" (Eqs. 6-7) or "none" (patch copied as is).
        max_tries: random positions tried per patch before it is skipped.
        class_label: label of the pasted boxes, in every label field (encoded like the input labels).
    """

    _targets = (Targets.IMAGE, Targets.BBOXES)

    def __init__(
        self,
        records: Sequence[FrameRecord],
        num_targets: int | Sequence[int] = 1,
        num_ops: int = 2,
        magnitude: float = 5.0,
        ops: Sequence[str] = DEFAULT_OPS,
        context: float = 0.5,
        min_context_px: float = 4.0,
        fusion: str = "adaptive",
        max_tries: int = 20,
        class_label: float = 0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        lo, hi = (num_targets, num_targets) if isinstance(num_targets, int) else tuple(num_targets)
        if not 0 <= lo <= hi:
            raise ValueError(f"ACB: num_targets must be n or [min, max] with 0 <= min <= max, got {num_targets}")
        unknown = [o for o in ops if o not in OPS]
        if unknown:
            raise ValueError(f"ACB: unknown ops {unknown}. Available: {sorted(OPS)}")
        if fusion not in FUSIONS:
            raise ValueError(f"ACB: fusion must be one of {FUSIONS}, got '{fusion}'")
        if num_ops < 0 or magnitude < 0 or context < 0 or min_context_px < 0 or max_tries < 1:
            raise ValueError("ACB: num_ops, magnitude, context, min_context_px must be >= 0 and max_tries >= 1")

        self.sources = [r for r in records if r.bbox is not None]
        if not self.sources:
            raise ValueError("ACB: no training record with a box, nothing to copy.")
        self.num_targets = (int(lo), int(hi))
        self.num_ops = int(num_ops)
        self.magnitude = float(magnitude)
        self.ops = list(ops)
        self.context = float(context)
        self.min_context_px = float(min_context_px)
        self.fusion = fusion
        self.max_tries = int(max_tries)
        self.class_label = class_label
        logger.debug(f"ACB: {len(self.sources)} source targets, num_targets={self.num_targets}, "
                     f"N={self.num_ops}, M={self.magnitude}, ops={self.ops}, fusion={self.fusion}")

    @property
    def targets_as_params(self) -> list[str]:
        return ["image", "bboxes"]  # without boxes the pasted targets would be unlabelled

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        # `records` is left out on purpose: repr/serialization would print every source frame.
        return ("num_targets", "num_ops", "magnitude", "ops", "context", "min_context_px", "fusion",
                "max_tries", "class_label")

    def _encoded_labels(self, data: dict[str, Any]) -> list[float]:
        """`class_label` encoded for each label field, as Albumentations stores it in the bbox array.

        A frame without boxes reaches the transform without label columns, and Albumentations
        registers a label field only when it first sees a box. If this Compose has not seen one
        yet, the field is registered here, with the type of the caller's (empty) label input.
        """
        proc = self.get_processor("bboxes")
        if proc is None:
            return []
        codes = []
        for field in proc.params.label_fields or []:
            like = data.get(field)  # still in `data` only when the frame has no boxes
            if isinstance(like, np.ndarray) and np.issubdtype(like.dtype, np.number):
                sample = np.full(1, self.class_label, dtype=like.dtype)
            else:
                sample = [self.class_label]
            codes.append(float(proc.label_manager.process_field("bboxes", field, sample)[0, 0]))
        return codes

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
        if img.dtype != np.uint8:
            raise TypeError(f"ACB expects uint8 images, got {img.dtype}")
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
