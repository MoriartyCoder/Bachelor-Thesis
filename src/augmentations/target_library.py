"""Target library for PGDA: masked target cutouts from Anti-UAV410 (proof of concept).

Wang, Li, Huang, Wu, Jie, Tao: "Prior-Guided Data Augmentation for Infrared Small Target
Detection", IEEE JSTARS 15 (2022), Sec. III.

PGDA crops every training target *by its pixel mask* (the set T), pastes it with a
Gaussian-weighted mask ("Gaussian paste") and adjusts its intensity to the training SCR
distribution ("SCR perception"). Our datasets only have boxes, so this module builds the set T
differently:

1. Candidates: only frames whose target is clearly visible. Anti-UAV410 has per-frame tags in
   ``IR_label.json``; a frame is used only if none of ``exclude_tags`` is set (default: all of
   them). Frames near the HUD band or touching the image border are left out, and each sequence
   is thinned (every ``stride``-th clean frame, at most ``max_per_seq``) because neighbouring
   frames are near-duplicates.
2. Pseudo mask: inside the box, the target is separated from its local background by a contrast
   threshold. The background is the ring (box +- d) minus (box +- gap), with d = 20 as PGDA's SCR
   neighbourhood. The polarity (bright/dark target) comes from the stronger extreme in the box.
3. Cutout: the target pixels are kept and everything else is 0 (black). The mask is the alpha
   channel, so the background is also transparent ("LA" PNG).

Tags in ``IR_label.json`` (per frame; only 47 of the 200 train sequences have them, the older
``2019092*`` recordings have empty lists). The official README names the attributes; the key
mapping below is confirmed for these:
    OC = occlusion, FM = fast motion, SV = scale variation, DBC = dynamic background clutter,
    TS = size class 0..3 (measured edges on sqrt(w*h): <=10, <=30, <=50, >50 px).
Not documented: LR (always 0 in our copy), VE, IC. The README's thermal crossover is "TC",
which is not a key here. Measured (notebook 05): VE is set on every frame without a target, so it
is in effect out-of-view; IC frames have the lowest local contrast of all tags (median peak SCR ~3),
which fits thermal crossover. The default excludes every tag ("better safe than sorry").

Library layout (``config.LIBRARY_DIR``)::

    library.csv        one row per target (see LIBRARY_COLUMNS)
    build_info.json    LibraryConfig, filter funnel, segmentation results, timestamp
    targets/<dataset>__<seq>__<frame:06d>.png

Only the train split is used, because the library feeds a training augmentation.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from augmentations.acb import fuse
from config import LIBRARY_DIR, DatasetName
from data_handler.formats import FrameRecord
from data_handler.sources import NoDataError, load_dataset_split
from log_setup import get_logger

logger = get_logger(__name__)

TAG_KEYS = ("LR", "VE", "OC", "FM", "SV", "IC", "DBC")
SIZE_KEY = "TS"
SIZE_EDGES = (10.0, 30.0, 50.0)  # sqrt(w*h) in px; reproduces TS on all tagged train frames
SIZE_CLASSES = {0: "<=10 px", 1: "10-30 px", 2: "30-50 px", 3: ">50 px"}
TAG_NAMES = {
    "LR": "undokumentiert (immer 0)", "VE": "undokumentiert, gemessen: Ziel nicht sichtbar", "OC": "Okklusion",
    "FM": "schnelle Bewegung", "SV": "Skalenänderung", "IC": "undokumentiert, gemessen: geringer Kontrast",
    "DBC": "dynamischer Hintergrund-Clutter",
}

INFO_FILE = "build_info.json"
CSV_FILE = "library.csv"
TARGETS_DIR = "targets"

LIBRARY_COLUMNS = [
    "id", "file", "dataset", "split", "seq", "frame_idx", "src_img", "img_w", "img_h",
    "src_x", "src_y", "src_w", "src_h",          # annotated box in the source frame
    "cut_x", "cut_y", "cut_w", "cut_h",          # cutout position/size in the source frame
    "box_x", "box_y", "box_w", "box_h",          # annotated box in cutout coordinates
    "mask_x", "mask_y", "mask_w", "mask_h",      # tight mask box in cutout coordinates
    "ts", "polarity", "mu_b", "sigma_b", "mu_t", "peak", "scr", "scr_peak", "threshold",
    "mask_px", "fill", "area_ratio", "saturated", *TAG_KEYS,
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LibraryConfig:
    """Every knob of the library build; written to build_info.json."""

    dataset: DatasetName = DatasetName.ANTI_UAV_410
    split: str = "train"
    # candidate selection
    require_tags: bool = True               # False: untagged sequences too (no tag filter for them)
    exclude_tags: tuple[str, ...] = TAG_KEYS  # a frame with any of these set is dropped
    size_classes: tuple[int, ...] = (0, 1, 2, 3)  # TS classes kept
    hud_rows: int = 96                      # window (box +- d) must start below this row
    stride: int = 10                        # every n-th clean frame per sequence
    max_per_seq: int = 100                  # then at most this many, evenly spaced
    # local contrast (PGDA Eq. 2 neighbourhood)
    d: int = 20                             # ring radius around the box
    gap: int = 2                            # pixels around the box left out of the ring (halo)
    sigma_floor: float = 1.0                # lower bound for sigma_b (flat backgrounds)
    min_scr: float = 5.0                    # peak SCR below this -> rejected
    # pseudo mask
    k_sigma: float = 3.0                    # threshold >= k_sigma * sigma_b ...
    rel_thresh: float = 0.2                 # ... and >= rel_thresh * peak contrast (0.3 cut off thin arms)
    box_tol: int = 2                        # mask may exceed the box by this many px
    min_component_frac: float = 0.25        # keep blobs >= this fraction of the peak blob
    min_mask_px: int = 4
    fill_range: tuple[float, float] = (0.05, 0.9)  # mask px / box area
    pad: int = 2                            # border around the cutout

    def __post_init__(self) -> None:
        self.dataset = DatasetName(self.dataset)
        self.exclude_tags = tuple(self.exclude_tags)
        self.size_classes = tuple(int(c) for c in self.size_classes)
        self.fill_range = tuple(float(v) for v in self.fill_range)
        unknown = set(self.exclude_tags) - set(TAG_KEYS)
        if unknown:
            raise ValueError(f"Unknown tags {sorted(unknown)}. Known: {TAG_KEYS}")
        if not set(self.size_classes) <= set(SIZE_CLASSES):
            raise ValueError(f"size_classes must be a subset of {sorted(SIZE_CLASSES)}")
        if self.stride < 1 or self.max_per_seq < 1:
            raise ValueError("stride and max_per_seq must be >= 1")
        if not 0 <= self.gap < self.d:
            raise ValueError("need 0 <= gap < d")
        if not 0.0 <= self.rel_thresh < 1.0:
            raise ValueError("rel_thresh must be in [0, 1)")
        lo, hi = self.fill_range
        if not 0.0 <= lo < hi:
            raise ValueError("fill_range must be (lo, hi) with 0 <= lo < hi")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["dataset"] = self.dataset.value
        return out


# ---------------------------------------------------------------------------
# Tags and candidate selection
# ---------------------------------------------------------------------------
def size_class(w, h):
    """TS class from the box size (scalar or arrays), with the edges measured on Anti-UAV410."""
    return np.digitize(np.sqrt(np.asarray(w, dtype=float) * np.asarray(h, dtype=float)), SIZE_EDGES, right=True)


def load_frame_tags(split_dir: Path) -> dict[str, dict[str, list[int]] | None]:
    """Per sequence the tag lists (TAG_KEYS + TS), or None if the sequence is untagged.

    List index i belongs to frame_idx i (same order as the boxes).
    """
    split_dir = Path(split_dir)
    out: dict[str, dict[str, list[int]] | None] = {}
    keys = (*TAG_KEYS, SIZE_KEY)
    for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label_file = seq_dir / "IR_label.json"
        if not label_file.exists():
            continue
        with open(label_file, encoding="utf-8") as f:
            ann = json.load(f)
        n = len(ann.get("exist") or ann.get("gt_rect") or ann.get("gt") or [])
        tags = {k: list(ann.get(k) or []) for k in keys}
        if n and all(len(tags[k]) == n for k in keys):
            out[seq_dir.name] = tags
        else:
            if any(tags[k] for k in keys):
                logger.warning(f"{seq_dir.name}: tag lists incomplete ({ {k: len(v) for k, v in tags.items()} } "
                               f"for {n} frames), treated as untagged.")
            out[seq_dir.name] = None
    n_tagged = sum(v is not None for v in out.values())
    logger.debug(f"{split_dir}: {n_tagged}/{len(out)} sequences tagged")
    return out


def tag_table(tags: dict[str, dict[str, list[int]] | None]) -> pd.DataFrame:
    """One row per frame of every tagged sequence: seq, frame_idx, TAG_KEYS..., TS."""
    parts = [pd.DataFrame({"seq": seq, "frame_idx": np.arange(len(t[SIZE_KEY])), **t})
             for seq, t in tags.items() if t is not None]
    if not parts:
        return pd.DataFrame(columns=["seq", "frame_idx", *TAG_KEYS, SIZE_KEY])
    return pd.concat(parts, ignore_index=True)


@dataclass(slots=True)
class Candidate:
    record: FrameRecord
    ts: int
    tags: dict[str, int] | None  # None = untagged sequence


def _even_idx(n_total: int, n_pick: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_total - 1, min(n_pick, n_total)).round().astype(int))


def _clear_of_hud_and_border(r: FrameRecord, cfg: LibraryConfig) -> bool:
    x, y, w, h = r.bbox
    return x >= 1 and y >= 1 and x + w <= r.img_w - 1 and y + h <= r.img_h - 1 and y - cfg.d >= cfg.hud_rows


def select_candidates(records: list[FrameRecord], tags: dict[str, dict[str, list[int]] | None],
                      cfg: LibraryConfig) -> tuple[list[Candidate], dict[str, int]]:
    """Frames whose target goes into the library, and the frame count after each filter step."""
    funnel: dict[str, int] = {"frames": len(records)}
    cands = []
    for r in records:
        if r.bbox is None:
            continue
        t = tags.get(r.seq)
        if t is not None and r.frame_idx >= len(t[SIZE_KEY]):
            t = None
        frame_tags = None if t is None else {k: int(t[k][r.frame_idx]) for k in (*TAG_KEYS, SIZE_KEY)}
        cands.append((r, frame_tags))
    funnel["with_target"] = len(cands)

    if cfg.require_tags:
        cands = [(r, t) for r, t in cands if t is not None]
        funnel["tagged_sequence"] = len(cands)

    cands = [(r, t) for r, t in cands if t is None or not any(t[k] for k in cfg.exclude_tags)]
    funnel["no_excluded_tag"] = len(cands)

    with_ts = []
    for r, t in cands:
        ts = t[SIZE_KEY] if t is not None else int(size_class(r.bbox[2], r.bbox[3]))
        if ts in cfg.size_classes:
            with_ts.append(Candidate(r, ts, None if t is None else {k: t[k] for k in TAG_KEYS}))
    funnel["size_class"] = len(with_ts)

    clean = [c for c in with_ts if _clear_of_hud_and_border(c.record, cfg)]
    funnel["clear_of_hud_and_border"] = len(clean)

    by_seq: dict[str, list[Candidate]] = defaultdict(list)
    for c in clean:
        by_seq[c.record.seq].append(c)
    selected = []
    for seq in sorted(by_seq):
        group = sorted(by_seq[seq], key=lambda c: c.record.frame_idx)[::cfg.stride]
        selected.extend(group[i] for i in _even_idx(len(group), cfg.max_per_seq))
    funnel["thinned"] = len(selected)

    logger.info("Candidates: " + " -> ".join(f"{k} {v}" for k, v in funnel.items())
                + f" ({len(by_seq)} sequences)")
    return selected, funnel


# ---------------------------------------------------------------------------
# Local contrast and pseudo mask
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Contrast:
    mu_b: float       # mean of the background ring
    sigma_b: float    # std of the background ring (>= sigma_floor)
    polarity: int     # +1 bright target, -1 dark target
    scr_peak: float   # |extreme in box - mu_b| / sigma_b, extreme in the polarity direction
    n_bg: int         # ring pixels used


def _box_int(bbox, W: int, H: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = bbox
    x0, y0 = max(math.floor(x), 0), max(math.floor(y), 0)
    x1, y1 = min(math.ceil(x + w), W), min(math.ceil(y + h), H)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def local_contrast(img: np.ndarray, bbox, d: int = 20, gap: int = 2, sigma_floor: float = 1.0,
                   hud_rows: int = 0) -> Contrast | None:
    """Ring statistics around the box and the peak SCR (same as `scr_extreme` in notebooks/06_measure.ipynb).

    Ring = (box +- d) minus (box +- gap); rows above `hud_rows` are left out.
    None if the box is empty or fewer than 30 ring pixels remain.
    """
    H, W = img.shape[:2]
    b = _box_int(bbox, W, H)
    if b is None:
        return None
    
    x0, y0, x1, y1 = b
    wx0, wy0, wx1, wy1 = max(x0 - d, 0), max(y0 - d, 0), min(x1 + d, W), min(y1 + d, H)
    win = img[wy0:wy1, wx0:wx1].astype(np.float32)
    ring = np.ones(win.shape, bool)
    ring[max(y0 - gap, 0) - wy0:min(y1 + gap, H) - wy0, max(x0 - gap, 0) - wx0:min(x1 + gap, W) - wx0] = False

    if hud_rows > wy0:
        ring[:hud_rows - wy0, :] = False

    bg = win[ring]
    if bg.size < 30:
        return None
    
    mu_b, sigma_b = float(bg.mean()), max(float(bg.std()), sigma_floor)
    box = img[y0:y1, x0:x1].astype(np.float32)
    sp, st = (float(box.max()) - mu_b) / sigma_b, (mu_b - float(box.min())) / sigma_b
    pol = 1 if sp >= st else -1
    return Contrast(mu_b, sigma_b, pol, max(sp, st), int(bg.size))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    cv2.floodFill(padded, None, (0, 0), 1)  # everything reachable from outside becomes 1
    return mask | (padded[1:-1, 1:-1] == 0)


@dataclass
class TargetCut:
    """Result of `segment_target`. `ok` only if `reason == "ok"`; the other fields are filled as
    far as the computation got (for the notebook, rejected targets keep context + mask)."""

    reason: str
    contrast: Contrast | None = None
    context: np.ndarray | None = None        # uint8 window box +- d
    context_origin: tuple[int, int] = (0, 0)  # (x, y) of the window in the frame
    context_mask: np.ndarray | None = None   # bool mask in window coordinates
    threshold: float = float("nan")          # on the signed contrast pol * (I - mu_b)
    gray: np.ndarray | None = None           # uint8 cutout, 0 outside the mask
    mask: np.ndarray | None = None           # bool cutout mask
    origin: tuple[int, int] = (0, 0)         # (x, y) of the cutout in the frame
    box: tuple[float, float, float, float] | None = None       # annotated box, cutout coords
    mask_box: tuple[int, int, int, int] | None = None          # tight mask box, cutout coords
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


def segment_target(img: np.ndarray, bbox, cfg: LibraryConfig) -> TargetCut:
    """Pseudo mask of the target in `bbox` (x, y, w, h) of the grayscale uint8 HxW frame `img`."""
    H, W = img.shape
    b = _box_int(bbox, W, H)
    if b is None:
        return TargetCut("empty_box")
    x0, y0, x1, y1 = b
    d = cfg.d
    wx0, wy0, wx1, wy1 = max(x0 - d, 0), max(y0 - d, 0), min(x1 + d, W), min(y1 + d, H)
    win = img[wy0:wy1, wx0:wx1]
    cut = TargetCut("ok", context=win.copy(), context_origin=(wx0, wy0))

    c = local_contrast(img, bbox, d, cfg.gap, cfg.sigma_floor, cfg.hud_rows)
    if c is None:
        cut.reason = "no_background"
        return cut
    cut.contrast = c
    if c.scr_peak < cfg.min_scr:
        cut.reason = "low_scr"
        return cut

    # 1. Threshold on the signed contrast, only inside box +- box_tol
    s = c.polarity * (win.astype(np.float32) - c.mu_b)
    t = cfg.box_tol
    allowed = np.zeros(win.shape, bool)
    allowed[max(y0 - t, wy0) - wy0:min(y1 + t, wy1) - wy0, max(x0 - t, wx0) - wx0:min(x1 + t, wx1) - wx0] = True
    peak = c.scr_peak * c.sigma_b
    thr = max(cfg.k_sigma * c.sigma_b, cfg.rel_thresh * peak)
    cut.threshold = thr
    fg = (s > thr) & allowed

    # 2. Close small gaps, keep the blob with the peak pixel (+ large neighbours), fill holes
    closed = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & allowed
    n_lab, labels, cc_stats, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    if n_lab <= 1:
        cut.context_mask = closed
        cut.reason = "tiny_mask"
        return cut
    box_s = np.where(allowed, s, -np.inf)
    py, px = np.unravel_index(np.argmax(box_s), s.shape)
    main = labels[py, px]
    areas = cc_stats[:, cv2.CC_STAT_AREA]
    if main == 0:
        main = 1 + int(np.argmax(areas[1:]))
    keep = [lab for lab in range(1, n_lab) if lab == main or areas[lab] >= cfg.min_component_frac * areas[main]]
    mask = _fill_holes(np.isin(labels, keep))
    cut.context_mask = mask

    # 3. Quality checks
    mask_px = int(mask.sum())
    fill = mask_px / max(bbox[2] * bbox[3], 1e-6)
    cut.stats = {"mask_px": mask_px, "fill": fill}
    if mask_px < cfg.min_mask_px:
        cut.reason = "tiny_mask"
        return cut
    if fill < cfg.fill_range[0]:
        cut.reason = "fill_low"
        return cut
    if fill > cfg.fill_range[1]:
        cut.reason = "fill_high"
        return cut

    # 4. Cutout = union(box, mask box) + pad; 0 outside the mask
    ys, xs = np.nonzero(mask)
    mx0, my0, mx1, my1 = xs.min() + wx0, ys.min() + wy0, xs.max() + 1 + wx0, ys.max() + 1 + wy0
    cx0, cy0 = max(min(x0, mx0) - cfg.pad, 0), max(min(y0, my0) - cfg.pad, 0)
    cx1, cy1 = min(max(x1, mx1) + cfg.pad, W), min(max(y1, my1) + cfg.pad, H)
    full = np.zeros((H, W), bool)
    full[wy0:wy1, wx0:wx1] = mask
    cmask = full[cy0:cy1, cx0:cx1]
    raw = img[cy0:cy1, cx0:cx1]
    cut.gray = np.where(cmask, raw, 0).astype(np.uint8)
    cut.mask = cmask
    cut.origin = (cx0, cy0)
    x, y, w, h = bbox
    cut.box = (float(x - cx0), float(y - cy0), float(w), float(h))
    cut.mask_box = (int(mx0 - cx0), int(my0 - cy0), int(mx1 - mx0), int(my1 - my0))

    # 5. Statistics (PGDA Eq. 1 and Eq. 2 with the pseudo mask)
    vals = raw[cmask].astype(np.float32)
    mu_t = float(vals.mean())
    extreme = float(vals.max()) if c.polarity > 0 else float(vals.min())
    cut.stats.update({
        "mu_t": mu_t, "peak": extreme,
        "scr": abs(mu_t - c.mu_b) / c.sigma_b,
        "area_ratio": mask_px / (W * H),
        "saturated": bool(extreme >= 250 if c.polarity > 0 else extreme <= 5),
    })
    return cut


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def save_cutout(path: Path, gray: np.ndarray, mask: np.ndarray) -> None:
    """Gray + alpha PNG: gray is 0 and alpha is 0 outside the mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    la = np.dstack([np.where(mask, gray, 0), mask.astype(np.uint8) * 255]).astype(np.uint8)
    Image.fromarray(la).save(path)  # (H, W, 2) uint8 -> mode "LA"


def read_cutout(path) -> tuple[np.ndarray, np.ndarray]:
    """(gray uint8, mask bool) of a library PNG."""
    with Image.open(path) as im:
        la = np.asarray(im.convert("LA"))
    return la[..., 0].copy(), la[..., 1] > 127


def _row(cut: TargetCut, cand: Candidate, tid: str, rel: str, cfg: LibraryConfig) -> dict:
    r, c, st = cand.record, cut.contrast, cut.stats
    tags = cand.tags or {}
    return {
        "id": tid, "file": rel, "dataset": r.dataset.value, "split": cfg.split, "seq": r.seq,
        "frame_idx": r.frame_idx, "src_img": r.img_path, "img_w": r.img_w, "img_h": r.img_h,
        "src_x": r.bbox[0], "src_y": r.bbox[1], "src_w": r.bbox[2], "src_h": r.bbox[3],
        "cut_x": cut.origin[0], "cut_y": cut.origin[1], "cut_w": cut.gray.shape[1], "cut_h": cut.gray.shape[0],
        "box_x": cut.box[0], "box_y": cut.box[1], "box_w": cut.box[2], "box_h": cut.box[3],
        "mask_x": cut.mask_box[0], "mask_y": cut.mask_box[1], "mask_w": cut.mask_box[2], "mask_h": cut.mask_box[3],
        "ts": cand.ts, "polarity": "bright" if c.polarity > 0 else "dark",
        "mu_b": c.mu_b, "sigma_b": c.sigma_b, "mu_t": st["mu_t"], "peak": st["peak"],
        "scr": st["scr"], "scr_peak": c.scr_peak, "threshold": cut.threshold,
        "mask_px": st["mask_px"], "fill": st["fill"], "area_ratio": st["area_ratio"],
        "saturated": st["saturated"], **{k: tags.get(k, np.nan) for k in TAG_KEYS},
    }


def build_library(cfg: LibraryConfig | None = None, out_dir: Path = LIBRARY_DIR,
                  overwrite: bool = False) -> pd.DataFrame:
    """Select, segment and save the targets. Returns the library table (as `load_library`).

    An existing `out_dir` is only replaced with `overwrite=True`, and only if it is a library
    (contains build_info.json); otherwise nothing is deleted.
    """
    cfg = cfg or LibraryConfig()
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists. Use overwrite=True to rebuild it.")
        if any(out_dir.iterdir()) and not (out_dir / INFO_FILE).exists():
            raise FileExistsError(f"{out_dir} is not empty and has no {INFO_FILE}: not a target library, "
                                  "nothing deleted.")
        logger.info(f"Deleting existing library {out_dir}")
        shutil.rmtree(out_dir)
    start = time.perf_counter()

    records = load_dataset_split(cfg.dataset, cfg.split)
    if not records:
        raise NoDataError(f"No frames in {cfg.dataset}/{cfg.split}.")
    tags = load_frame_tags(cfg.dataset.root / cfg.split)
    candidates, funnel = select_candidates(records, tags, cfg)

    (out_dir / TARGETS_DIR).mkdir(parents=True)
    info = {"created": datetime.now().isoformat(timespec="seconds"), "status": "incomplete",
            "config": cfg.to_dict(), "funnel": funnel}
    (out_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")  # marks the folder

    rows, reasons = [], Counter()
    for i, cand in enumerate(candidates, 1):
        r = cand.record
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Cannot read {r.img_path}, skipped.")
            reasons["unreadable"] += 1
            continue
        cut = segment_target(img, r.bbox, cfg)
        reasons[cut.reason] += 1
        if cut.ok:
            tid = f"{r.dataset.value}__{r.seq}__{r.frame_idx:06d}"
            rel = f"{TARGETS_DIR}/{tid}.png"
            save_cutout(out_dir / rel, cut.gray, cut.mask)
            rows.append(_row(cut, cand, tid, rel, cfg))
        if i % 500 == 0:
            logger.info(f"  {i}/{len(candidates)} candidates, {len(rows)} saved")

    df = pd.DataFrame(rows, columns=LIBRARY_COLUMNS)
    df.to_csv(out_dir / CSV_FILE, index=False)
    funnel["saved"] = len(df)
    info.update({"status": "complete", "funnel": funnel, "segmentation": dict(reasons.most_common()),
                 "n_targets": len(df), "n_sequences": int(df.seq.nunique()),
                 "duration_s": round(time.perf_counter() - start, 1)})
    (out_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info(f"Library: {len(df)} targets from {df.seq.nunique()} sequences -> {out_dir} "
                f"({info['duration_s']} s). Segmentation: {dict(reasons.most_common())}")
    return load_library(out_dir)


def load_library(lib_dir: Path = LIBRARY_DIR) -> pd.DataFrame:
    """The library table with an absolute `path` column per cutout."""
    lib_dir = Path(lib_dir)
    csv_path = lib_dir / CSV_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Build the library first (notebook 05, build_library).")
    df = pd.read_csv(csv_path)
    df["path"] = [str(lib_dir / f) for f in df.file]
    return df


# ---------------------------------------------------------------------------
# Paste (PGDA "Gaussian paste")
# ---------------------------------------------------------------------------
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
