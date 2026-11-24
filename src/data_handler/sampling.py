"""Frame sampling registry.

A sampler reduces the full frame list of one split (all datasets combined) to the
frames actually used. Select per split in the run config, e.g.::

    sampling:
      train: {method: stride_fraction, stride: 5, fraction: 0.5}

All keys besides ``method`` are passed to the sampler as keyword arguments.
New strategies: write a function ``(records, rng, **params) -> records`` and
decorate it with ``@register_sampler("<name>")``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from data_handler.formats import FrameRecord
from log_setup import get_logger

logger = get_logger(__name__)

Sampler = Callable[..., list[FrameRecord]]
SAMPLERS: dict[str, Sampler] = {}


def register_sampler(name: str):
    def deco(fn: Sampler) -> Sampler:
        SAMPLERS[name] = fn
        return fn
    return deco


def _by_sequence(records: list[FrameRecord]) -> dict[tuple[str, str], list[FrameRecord]]:
    groups: dict[tuple[str, str], list[FrameRecord]] = defaultdict(list)
    for r in records:
        groups[(r.dataset, r.seq)].append(r)
    return groups


@register_sampler("all")
def sample_all(records: list[FrameRecord], rng: np.random.Generator) -> list[FrameRecord]:
    """Use every frame."""
    return list(records)


@register_sampler("stride_fraction")
def stride_fraction(records: list[FrameRecord], rng: np.random.Generator,
                    stride: int = 5, fraction: float = 1.0) -> list[FrameRecord]:
    """Keep every ``stride``-th frame of each sequence, then randomly keep ``fraction`` of those."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    strided = [r for group in _by_sequence(records).values() for r in group if r.frame_idx % stride == 0]
    logger.debug(f"stride_fraction: stride {stride} -> {len(strided)}/{len(records)} frames")
    if fraction >= 1.0:
        return strided
    k = max(1, round(len(strided) * fraction))
    idx = np.sort(rng.choice(len(strided), size=k, replace=False))
    logger.debug(f"stride_fraction: fraction {fraction} -> {k} frames")
    return [strided[i] for i in idx]


def apply_sampler(records: list[FrameRecord], sampling_cfg: dict, seed: int) -> list[FrameRecord]:
    params = dict(sampling_cfg)
    method = params.pop("method")
    if method not in SAMPLERS:
        raise KeyError(f"Unknown sampling method '{method}'. Available: {sorted(SAMPLERS)}")
    logger.debug(f"Sampler '{method}' with {params}, seed {seed}")
    return SAMPLERS[method](records, np.random.default_rng(seed), **params)
