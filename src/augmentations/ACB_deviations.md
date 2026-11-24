# ACB: deviations from the paper and how they are handled

**Paper:** Y. Tong, Y. Leng, H. Yang, Z. Wang, "Target-Focused Enhancement Network for Distant Infrared Dim
and Small Target Detection", *IEEE TGRS* 62 (2024), Art. 4109711, DOI 10.1109/TGRS.2024.3459652.
- ACB: Sec. III-D, Fig. 4, Eqs. 6–7.
- Ablation: Tables V/VI, Fig. 6.
- Setup: Sec. IV-A/B.

**Implementation:** `ACB` in `src/augmentations/sota.py`; config `augmentation.sota` in `configs/template.yaml`.

ACB (auto-augmented copy-paste bounding-box) is implemented as close to the paper as our task and data
allow. This file lists every point where it differs, why, and what was done about it.

## 1. ACB in short

ACB increases the number and variety of small targets in the training data. For each pasted target:

1. **Bounding-box copy.** A target of the training set is cropped **together with its surroundings**
   (a box, not a mask): "the target tends to have radiometric effects on the surrounding background, which
   are crucial for small target detection".
2. **Random augment.** The patch is transformed by randomly chosen operations (Fig. 4: rotate, flip,
   color, scale, brightness, sharpness, "…"), based on RandAugment (Cubuk et al. 2020, the paper's ref. [27]).
3. **Adaptive paste.** The patch is blended into a background patch I of the same size, with weights that
   follow the target's grey values (motivated by the Gaussian grey-value profile of small targets):

   T'(x, y) = T(x, y) / max T   (Eq. 6)      I' = T' ⊙ T + (1 − T') ⊙ I   (Eq. 7)

**Effect in the paper (Tables V/VI).** The paper reports the share of target pixels ("target
propagation"):
- NUDT-SIRST: 0.6752 ‰ → 1.3648 ‰ (×2.02)
- SIATD: 0.1842 ‰ → 0.3790 ‰ (×2.06)

| Method | Copy area | Random augment | Adaptive fusion | NUDT-SIRST mIoU / F1 | SIATD mIoU / F1 |
|---|---|---|---|---|---|
| no aug. | – | – | – | 0.9139 / 0.9548 | 0.9247 / 0.9604 |
| general aug. (transpose, crop, scale change, Gaussian blur) | – | ✓ | – | 0.9157 / 0.9560 | 0.9166 / 0.9569 |
| Copy-Paste (Ghiasi et al.) | mask | – | – | 0.9190 / 0.9577 | 0.9254 / 0.9611 |
| – | mask | ✓ | ✓ | 0.9118 / 0.9536 | 0.9215 / 0.9582 |
| CutMix | box | – | – | 0.9103 / 0.9535 | 0.9268 / 0.9620 |
| **ACB** | box | ✓ | ✓ | **0.9280 / 0.9627** | **0.9346 / 0.9662** |

The paper's explanations:
- the box beats the mask because accurate masks of IR small targets are hard to annotate and the
  surroundings carry information;
- general transforms disturb the compact target shapes;
- random duplication and pasting (CutMix, Copy-Paste) "can disrupt the overall semantic integrity of the
  original image";
- plain duplication overfits because the targets lack texture.

## 2. Deviations forced by our task, data or setup

| # | Paper | Our task / data | Handling in ACB | Consequence |
|---|---|---|---|---|
| 1 | **Segmentation**: DNANet baseline, SoftIoU loss. Fig. 6 shows the pasted patch with its **mask label**. | **Detection** with YOLO, boxes only | Each paste gets a **box** (YOLO label). After rotate/shear the box is the bounding box of the transformed ellipse inscribed in the original box. | Same method, different label type. The ellipse box stays tight for compact targets; the bounding box of the rotated corners would be 37 % wider at 30°. |
| 2 | Source targets: the training set (with masks) | Boxes, two datasets | Sources = **all training boxes of the run** (Anti-UAV410 + CST, after the configured frame sampling; in a dry run the sampled train split, not the 32-frame dry-run subset), clipped to the frame | No mask is needed: ACB copies boxes, as in the paper. Faint and occluded targets are included, as in "the training set". |
| 3 | SIATD scenes are 640×512 like ours (Sec. I, Fig. 1a), but the network input is fixed to 256×256 (Sec. IV-B; NUDT-SIRST is 256×256). Targets occupy 3–5 pixels **in area** (0.0018 % of the image, Sec. I)¹ | 640×512 frames at input size 640; box area ratio about 0.01–0.6 % (20th–80th percentile) | Context is **relative to the box** (`context`, at least `min_context_px`), so it scales with the target | Larger patches for larger targets; the method itself is unchanged. |
| 4 | "Datasets are initially enriched using an ACB strategy" (Fig. 2); offline or online is not stated explicitly | Online Albumentations pipeline | Online transform with probability `p` (default 1: every training image gets `num_targets` pastes; new pastes every epoch) | About the same target share: ×2.10 measured on Anti-UAV410 alone with the defaults (×2.02 / ×2.06 in the paper); the sources now also include CST. More variety than a fixed enriched set. |
| 5 | "Color" among the operations | Single-channel IR | `color` (saturation) is the identity on grayscale; it is still selectable but **not in the default** ops | No effect lost; a "color" draw would only waste one of the N operations. |
| 6 | Not stated in the paper. The RandAugment reference implementation (torchvision, `_blend`) casts to uint8 and truncates | uint8 conventions of the project | Point operations as lookup tables, filters in OpenCV on uint8, Eq. 7 in float once, results **rounded** | Avoids the systematic darkening by ~0.5 grey values from truncation. |

¹ The paper is inconsistent here: Table VI gives a target share of 0.1842 ‰ = 0.018 % for SIATD, ten times
the 0.0018 % of Sec. I.

## 3. Points the paper leaves open, and our choices

All of them are config parameters (`params` of the `ACB` entry).

| Open point | Choice (default) | Reason |
|---|---|---|
| **N, M and the "auto" part.** Fig. 4: ACB "automatically learns the optimal strategy … combining copy-paste with the autorandom augment approach". N, M and the search are not given. (Sec. IV-F calls ACB "parameter-free", but only in the sense of no trainable network parameters, in contrast to DLDC; it says nothing about N and M.) RandAugment's grid search appears only in Related Work (Sec. II-A). | `num_ops: 2`, `magnitude: 5`; "auto" = RandAugment's **grid search over (N, M)**, one run config per grid point | RandAugment is the method the paper builds on, and the paper names no other mechanism. M = 5 instead of RandAugment's ImageNet optimum M = 9, because the brightness factors 0.19 / 1.81 at M = 9 make many small targets vanish or saturate. |
| Size of the context | `context: 0.5` of the box side on each side, at least `min_context_px: 4` | Not given; Fig. 4 shows crops with clearly visible surroundings. |
| Pastes per image | `num_targets: 1` (n or [min, max]) | Reproduces the paper's doubling of the target share (deviation 4). |
| The "…" in the operation list | Default `ops`: identity, rotate, flip, scale, brightness, sharpness (the named ones + RandAugment's identity, without color). Also selectable: contrast, color, shear_x, shear_y, posterize, solarize, autocontrast, equalize. | Translate is not offered, because the paste position is random anyway. |
| Strength of each operation | torchvision RandAugment ranges at M = 10: rotate 30°, shear 0.3, enhance factors 1 ± 0.9, posterize 8 → 4 bits, solarize 255 → 0. Scale (not in RandAugment) 1 ± 0.5. Signed ops get a random sign; ops are drawn with replacement. | As in the reference implementation of RandAugment. |
| "Random backgrounds" | Random position in the **current training image** | Fig. 4 pastes into a training image, and the paper wants to "preserve the overall semantic integrity of the original images". |
| Geometric ops on the patch | The canvas grows, so nothing is cut off. The uncovered canvas is 0, so Eq. 6 gives it weight 0 and the background stays visible there. | Fig. 4 shows rotated square patches; this avoids visible patch corners. |
| Overlap | A paste never overlaps an existing box or another paste (`max_tries: 20` random positions, otherwise skipped) | Otherwise a labelled target could be painted over and its label would be wrong. |

## 4. Interpretations and implementation details

- Flip = horizontal or vertical with 50 % each. The paper only says "flip".
- Scale never shrinks a box below 2 px.
- Sharpness leaves the border pixels unchanged and equalize uses the same lookup table as PIL / torchvision.
- A patch whose maximum is 0 is skipped, because Eq. 6 is undefined there.
- **Properties of Eq. 6 (paper formula, unchanged).** The weight follows the brightness.
  - A **cold (dark)** target gets a low weight and is mostly replaced by the background, while its bright
    surroundings are pasted. In the paper's data the targets are warmer than their background; the
    "dark targets" of Fig. 1(d) and Sec. IV-A are dim (low energy), not colder than their surroundings.
    In ours, cold targets are rare (about 1 % of the target library). Such pastes still get a box.
  - If the **surroundings are almost as bright as the target**, they get a weight close to 1
    (surroundings / peak), so the patch rectangle stays visible in the image. The paper's targets are
    bright points on darker sky, where this weight is low. In our data it occurs regularly (seen in the
    dry-run previews).

## 5. Ablations (paper Tables V/VI)

| Paper row | ACB config |
|---|---|
| CutMix (box, no random augment, no fusion) | `num_ops: 0, fusion: none` |
| ACB | defaults |
| (not in the paper) one component off | `fusion: none` or `num_ops: 0` alone |
| Copy-Paste with mask; mask + augment + fusion | not available: needs pixel masks. The PGDAA target library could provide pseudo masks (separate experiment). |

The paper's baseline for Tables V/VI is the network with the DLDC module ("no aug." = 0.9139 mIoU =
"+DLDC" in Table III). In our setup the baseline is the YOLO run with the same config and ACB switched off.
