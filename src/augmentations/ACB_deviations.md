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

   Our default is Eq. 6 as written; on our data it leaves the patch rectangle partly visible (deviation 7).

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
| 2 | Source targets: the training set (with masks), all of them **small** (3–5 px in area). ACB is a method for dim and small targets; Sec. IV-D sets it apart from Copy-Paste, which "is applied to normal-sized targets" | Boxes, two datasets, targets of all sizes: 45.6 % of the training boxes are above 0.15 % of the frame (Anti-UAV410 67.5 %, CST 4.2 %); median size √(w·h) = 18 px | Sources = the **training boxes of the run below `max_area_ratio`** (default 0.0015 = 0.15 % of the frame, the small-target limit also used by PGDAA; `null` = all boxes). Run = Anti-UAV410 + CST after the configured frame sampling; in a dry run the sampled train split, not the 32-frame dry-run subset. Boxes are clipped to the frame | Only small targets are copied, as in the paper: 54.4 % of the boxes (Anti-UAV410 32.5 %, CST 95.8 %). Large targets already in a frame stay. No mask is needed: ACB copies boxes, as in the paper. Faint and occluded small targets are included, as in "the training set". The paper's own 3–5 px would leave almost no source. |
| 3 | SIATD scenes are 640×512 like ours (Sec. I, Fig. 1a), but the network input is fixed to 256×256 (Sec. IV-B; NUDT-SIRST is 256×256). Targets occupy 3–5 pixels **in area** (0.0018 % of the image, Sec. I)¹ | 640×512 frames at input size 640; box area ratio about 0.01–0.6 % (20th–80th percentile) | Context is **relative to the box** (`context`, at least `min_context_px`), so it scales with the target | Larger patches for larger targets; the method itself is unchanged. |
| 4 | "The … dataset is initially enriched using an ACB strategy" (Sec. III-A, Fig. 2), and Table V reports the target share of the enriched training set: **one fixed enriched dataset** | Online Albumentations pipeline | Online transform with probability `p` (default 1: every training image gets `num_targets` pastes). With `fixed: true` (default) ACB seeds its own RNG per frame from "<run seed>\|<dataset>/<split>/<seq>/<frame>" (passed by `AlbuStep`), so a frame gets the same pastes in every epoch = the fixed enriched set, without writing files. Another `run.seed` gives another enriched set. `fixed: false`: new pastes every epoch (more variety than the paper). The other augmentations stay random per epoch; a `one_of` group around ACB would still choose per epoch | Same structure as the paper: one fixed enriched training set. Target share with one paste per frame (500 random training frames): target count ×2.07; pixel share ×2.02 on CST (small targets, as in the paper; paper ×2.02 / ×2.06), ×1.13 on Anti-UAV410 and ×1.18 overall, because the large Anti-UAV410 targets stay and are not copied (deviation 2). |
| 5 | "Color" among the operations | Single-channel IR | `color` (saturation) is the identity on grayscale; it is still selectable but **not in the default** ops | No effect lost; a "color" draw would only waste one of the N operations. |
| 6 | Not stated in the paper. The RandAugment reference implementation (torchvision, `_blend`) casts to uint8 and truncates | uint8 conventions of the project | Point operations as lookup tables, filters in OpenCV on uint8, Eq. 7 in float once, results **rounded** | Avoids the systematic darkening by ~0.5 grey values from truncation. |
| 7 | Eq. 6, T' = T / max T on the raw grey values, motivated by "the target grayscale distribution adhering to a Gaussian distribution". The data is **semi-simulated** (Sec. IV-A: synthetic targets embedded in real backgrounds), and the crops have a dark context (Fig. 4: points on dark sky). The result is seamless: Sec. III-D wants "to seamlessly integrate the target patch", and Fig. 4 (step 3) and Fig. 6 show no patch edges | Real drones on brighter, often textured backgrounds. Eq. 6 gives the context of a small source target (< 0.15 %) a median weight of 0.44 (25th–75th percentile 0.31–0.60; 558 random small training targets, `context: 0.5`), so part of the source background is pasted along | **Eq. 6 unchanged** (`fusion: paper`, default). Variant, not in the paper: `fusion: background`, Eq. 6 relative to the context level, T' = max(T − b, 0) / max(T − b), b = median grey value of the patch outside the box (Eq. 7 unchanged) | With `paper` the patch rectangle stays partly visible in the image; this is a property of the paper's formula on our data. `background` equals Eq. 6 for a dark context (b ≈ 0) and gives the context weight ≈ 0, so the paste blends in as in the paper's figures (targets in front of clutter can still leave visible structure). |

¹ The paper is inconsistent here: Table VI gives a target share of 0.1842 ‰ = 0.018 % for SIATD, ten times
the 0.0018 % of Sec. I.

## 3. Points the paper leaves open, and our choices

All of them are config parameters (`params` of the `ACB` entry).

| Open point | Choice (default) | Reason |
|---|---|---|
| **N, M and the "auto" part.** Fig. 4: ACB "automatically learns the optimal strategy … combining copy-paste with the autorandom augment approach". N, M and the search are not given. (Sec. IV-F calls ACB "parameter-free", but only in the sense of no trainable network parameters, in contrast to DLDC; it says nothing about N and M.) RandAugment's grid search appears in Related Work (Sec. II-A: a strategy space of 10² instead of 10³², "can even use simple grid search"), and the abstract describes ACB as "mapping augment strategies to 2-D space". | `num_ops: 2`, `magnitude: 5`; "auto" = RandAugment's **grid search over (N, M)**, one run config per grid point | RandAugment is the method the paper builds on, its strategy space is exactly the 2-D (N, M) space of the abstract, and the paper names no other mechanism. M = 5 instead of RandAugment's ImageNet optimum M = 9, because the brightness factors 0.19 / 1.81 at M = 9 make many small targets vanish or saturate. |
| Size of the context | `context: 1.0` of the box side on each side (patch = 3× the box), at least `min_context_px: 8` | Not given in the text. The mask crops of Fig. 4 (step 1), measured at 300 dpi, show per side 0.56–0.89×, 0.79–1.88× and 1.6× the target size for the three extended targets and 4.4–5.3× for the point target. 1.0× lies in the middle of the extended ones; the 8 px minimum gives a 2–3 px point target about 3–4× its size. The figure is a schematic, so these are estimates. |
| Pastes per image | `num_targets: 1` (n or [min, max]) | The paper doubles the target share with pastes from the same (small) distribution. One paste doubles our target count and, for small targets, the pixel share (deviation 4). Doubling the overall pixel share would need more than 4 pastes per frame (×1.74 at 4), i.e. about 5× the targets. |
| The "…" in the operation list | Default `ops`: identity, rotate, flip, scale, brightness, sharpness (the named ones + RandAugment's identity, without color). Also selectable: contrast, color, shear_x, shear_y, posterize, solarize, autocontrast, equalize. | Translate is not offered, because the paste position is random anyway. |
| Strength of each operation | torchvision RandAugment ranges at M = 10: rotate 30°, shear 0.3, enhance factors 1 ± 0.9, posterize 8 → 4 bits, solarize 255 → 0. Scale (not in RandAugment) 1 ± 0.5. Signed ops get a random sign; ops are drawn with replacement. | As in the reference implementation of RandAugment. |
| "Random backgrounds" | Random position in the **current training image** | Fig. 4 pastes into a training image, and the paper wants to "preserve the overall semantic integrity of the original images". |
| Geometric ops on the patch | The canvas grows, so nothing is cut off. The uncovered canvas is 0, so Eq. 6 gives it weight 0 and the background stays visible there. | Fig. 4 shows rotated square patches; this avoids visible patch corners. |
| Overlap | A paste never overlaps an existing box or another paste (`max_tries: 20` random positions, otherwise skipped) | Otherwise a labelled target could be painted over and its label would be wrong. |

## 4. Interpretations and implementation details

- Flip = horizontal or vertical with 50 % each. The paper only says "flip".
- Scale never shrinks a box below 2 px.
- Sharpness leaves the border pixels unchanged and equalize uses the same lookup table as PIL / torchvision.
- A patch whose weight would be 0 everywhere is skipped (`paper`: max T = 0, where Eq. 6 is undefined;
  `background`: no pixel brighter than b).
- `background`: b = 0 if the patch has no context pixel (box clipped at the frame border on all sides), so
  the weight is Eq. 6 there.
- **Properties of the weight.** It follows the brightness.
  - In the paper's data the targets are warmer than their background; the "dark targets" of Fig. 1(d)
    and Sec. IV-A are dim (low energy), not colder than their surroundings. In ours, cold targets are
    rare (about 1 % of the target library). Such pastes still get a box.
  - `paper` (Eq. 6, default): a cold target gets a low weight and is mostly replaced by the background,
    while its bright surroundings are pasted. If the **surroundings are almost as bright as the target**,
    they get a weight close to 1 (surroundings / peak), so the patch rectangle stays visible (deviation 7).
  - `background`: a **cold (dark)** target gets (almost) no weight: only the noise above b is pasted, so
    the box marks a target that is barely visible.

## 5. Ablations (paper Tables V/VI)

| Paper row | ACB config |
|---|---|
| CutMix (box, no random augment, no fusion) | `num_ops: 0, fusion: none` |
| ACB | defaults (`fusion: paper` = Eq. 6) |
| (not in the paper) one component off | `fusion: none` or `num_ops: 0` alone |
| (not in the paper) ACB with the background-relative weight (deviation 7) | `fusion: background` |
| Copy-Paste with mask; mask + augment + fusion | not available: needs pixel masks. The PGDAA target library could provide pseudo masks (separate experiment). |

The paper's baseline for Tables V/VI is the network with the DLDC module ("no aug." = 0.9139 mIoU =
"+DLDC" in Table III). In our setup the baseline is the YOLO run with the same config and ACB switched off.
