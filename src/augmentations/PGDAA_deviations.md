# PGDAA: deviations from PGDA and how they are handled

**Paper:** A. Wang, W. Li, Z. Huang, X. Wu, F. Jie, R. Tao, "Prior-Guided Data Augmentation for Infrared
Small Target Detection", *IEEE JSTARS* 15 (2022), pp. 10027–10040, DOI 10.1109/JSTARS.2022.3222758.

**Implementation:**
- `PGDAA` in `src/augmentations/sota.py`
- target library: `src/data_handler/target_library.py`, built with `python src/main.py -l`
- config: `augmentation.sota` in `configs/template.yaml`

PGDAA ("PGDA adapted") follows the paper as closely as our data allows. This file lists every point
where it differs, why, and what was done about it. Numbers refer to the train split of Anti-UAV410 and
CST-AntiUAV (library build of 2026-11-24).

## 1. PGDA in short

PGDA creates synthetic training images: small targets pasted onto pure-background crops. Two priors
measured on the training set keep them realistic:

- **A** (Eq. 1): area ratio target pixels / image pixels of every training target. Only values below
  0.15 % are kept (the paper's definition of a small target).
- **S** (Eq. 2): SCR = |μ_t − μ_b| / σ_b. μ_t is the mean over the target pixels; μ_b and σ_b are the
  mean and std of the neighbourhood with d = 20. Only the 20th–80th percentile is kept.

**Subfigure split** (Sec. III-B): multiscale random crops of the training images serve as backgrounds,
preferably pure background (B_O). Each target gets an area ratio β drawn from A (±10 %).

**Target enhancement** (Sec. III-C, Algorithm 1), per target:
1. geometric transforms (flips, affine);
2. non-rigid transforms (elastic, grid distortion, piecewise affine);
3. Gaussian paste: the weights are the Gaussian-blurred mask;
4. SCR perception: if the SCR at the paste position lies outside S, the target's grey values are shifted
   to a random S value, followed by ColorJitter.

Ablation (Table V, DNANet on MSISTD, IoU / nIoU):

| Configuration | IoU | nIoU |
|---|---|---|
| Gaussian blur only | 59.48 | 61.53 |
| + SCR perception | 60.61 | 61.81 |
| Subfigure split only | 63.89 | 65.13 |
| All | 65.08 | 66.87 |

## 2. Deviations forced by our data or training setup

| # | Paper | Our data / setup | Handling in PGDAA | Consequence |
|---|---|---|---|---|
| 1 | Target set T: **every** training target, cut by its **pixel mask** (segmentation labels) | Only bounding boxes | Target library with **pseudo masks** (contrast threshold inside the box against the ring box ± 20 px, closing, hole filling, quality checks). Built from the train split of **all datasets, without tag filter**, each sequence thinned (every 10th frame, at most 100 per sequence) because consecutive frames are near-duplicates. | 19,849 targets (15,136 Anti-UAV410, 4,713 CST) from 311 sequences. Missing: targets whose pseudo mask fails (7,731 rejected as too faint, 364 for other quality reasons). T therefore contains fewer very faint targets than the training set. |
| 2 | **A** = Eq. 1 with **target pixels** | Target pixels exist only for the library subset | A = **box area / frame area** of all training boxes of the run, below `max_area_ratio` = 0.15 %. The pasted target is scaled so that its **box** has the drawn ratio (box against box). | A covers every training target, but in box units instead of pixel units. Box against box keeps the pasted sizes consistent with the real annotations (pixels against box would make the pasted targets 2–3× too large). |
| 3 | **S** = Eq. 2 over **all** targets | μ_t needs a mask | S = Eq. 2 values of the **library** (pseudo mask, ring d = 20, box ± 2 px left out), 20th–80th percentile → 11,909 values in [6.68, 25.99]. | S uses exactly the definition the SCR perception applies to the pasted target. Bias: the faintest targets are missing (item 1), which shifts S slightly upwards; the 20 % lower cut softens this. |
| 4 | 0.15 % limit = small targets | Only 32.5 % of the Anti-UAV410 targets and 95.9 % of the CST targets are below 0.15 % | Limit kept as in the paper (`max_area_ratio`, `None` = no limit). | PGDAA only adds small targets, which is the paper's intent. Large Anti-UAV410 drones are not reproduced. |
| 5 | Pure backgrounds B_O from subfigure crops of the images, using the segmentation to tell B_T from B_O | Boxes only | B_O = training frames **without a box** (23,238: 5,163 Anti-UAV410 + 18,075 CST). Algorithm 1 (line 1) uses B_O only, so B_T is never used. | Same as the paper's preferred case. Frames without a box are frames where the target is out of view; a faint unannotated target in them cannot be excluded. |
| 6 | Background resolution is changed so that the target has area ratio β | Fixed input size (frames 640×512, YOLO letterbox 640) | Crop → resize to the frame size → **the target is scaled** to β (each target its own β, as in Fig. 5). | The area ratio is the same; only which part gets resampled differs (the paper resamples the background, we resample the target). The detector's resize resamples both anyway. |
| 7 | **Offline**: synthetic images are generated once and added to the training set (synthesis ratio r = 0.5…2, Fig. 3) | Online augmentation pipeline (Albumentations) | Online transform: with probability p the training sample is **replaced** by a synthetic image. Its boxes are only the pasted targets. p = r / (1 + r) (r = 1 → p = 0.5). | Same share of synthetic samples in expectation, and new synthetic images every epoch (more variety than a fixed offline set). |
| 8 | All targets are warmer than their background ("no cases where the temperature of the target is lower") | 208 library targets (1 %) are dark (cold) | Only **bright** library targets are used (19,641). | Exactly as in the paper; cold drones are not synthesised. |

## 3. Points the paper leaves open, and our choices

All of them are config parameters (`params` of the `PGDAA` entry), so they can be varied and reported.

| Open point | Choice (default) | Reason |
|---|---|---|
| N_t, targets per image | `num_targets: 1` (int or [min, max]) | The paper only says the number of targets per image should stay as in the data; our frames have one target. |
| Size range of the subfigure crops | `crop_scale: [0.5, 1.0]` (side length as a fraction of the frame) | Not given; moderate zoom up to 2×. |
| σ of the Gaussian paste | `gaussian_sigma: 1.0` px | Not given; σ = 0 is a hard mask. |
| Strengths of the geometric and non-rigid transforms | `DEFAULT_TARGET_OPS` in `sota.py`: flips p = 0.5, rotation ±30°, elastic (α 10, σ 3), grid distortion (3 steps, ±0.2), piecewise affine (scale 0.02–0.04, 3×3), each p = 0.5; overridable with `target_ops` | The paper names the operations, not their parameters. Moderate values for patches of about 10–100 px. |
| ColorJitter after the SCR shift | brightness and contrast ±0.1 (`color_jitter`) | IR is single-channel, so saturation and hue do not apply. |
| "Data binning" for the 20–80 % cut | Plain percentiles (`scr_percentiles: [20, 80]`) | The binning is not described. |
| Paste position | Random; the target fully inside the frame; no overlap between pasted targets (`max_tries: 20`) | Not specified. Overlaps would make the boxes ambiguous. |
| Neighbourhood R for the SCR check | Window (target patch ± d) of the background **before** pasting, d = 20; σ ≥ 1 | Matches "location neighbourhood R(B_o(x_o, y_o)) with radius d". |

## 4. Interpretations and implementation details

- **Signed SCR in the check (Algorithm 1, line 9).** Eq. 2 uses |μ_t − μ_b|, but SCR perception is meant
  to keep the target warmer than its background, and the shift Diff = S_i·std(R) − avg(T) + avg(R) always
  makes it warmer. With |·|, a target that is clearly *colder* than its new background would pass the
  check unchanged. PGDAA therefore checks the signed value (avg(T) − avg(R)) / std(R). This is exactly
  the paper's "background migration" problem (Fig. 4), which SCR perception is supposed to fix.
- **Saturation.** If avg(R) + S_i·std(R) exceeds 255 (bright backgrounds such as building walls), the
  target is clipped at 255. Many library targets are already saturated (peak ≥ 250: 10,559 of 15,136
  Anti-UAV410, 561 of 4,713 CST). The paper does not discuss this.
- **Interpolation at the mask edge.** The cutouts are 0 outside the mask. Rotation, distortion and
  resizing would mix this 0 into the edge pixels of the target, giving it a cold rim. Before the shape
  transforms, the outside of the cutout is therefore filled from the nearest target pixels (OpenCV
  inpainting), and set to 0 again afterwards. This is not a change to the method; it only avoids an
  artifact the paper's real masks would also produce.
- **Box of the pasted target.** The annotated box of the library target is carried through all
  transforms (Albumentations bbox handling) and scaled with the target. If a transform loses it, the
  tight box of the mask is used.
- **Order inside the target enhancement:** gt → nt → scaling to β → gp → SCR perception → paste.
  Scaling belongs to the subfigure split in the paper. It comes before gp so that the Gaussian σ acts on
  the final pixel size.
- **Priors per run.** A and B_O come from the run's training records (after the configured frame
  sampling). Even in a dry run they come from the full training split. S and T come from the library.

## 5. Ablations (paper Table V)

| Paper row | PGDAA config |
|---|---|
| Gaussian blur only (≈ plain copy-paste) | `subfigure_split: false, scr_perception: false` |
| Gaussian blur + SCR perception | `subfigure_split: false` |
| Subfigure split only | `scr_perception: false, gaussian_sigma: 0` |
| Full PGDA | defaults |

With `subfigure_split: false` the whole background frame is used and the targets keep their library
size. Without the subfigure split our version still pastes onto pure backgrounds, while the paper's
copy-paste baseline (Table IV) pastes onto the original images. The paper does not say which
background Table V's first row used.
