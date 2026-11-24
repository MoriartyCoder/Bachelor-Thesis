"""Section "ir" of the augmentation config: IR camera effects.

Every function takes a grayscale uint8 HxW frame and returns one (see augmentations/uint8.py).
They draw from the global NumPy RNG. `TRANSFORMS` (end of the file) is the registry of the
names allowed in the `ir:` section of a run config.
"""
from functools import partial
from typing import Callable

import albumentations as A
import cv2
import numpy as np

from augmentations.uint8 import RAMP, check_gray_uint8, to_uint8


def netd_noise(image, sigma=0.0):
    """
    NETD-ähnliches Rauschen (Borovskikh et al. 2026):
        T_noise(x, y) = T_ref(x, y) + eta(x, y),   eta ~ N(0, sigma_T^2)
    eta wird für jedes Pixel und bei jedem Aufruf (= jedem Frame) neu gezogen, Mittelwert 0.

    sigma: Standardabweichung relativ zum vollen Wertebereich [0, 1] (also sigma * 255 Grauwerte).
           Größeres sigma = weniger empfindliche (z.B. günstige) Kamera.
    """
    check_gray_uint8(image)
    if sigma <= 0:
        return image.copy()
    eta = np.random.normal(0.0, sigma * 255.0, image.shape)
    return to_uint8(image + eta)

def intensity_shift(image, delta):
    """
    Globaler Intensitäts-Shift (deterministisch):
    Y_tilde = Clip(Y + delta, 0, 255)
    Alle Grauwerte werden um denselben Betrag verschoben (globale Temperatur-Drift).
    Werte, die über 0 / 255 hinausgehen, sättigen; dort geht Kontrast verloren.
    """
    check_gray_uint8(image)
    return cv2.LUT(image, to_uint8(RAMP + float(delta)))

def thermal_histogram_shift(image, delta_limit=30):
    """
    Greyvalue histogram shifting (zufällig):
    Y_tilde = Clip(Y + delta_g, 0, G_max),  delta_g ~ U(-delta_limit, +delta_limit)
    Moduliert tageszeitabhängige Temperaturvariationen und Solar-Drift.
    """
    delta_g = np.random.uniform(-delta_limit, delta_limit)
    return intensity_shift(image, delta_g)

def thermal_histogram_compression(image, gamma_min=0.3, gamma_max=0.9):
    """
    Histogram compression:
    Y_tilde = Y_mid + gamma * (Y - Y_mid)
    Simuliert Thermal Crossover / niedrigen Signal-to-Clutter Ratio (SCR < 2.0).
    """
    check_gray_uint8(image)
    # Faktor gamma aus (0, 1] sampeln (z.B. 0.3 bis 0.9 für starke Kompression)
    gamma = np.random.uniform(gamma_min, gamma_max)

    # Median-Intensität Y_mid des Bildes
    y_mid = float(np.median(image))

    # Kompression um Y_mid als Lookup-Tabelle (das Bild selbst bleibt uint8)
    return cv2.LUT(image, to_uint8(y_mid + gamma * (RAMP - y_mid)))


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


def fpn(img,
            sigma_gain=0.0,       # std of PRNU a(i,j) around 1 (unitless, e.g. 0.01-0.05)
            sigma_offset=0.0,     # std of DSNU b(i,j) around 0 (in grey values, e.g. 1-5)
            mode="pixel"):        # "pixel", "column" or "row"
    """
    Fixed pattern noise model:  y = a * x + b

    a: multiplicative term (PRNU), drawn as N(1, sigma_gain^2)
    b: additive term (DSNU),       drawn as N(0, sigma_offset^2)

    With the default arguments (all sigmas = 0) a = 1, b = 0,
    so the output equals the input.

    mode:
        "pixel"  -> independent a, b per pixel (H x W)
        "column" -> one a, b per column (vertical stripes, typical of uncooled IR readout).
                    Stripe noise (Fayyaz et al. 2022) = mode "column" with sigma_gain = 0,
                    i.e. only an additive offset per column.
        "row"    -> one a, b per row (horizontal stripes)

    The result is rounded and clipped to [0, 255].
    """
    check_gray_uint8(img)
    H, W = img.shape
    shapes = {"pixel": (H, W), "column": (1, W), "row": (H, 1)}
    if mode not in shapes:
        raise ValueError("mode must be 'pixel', 'column' or 'row'")
    if sigma_gain <= 0 and sigma_offset <= 0:
        return img.copy()

    # (1, W) / (H, 1) maps broadcast over the image: stripes
    a = 1.0 + np.random.normal(0.0, sigma_gain, shapes[mode]) if sigma_gain > 0 else 1.0
    b = np.random.normal(0.0, sigma_offset, shapes[mode]) if sigma_offset > 0 else 0.0
    return to_uint8(a * img + b)

def stripe_noise(img, sigma_offset=0.0):
    """
    Stripe noise (Fayyaz et al. 2022): one additive offset per column,
        y(i, j) = x(i, j) + b(j),   b(j) ~ N(0, sigma_offset^2)   (in grey values)
    Same as fpn(img, sigma_gain=0, sigma_offset=sigma_offset, mode="column").
    """
    return fpn(img, sigma_gain=0.0, sigma_offset=sigma_offset, mode="column")

def salt_and_peper(img, p=0.005, q=0.005):
    """
    Applies salt-and-pepper (impulse) noise to an image array u(x):
        f(x) = 0     with probability p
        f(x) = 255   with probability q
        f(x) = u(x)  with probability 1 - p - q

    Parameters:
    -----------
    img : ndarray
        Input clean image u(x), uint8 HxW.
    p : float
        Probability of setting a pixel to 0 ("pepper" noise).
    q : float
        Probability of setting a pixel to 255 ("salt" noise).

    Returns:
    --------
    f : ndarray
        The noisy image, uint8 HxW.
    """
    check_gray_uint8(img)
    if p < 0 or q < 0 or (p + q) > 1.0:
        raise ValueError("Probabilities must satisfy p >= 0, q >= 0, and p + q <= 1.")

    # Copy input image so original is not modified in place
    img_aug = img.copy()

    # Draw uniform random numbers in [0, 1) for every pixel
    rand_map = np.random.random(size=img.shape)

    # Assign 0 with probability p
    img_aug[rand_map < p] = 0

    # Assign 255 with probability q
    img_aug[(rand_map >= p) & (rand_map < (p + q))] = 255

    # Remaining pixels (rand_map >= p + q) keep u(x) with probability 1 - p - q
    return img_aug

def blurring_with_psf(img, sigma_0=2.0, rho=1.5, theta=0.0):
    """
    Applies the parametric anisotropic blur to a single-channel IR image.
    For motion blur???: std: 0.3–3.0

    Parameters:
    -----------
    img : ndarray
        Clean IR image, uint8 HxW.
    sigma_0 : float
        Standard deviation of the main axis.
    rho : float
        Axis ratio (sigma_1 / sigma_0).
    theta : float
        Rotation angle in radians.
    """
    def generate_anisotropic_gaussian_kernel(sigma_0, rho=1.0, theta=0.0, k_size=None):
        """
        Generates an anisotropic Gaussian blur kernel:
            k(x, y) = Z * exp(-(a0*x^2 + 2*a1*x*y + a2*y^2))

        Parameters:
        -----------
        sigma_0 : float
            Standard deviation of the main axis.
        rho : float
            Ratio between the principal axis and the orthogonal one (sigma_1 / sigma_0).
        theta : float
            Angle between the major axis and the horizontal (in radians).
        k_size : int, optional
            Size of the square kernel grid. If None, it is automatically
            calculated based on sigma_0 and rho.
        """
        if sigma_0 <= 0:
            raise ValueError("sigma_0 must be strictly positive.")

        # Automatically determine grid size based on the maximum spread
        sigma_max = sigma_0 * max(1.0, rho)
        if k_size is None:
            radius = int(np.ceil(3.0 * sigma_max))
            k_size = 2 * radius + 1
        else:
            radius = k_size // 2

        # Create coordinate grid centered at (0, 0)
        y, x = np.mgrid[-radius:radius+1, -radius:radius+1]

        # Precompute trigonometric terms and common factors
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rho_sq = rho ** 2
        two_sigma0_sq = 2.0 * (sigma_0 ** 2)

        # Compute coefficients a0, a1, a2 as defined in the model
        a0 = (cos_t ** 2) / two_sigma0_sq + (sin_t ** 2) / (2.0 * rho_sq * (sigma_0 ** 2))
        a1 = (np.sin(2.0 * theta) / (4.0 * (sigma_0 ** 2))) * ((1.0 / rho_sq) - 1.0)
        a2 = (sin_t ** 2) / two_sigma0_sq + (cos_t ** 2) / (2.0 * rho_sq * (sigma_0 ** 2))

        # Evaluate the Gaussian exponent
        exponent = -(a0 * (x ** 2) + 2.0 * a1 * x * y + a2 * (y ** 2))
        kernel = np.exp(exponent)

        # Normalization constant Z so the area (sum) of the kernel is 1
        kernel /= np.sum(kernel)

        return kernel

    check_gray_uint8(img)

    # 1. Generate the anisotropic blur kernel k(x, y)
    kernel = generate_anisotropic_gaussian_kernel(sigma_0, rho, theta)

    # 2. Convolve the IR image with the kernel (k \otimes f), directly on uint8.
    #    filter2D computes a correlation; the kernel is point-symmetric (k(x,y) = k(-x,-y)),
    #    so this equals the convolution. BORDER_REFLECT = scipy's boundary='symm'.
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------------------
# Registry of the `ir:` section
# ---------------------------------------------------------------------------
# Config name -> effect function (uint8 HxW in, uint8 HxW out).
# Optical and motion blur are the same PSF model with different parameters:
#   optical_blur: rho ~ 1 (isotropic / mildly anisotropic), motion_blur: rho < 1, stretched along theta.
# The PSF is centred and point-symmetric, so the image does not shift and the boxes stay exact.
IR_EFFECTS: dict[str, Callable[..., np.ndarray]] = {
    "optical_blur": blurring_with_psf,
    "motion_blur": blurring_with_psf,
    "fpn": fpn,
    "stripe_noise": stripe_noise,
    "netd": netd_noise,
    "salt_and_pepper": salt_and_peper,
    "histogram_shift": thermal_histogram_shift,              # open: section not decided yet
    "histogram_compression": thermal_histogram_compression,  # open: section not decided yet
}


class IRTransform(A.ImageOnlyTransform):
    """Wraps one IR effect as a pixel-only Albumentations transform (boxes are unchanged)."""

    def __init__(self, effect: str, fn_kwargs: dict | None = None, p: float = 0.5):
        super().__init__(p=p)
        if effect not in IR_EFFECTS:
            raise KeyError(f"Unknown IR effect '{effect}'. Available: {sorted(IR_EFFECTS)}")
        self.effect = effect
        self.fn_kwargs = dict(fn_kwargs or {})

    def apply(self, img: np.ndarray, **_) -> np.ndarray:
        return IR_EFFECTS[self.effect](img, **self.fn_kwargs)

    def get_transform_init_args_names(self):
        return ("effect", "fn_kwargs")


# Name in the `ir:` section -> factory(params, p=...) -> transform.
# Unlike the other sections the effect parameters are passed as ONE dict, because an effect may
# have its own parameter called `p` (salt_and_pepper: share of black pixels).
TRANSFORMS: dict[str, Callable[..., A.BasicTransform]] = {name: partial(IRTransform, name) for name in IR_EFFECTS}
