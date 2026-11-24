import cv2, os, torch
import numpy as np
from scipy.signal import convolve2d



def netd_noise(image, sigma=0.0):
    """
    NETD-ähnliches Rauschen (Borovskikh et al. 2026):
        T_noise(x, y) = T_ref(x, y) + eta(x, y),   eta ~ N(0, sigma_T^2)
    eta wird für jedes Pixel und bei jedem Aufruf (= jedem Frame) neu gezogen, Mittelwert 0.

    sigma: Standardabweichung relativ zum vollen Wertebereich [0, 1] (bei 8 bit: sigma * 255 Grauwerte).
           Größeres sigma = weniger empfindliche (z.B. günstige) Kamera.
    image: Ganzzahl-Bild (z.B. uint8, Wertebereich 0..dtype-Max) oder Float-Bild in [0, 1].
           Rückgabe im selben dtype, auf den Wertebereich geclippt.
    """
    image = np.asarray(image)
    integer = np.issubdtype(image.dtype, np.integer)
    scale = float(np.iinfo(image.dtype).max) if integer else 1.0

    eta = np.random.normal(0.0, sigma * scale, image.shape)
    noisy = np.clip(image.astype(np.float32) + eta, 0.0, scale)

    if integer:
        noisy = np.rint(noisy)
    return noisy.astype(image.dtype)

def intensity_shift(image, delta):
    """
    Globaler Intensitäts-Shift (deterministisch):
    Y_tilde = Clip(Y + delta, 0, 255)
    Alle Grauwerte werden um denselben Betrag verschoben (globale Temperatur-Drift).
    Werte, die über 0 / 255 hinausgehen, sättigen; dort geht Kontrast verloren.
    """
    shifted = image.astype(np.float32) + float(delta)
    return np.clip(shifted, 0, 255).astype(np.uint8)

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
    # Faktor gamma aus (0, 1] sampeln (z.B. 0.3 bis 0.9 für starke Kompression)
    gamma = np.random.uniform(gamma_min, gamma_max)
    
    # Median-Intensität Y_mid des Bildes berechnen
    y_mid = np.median(image)
    
    # Kompression um Y_mid durchführen
    compressed = y_mid + gamma * (image.astype(np.float32) - y_mid)
    
    # Clippen und zurück zu uint8
    return np.clip(compressed, 0, 255).astype(np.uint8)


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


def fpn(img,
            sigma_gain=0.0,       # std of PRNU a(i,j) around 1 (unitless, e.g. 0.01-0.05)
            sigma_offset=0.0,     # std of DSNU b(i,j) around 0 (in intensity units, e.g. 1-5 on 0-255)
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

    Integer images are clipped to the range of their dtype; float images are not clipped.
    For multi-channel images the same (a, b) pattern is applied to every channel.
    """
    img = np.asarray(img)
    H, W = img.shape[:2]

    def draw(mean, sigma):
        if mode == "pixel":
            shape = (H, W)
        elif mode == "column":
            shape = (1, W)
        elif mode == "row":
            shape = (H, 1)
        else:
            raise ValueError("mode must be 'pixel', 'column' or 'row'")
        m = np.full(shape, mean, dtype=np.float32)
        if sigma > 0:
            m += np.random.normal(0.0, sigma, size=shape).astype(np.float32)
        return np.broadcast_to(m, (H, W)).copy()

    a_map = draw(1.0, sigma_gain)
    b_map = draw(0.0, sigma_offset)

    x = img.astype(np.float32)
    if x.ndim == 3:
        a_map = a_map[..., None]
        b_map = b_map[..., None]

    y = a_map * x + b_map

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        y = np.rint(np.clip(y, info.min, info.max))
    return y.astype(img.dtype)

def salt_and_peper(img, p=0.005, q=0.005):
    """
    Applies salt-and-pepper (impulse) noise to an image array u(x):
        f(x) = s_min   with probability p
        f(x) = s_max   with probability q
        f(x) = u(x)    with probability 1 - p - q

    s_min / s_max are taken from the dtype: integer images 0 / dtype max (uint8: 0 / 255),
    float images 0 / 1.

    Parameters:
    -----------
    img : ndarray
        Input clean image array u(x).
    p : float
        Probability of setting a pixel to s_min ("pepper" noise).
    q : float
        Probability of setting a pixel to s_max ("salt" noise).

    Returns:
    --------
    f : ndarray
        The noisy image array with the same shape and dtype as u.
    """
    if p < 0 or q < 0 or (p + q) > 1.0:
        raise ValueError("Probabilities must satisfy p >= 0, q >= 0, and p + q <= 1.")

    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.integer):
        s_min, s_max = 0, np.iinfo(img.dtype).max
    else:
        s_min, s_max = 0.0, 1.0

    # Copy input image so original is not modified in place
    img_aug = np.array(img, copy=True)
    
    # Draw uniform random numbers in [0, 1) for every pixel
    rand_map = np.random.random(size=img.shape)
    
    # Assign s_min with probability p
    img_aug[rand_map < p] = s_min
    
    # Assign s_max with probability q
    img_aug[(rand_map >= p) & (rand_map < (p + q))] = s_max
    
    # Remaining pixels (rand_map >= p + q) keep u(x) with probability 1 - p - q
    return img_aug

def blurring_with_psf(img, sigma_0=2.0, rho=1.5, theta=0.0):
    """
    Applies the parametric anisotropic blur to a single-channel IR image.
    For motion blur???: std: 0.3–3.0

    Parameters:
    -----------
    img : ndarray (2D)
        Single-channel clean IR image.
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

    # 1. Generate the anisotropic blur kernel k(x, y)
    kernel = generate_anisotropic_gaussian_kernel(sigma_0, rho, theta)
    
    # 2. Convolve the single-channel IR image with the kernel (k \otimes f)
    img_aug = convolve2d(img, kernel, mode='same', boundary='symm')

    return img_aug

