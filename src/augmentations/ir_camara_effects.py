import cv2, os, torch
import numpy as np
from scipy.signal import convolve2d



def netd_augmentation(image, min_sigma=2.0, max_sigma=16.0, alpha_range=(0.65, 0.95)):
    """
    Simuliert unterschiedliche NETD-Qualitäten für TIR-Bilder im Wertebereich [0, 255] (NumPy).
    
    Parameters:
    - image: np.ndarray im Bereich [0, 255], Shape: (H, W) oder (H, W, C)
    - min_sigma / max_sigma: Standardabweichung des Gauß-Rauschens (direkt auf 0-255 Skala)
    - contrast_range: Steuert den Kontrastverlust
    """
    # Kopie erstellen und zu float32 konvertieren (wichtig für saubere Rausch- und Kontrastberechnung)
    augmented = image.copy().astype(np.float32)
    
    # 1. Kontrast- und Dynamikreduktion (Simuliert sinkenden thermischen Kontrast)
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    mean = augmented.mean()
    augmented = mean + alpha * (augmented - mean)
    
    # 2. Gaußsches Rauschen (Simuliert das thermische Eigenrauschen des Sensors / NETD)
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0.0, sigma, augmented.shape)
    augmented = augmented + noise

    # Sicherstellen, dass die Werte im gültigen Bereich [0, 255] bleiben
    return np.clip(augmented, 0.0, 255.0)

def simulate_fpn_corruption(clean_image, alpha=1.0, noise_type='laplace'):
    """
    Simuliert FPN-Korruption exakt nach dem Signalmodell aus Song & Huang (2023):
    Y = X + alpha * M
    
    Parameter:
    - clean_image: Das saubere Eingangsbild (z.B. Grauwertbild aus Dataset)
    - alpha: Skalierungsparameter für das FPN (abhängig von Temperatur/Belichtung)
    - noise_type: 'gaussian', 'laplace' (wichtig wegen 'Hot Pixels') oder 'uniform'
    """

    # Sicherstellen, dass das Bild im Float-Format vorliegt
    img_float = clean_image.astype(np.float32)
    h, w = img_float.shape[:2]
    
    # 1. Noise Pattern M basierend auf den im Paper genannten Verteilungen generieren
    if noise_type == 'gaussian':
        M = np.random.normal(loc=0.0, scale=1.0, size=(h, w))
    elif noise_type == 'laplace':
        # Laplace eignet sich besonders gut für FPN, da es "Heavy Tails" 
        # und damit realistische "Hot Pixels" simuliert (siehe Paper Abschnitt V)
        M = np.random.laplace(loc=0.0, scale=1.0 / np.sqrt(2), size=(h, w))
    elif noise_type == 'uniform':
        M = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(h, w))
    else:
        raise ValueError(f"Unbekannter Rauschtyp: {noise_type}")
        
    # Varianz des Noise Patterns auf 1 normieren (wie im Paper beschrieben)
    M = M / np.std(M)
    
    # 2. Lineares Signalmodell anwenden: Y = X + alpha * M
    noisy_image = img_float + alpha * M
    
    # Auf gültigen Bereich [0, 255] clippen
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8), M


def add_common_mode_noise(image, row_noise_level=0.03, col_noise_level=0.01):
    """
    Fügt einem Bild (z.B. normalisiert auf [0, 1]) Common Mode Noise 
    in Form von Zeilen- und Spaltenstreifen hinzu.
    """
    h, w = image.shape[:2]
    noisy_img = image.copy()
    
    # 1. Zeilenrauschen (Row Noise): 
    # Ein Rauschwert pro Zeile, der über alle Spalten gezogen wird.
    row_noise = np.random.normal(0, row_noise_level, (h, 1))
    
    # 2. Spaltenrauschen (Column Noise):
    # Ein Rauschwert pro Spalte, der über alle Zeilen gezogen wird.
    col_noise = np.random.normal(0, col_noise_level, (1, w))
        
    noisy_img = noisy_img + row_noise + col_noise
    
    # 3. Optional: Globales Frame-Flicker (Gesamthelligkeit schwankt leicht)
    global_flicker = np.random.normal(0, 0.02)
    noisy_img = noisy_img + global_flicker
    
    # Wertebereich absichern
    return np.clip(noisy_img, 0.0, 1.0)


def thermal_histogram_shift(image, delta_limit=30):
    """
    Greyvalue histogram shifting:
    Y_tilde = Clip(Y + delta_g, 0, G_max)
    Moduliert tageszeitabhängige Temperaturvariationen und Solar-Drift.
    """
    # delta_g gleichmäßig aus dem Bereich [-delta, +delta] sampeln
    delta_g = np.random.uniform(-delta_limit, delta_limit)
    
    # Verschiebung anwenden und auf [0, 255] clippen
    shifted = image.astype(np.float32) + delta_g
    return np.clip(shifted, 0, 255).astype(np.uint8)


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


def fpn2(img, prnu=None, dcnu=None, n_random=None, 
                         sigma_prnu=0.01, dcnu_mean=0.0, dcnu_std=2.0, 
                         read_noise_std=1.0):
    """
    Simulates a noisy image using the standard sensor noise model:
    I_noisy(i,j) = [I_clean(i,j) * (1 + PRNU(i,j))] + DCNU(i,j) + N_random(i,j)
    
    Parameters:
    -----------
    I_clean : ndarray
        The true incoming irradiance/signal image.
    prnu : ndarray, optional
        Multiplicative Photo-Response Non-Uniformity map. If None, 
        generated as a zero-mean Gaussian field.
    dcnu : ndarray, optional
        Additive Dark Current Non-Uniformity map. If None, 
        generated as a spatial Gaussian map.
    n_random : ndarray, optional
        Time-varying temporal noise (shot + read noise). If None, 
        generated using shot noise (Poisson) and Gaussian read noise.
    sigma_prnu : float
        Standard deviation for PRNU (default 0.01 - 0.02).
    dcnu_mean, dcnu_std : float
        Parameters for generating synthetic DCNU if not provided.
    read_noise_std : float
        Standard deviation for temporal read noise.
        
    Returns:
    --------
    I_noisy : ndarray
        The resulting simulated noisy image.
    """
    shape = img.shape
    
    # 1. Multiplicative PRNU component
    if prnu is None:
        # Zero-mean Gaussian field with small standard deviation
        prnu = np.random.normal(loc=0.0, scale=sigma_prnu, size=shape)
        
    # 2. Additive DCNU component
    if dcnu is None:
        # Modeled as a spatial Gaussian or offset map
        dcnu = np.random.normal(loc=dcnu_mean, scale=dcnu_std, size=shape)
        
    # 3. Time-varying temporal noise N_random (Shot noise + Read noise)
    if n_random is None:
        # Shot noise depends on the signal magnitude (Poissonian approximation)
        # For general signals, we can simulate shot noise or use a normal approximation
        shot_noise = np.random.normal(loc=0.0, scale=np.sqrt(np.maximum(img, 0)), size=shape)
        read_noise = np.random.normal(loc=0.0, scale=read_noise_std, size=shape)
        n_random = shot_noise + read_noise

    # Apply the full equation
    I_noisy = (img * (1.0 + prnu)) + dcnu + n_random
    
    return I_noisy
import numpy as np


def fpn(img,
            sigma_gain=0.0,       # std of PRNU a(i,j) around 1 (unitless, e.g. 0.01-0.05)
            sigma_offset=0.0,     # std of DSNU b(i,j) around 0 (in intensity units, e.g. 1-5 on 0-255)
            sigma_noise=0.0,      # std of random noise n(i,j) (intensity units)
            mode="pixel",         # "pixel", "column" or "row"
            a=None, b=None,       # optional fixed gain/offset maps (override sigma_gain/sigma_offset)
            value_range=None,     # (lo, hi) clip range; inferred from dtype if None
            rng=None,
            return_maps=False):
    """
    Fixed pattern noise model (Eq. 1):  y = a * x + b + n

    a: multiplicative term (PRNU), drawn as N(1, sigma_gain^2)
    b: additive term (DSNU),       drawn as N(0, sigma_offset^2)
    n: random noise,               drawn as N(0, sigma_noise^2)

    With the default arguments (all sigmas = 0, no maps given) a = 1, b = 0, n = 0,
    so the output equals the input.

    mode:
        "pixel"  -> independent a, b per pixel (H x W)
        "column" -> one a, b per column (vertical stripes, typical of uncooled IR readout)
        "row"    -> one a, b per row (horizontal stripes)

    FPN is a property of the sensor, so it is the same across frames. To apply
    the same pattern to several images, pass a and b (use return_maps=True to get them),
    or reuse an rng seeded identically.

    For multi-channel images the same (a, b) pattern is applied to every channel,
    while n is drawn independently per element.
    """
    rng = np.random.default_rng() if rng is None else rng
    img = np.asarray(img)
    H, W = img.shape[:2]

    # clip range
    if value_range is None:
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            value_range = (info.min, info.max)
        else:
            value_range = None          # float input: no clipping unless requested

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
            m += rng.normal(0.0, sigma, size=shape).astype(np.float32)
        return np.broadcast_to(m, (H, W)).copy()

    a_map = draw(1.0, sigma_gain) if a is None else np.asarray(a, dtype=np.float32)
    b_map = draw(0.0, sigma_offset) if b is None else np.asarray(b, dtype=np.float32)

    x = img.astype(np.float32)
    if x.ndim == 3:
        a_map = a_map[..., None]
        b_map = b_map[..., None]

    y = a_map * x + b_map
    if sigma_noise > 0:
        y += rng.normal(0.0, sigma_noise, size=x.shape).astype(np.float32)

    if value_range is not None:
        y = np.clip(y, *value_range)
    if np.issubdtype(img.dtype, np.integer):
        y = np.rint(y)
    y = y.astype(img.dtype)

    if return_maps:
        return y, np.squeeze(a_map), np.squeeze(b_map)
    return y

def salt_and_peper(img, p=0.005, q=0.005, s_min=0.0, s_max=1.0, seed=None):
    """
    Applies salt-and-pepper (impulse) noise to an image array u(x):
        f(x) = s_min   with probability p
        f(x) = s_max   with probability q
        f(x) = u(x)    with probability 1 - p - q

    Parameters:
    -----------
    u : ndarray
        Input clean image array, expected to be in range [s_min, s_max] (default [0, 1]).
    p : float
        Probability of setting a pixel to s_min ("pepper" noise).
    q : float
        Probability of setting a pixel to s_max ("salt" noise).
    s_min : float
        Minimum intensity value (default 0.0).
    s_max : float
        Maximum intensity value (default 1.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    f : ndarray
        The noisy image array with the same shape as u.
    """
    if p < 0 or q < 0 or (p + q) > 1.0:
        raise ValueError("Probabilities must satisfy p >= 0, q >= 0, and p + q <= 1.")

    rng = np.random.default_rng(seed)
    
    # Copy input image so original is not modified in place
    img_aug = np.array(img, copy=True)
    
    # Draw uniform random numbers in [0, 1) for every pixel
    rand_map = rng.random(size=img.shape)
    
    # Assign s_min with probability p
    img_aug[rand_map < p] = s_min
    
    # Assign s_max with probability q
    img_aug[(rand_map >= p) & (rand_map < (p + q))] = s_max
    
    # Remaining pixels (rand_map >= p + q) keep u(x) with probability 1 - p - q
    return img_aug


def motion_blur(img, tau_im=5.0, kernel_length=None, noise_std=0.0):
    """
    Simulates motion blur and noise using the model:
        g = h (x) f + n
        h(x) = (1 / tau_im) * exp(-x / tau_im) for x >= 0, else 0

    Parameters:
    -----------
    f : ndarray
        Input clean image (2D grayscale or 3D RGB).
    tau_im : float
        Time constant / decay parameter for the exponential kernel.
    kernel_length : int, optional
        Length of the 1D motion kernel. If None, automatically 
        set based on tau_im (e.g., 5 * tau_im).
    noise_std : float
        Standard deviation of the additive noise component n.

    Returns:
    --------
    g : ndarray
        The resulting blurred and noisy image.
    """
    if kernel_length is None:
        kernel_length = max(int(np.ceil(5 * tau_im)), 3)
        
    # 1. Generate the 1D exponential kernel h(x) for x >= 0
    x = np.arange(kernel_length, dtype=np.float32)
    h = (1.0 / tau_im) * np.exp(-x / tau_im)
    
    # Normalize kernel so image brightness is preserved (energy conservation)
    h /= np.sum(h)
    
    # Reshape into a 2D horizontal motion blur filter (1 x kernel_length)
    h_2d = h.reshape(1, -1)
    
    # 2. Convolve image with kernel h (h \otimes f)
    if img.ndim == 3:
        # Handle multi-channel (RGB) images channel-by-channel
        blurred = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            blurred[:, :, c] = convolve2d(img[:, :, c], h_2d, mode='same', boundary='symm')
    else:
        blurred = convolve2d(img, h_2d, mode='same', boundary='symm')
        
    # 3. Add noise component n
    if noise_std > 0:
        n = np.random.normal(loc=0.0, scale=noise_std, size=img.shape)
        img_aug = blurred + n
    else:
        img_aug = blurred
        
    return img_aug


def blurring_with_psf(img, sigma_0=2.0, rho=1.5, theta=0.0, noise_std=0.0):
    """
    Applies the parametric anisotropic blur and optional noise to a single-channel IR image.
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
    noise_std : float
        Standard deviation for optional additive noise.
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
    
    # 3. Add optional noise component
    if noise_std > 0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=img.shape)
        img_aug += noise
        
    return img_aug


def stripe_noise(img, beta=0.1, rng=None):
    """
    Adds column-wise stripe noise to a single-channel IR image.

    Model (Eqs. 1-3):
        y_i = x_i + s_i          (i-th column)
        s_i = (s_i, ..., s_i)    all n elements in a column are equal
        s_i ~ N(0, sigma^2)      independent per column
        sigma ~ U(0, beta)       drawn once per image

    Parameters
    ----------
    img : ndarray (n, m)
        Clean single-channel image X.
    beta : float
        Upper bound of the noise std; controls noise power.
        Choose relative to the value range of img
        (e.g. 0.05-0.2 for images in [0, 1], ~10-50 for 0..255).
    rng : np.random.Generator or None
        Optional generator for reproducibility.

    Returns
    -------
    img_aug : ndarray (n, m)
        Degraded image Y = X + S.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, m = img.shape

    # Eq. (3): one sigma per image
    sigma = rng.uniform(0.0, beta)

    # Eq. (2): one scalar per column, shape (1, m)
    s = rng.normal(loc=0.0, scale=sigma, size=(1, m))

    # Eq. (1): broadcast the same value down every row of a column
    S = np.broadcast_to(s, (n, m))

    img_aug = img.astype(np.float64) + S
    return img_aug
