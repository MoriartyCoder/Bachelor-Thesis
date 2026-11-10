import cv2, os, torch
import numpy as np


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


def thermal_histogram_shift(image, delta_limit=30, **kwargs):
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


def thermal_histogram_compression(image, gamma_min=0.3, gamma_max=0.9, **kwargs):
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
