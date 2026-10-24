from typing import Tuple

import numpy as np

def pixel_distribution(img: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """Counts per value 0-255 (index, all, then each channel) and summary stats of a (height, width, 1 or 3) uint8 image."""
    img = np.asarray(img)
    height, width, channels = img.shape
    pixel_count = np.zeros([2 + channels, 256], dtype=int)

    pixel_count[0] = np.arange(256)
    pixel_count[1] = np.bincount(img.ravel(), minlength=256)
    for c in range(channels):
        pixel_count[2 + c] = np.bincount(img[..., c].ravel(), minlength=256)

    counts = pixel_count[1]
    occupied = np.nonzero(counts)[0]
    pixels = height * width
    median = np.argmax(np.cumsum(counts) > counts.sum() / 2)
    nonzero = [occupied[0], occupied[-1]]
    empty_bins = np.sum(counts == 0)
    unique_colors = np.unique(img.reshape(pixels, channels), axis=0).shape[0]

    return pixel_count, (pixels, img.min(), img.max(), occupied[np.argmin(counts[occupied])], np.argmax(counts), round(np.mean(img), 2), round(np.std(img), 2), median, nonzero, empty_bins, unique_colors)

def principal_components(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA of (..., c) samples: mean, variances and unit axes (columns) largest first, and each sample's (..., c) projection."""
    pixels = img.reshape(-1, img.shape[-1])
    # float64 mean promotes centring to float64: float32 rounding of pixels - mean correlates the projections at ~1e-8 of PC1
    mean = pixels.mean(axis=0, dtype=np.float64)
    variance, axes = np.linalg.eigh(np.cov(pixels, rowvar=False))
    variance, axes = variance[::-1], axes[:, ::-1]
    # eigh signs are arbitrary: make each axis's largest loading positive so projections are reproducible
    axes *= np.sign(axes[np.abs(axes).argmax(axis=0), np.arange(axes.shape[1])])
    return mean, variance, axes, ((pixels - mean) @ axes).reshape(img.shape)

def independent_components(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Symmetric FastICA with g = tanh (Hyvärinen 1999) of sphered (n, d) data: the orthogonal (d, d) unmixing, one component per row."""
    def decorrelate(w: np.ndarray) -> np.ndarray:
        # (W Wᵀ)^-1/2 W, the nearest orthogonal matrix: every component is updated in parallel, none favoured (Hyvärinen & Oja 2000 §6.2)
        s, u = np.linalg.eigh(w @ w.T)
        return (u / np.sqrt(s)) @ u.T @ w

    w = np.linalg.qr(rng.standard_normal((z.shape[1], z.shape[1])))[0]
    while True:
        y = np.tanh(z @ w.T)
        update = decorrelate(y.T @ z / len(z) - (1 - y ** 2).mean(axis=0)[:, None] * w)
        # Fixed point up to sign: each row's cosine with its predecessor within sqrt(eps) of 1, MINPACK's default tolerance
        converged = 1 - np.abs(np.sum(update * w, axis=1)).min() < np.sqrt(np.finfo(float).eps)
        w = update
        if converged: return w

def sparse_coding(z: np.ndarray, atoms: int, noise: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Olshausen & Field (1997) MAP sparse coding of whitened (n, d) data, noise per dimension: (d, atoms) dictionary, (n, atoms) codes."""
    d = z.shape[1]
    # Laplace scale b by moment matching: unit-variance z carries signal energy d(1 - noise) = atoms · 2b² over unit-norm atoms
    scale = np.sqrt(d * (1 - noise) / (2 * atoms))
    # MAP coefficients solve the lasso ½‖z - Φa‖² + (noise / b)‖a‖₁; dividing by noise gives the negative log posterior in nats
    penalty = noise / scale
    nats = lambda codes, dictionary: (0.5 * np.sum((z - codes @ dictionary.T) ** 2) + penalty * np.abs(codes).sum()) / noise
    dictionary = rng.standard_normal((d, atoms))
    dictionary /= np.linalg.norm(dictionary, axis=0)
    codes = np.zeros((len(z), atoms))
    previous = np.inf
    while True:
        # FISTA (Beck & Teboulle 2009) warm-started from the last codes, step 1 / ‖Φ‖₂², the gradient's Lipschitz constant
        gram, target = dictionary.T @ dictionary, z @ dictionary
        step = 1 / np.linalg.eigvalsh(gram)[-1]
        search, momentum, inner = codes, 1.0, nats(codes, dictionary)
        while True:
            update = search - step * (search @ gram - target)
            update = np.sign(update) * np.maximum(np.abs(update) - step * penalty, 0)
            accelerated = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
            search = update + (momentum - 1) / accelerated * (update - codes)
            codes, momentum = update, accelerated
            value = nats(codes, dictionary)
            # Under 1 nat is a posterior ratio below e, within "not worth more than a bare mention" (Jeffreys 1961; Kass & Raftery 1995)
            if inner - value < 1: break
            inner = value
        # Checked after inference, so the codes returned are MAP for the dictionary returned
        if previous - value < 1: return dictionary, codes
        previous = value
        energy, correlation = codes.T @ codes, z.T @ codes
        # Each used atom's exact minimiser, rest fixed, on the unit ball (Mairal et al. 2010, Alg. 2); unused atoms keep their place
        for j in np.flatnonzero(np.diag(energy)):
            atom = dictionary[:, j] + (correlation[:, j] - dictionary @ energy[:, j]) / energy[j, j]
            dictionary[:, j] = atom / max(np.linalg.norm(atom), 1)
