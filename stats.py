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
    """PCA of (..., c) samples, e.g. an (h, w, c) image: mean, variances and unit axes (columns) largest first, and each sample's projection."""
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
