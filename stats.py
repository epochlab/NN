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
