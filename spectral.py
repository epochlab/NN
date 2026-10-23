from typing import Tuple

import numpy as np

from pathtracer import Camera

def lattice_geometry(camera: Camera) -> Tuple[float, float]:
    """Row pitch and odd-row shift in column pitches, per Camera.pixel_lattice."""
    if camera.pixel_lattice == "hexagonal": return np.sqrt(3.0) / 2.0, 0.5
    return 1.0, 0.0

def row_twiddle(shape: Tuple[int, int], row_shift: float) -> np.ndarray:
    """Per-row phase exp(-2 pi i fx shift(y)) that moves each row's DFT to its true sample positions."""
    return np.exp(-2j * np.pi * np.outer(row_shift * (np.arange(shape[0]) % 2), np.fft.fftfreq(shape[1])))

def fft(img: np.ndarray, twiddle: np.ndarray) -> np.ndarray:
    """DFT of a (height, width) image over its pixel centres' lattice, in fft order."""
    return np.fft.fft(np.fft.fft(img, axis=1) * twiddle, axis=0)

def ifft(spectrum: np.ndarray, twiddle: np.ndarray) -> np.ndarray:
    """Inverse of fft: a (height, width) image on the pixel lattice from its lattice DFT in fft order."""
    return np.fft.ifft(np.fft.ifft(spectrum, axis=0) * twiddle.conj(), axis=1)

def bin_freq(shape: Tuple[int, int], row_pitch: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bin frequencies (fy, fx) in cycles per column pitch, in fft order."""
    return np.fft.fftfreq(shape[0], d=row_pitch), np.fft.fftfreq(shape[1])

def alias_radius_sq(fy: np.ndarray, fx: np.ndarray, row_pitch: float, row_shift: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each bin folded to its nearest reciprocal-lattice point: squared radius and (fy, fx), the bin's true spatial frequency."""
    b1 = np.array([1.0, -row_shift / row_pitch])
    b2 = np.array([0.0, 1.0 / row_pitch])
    radius_sq = np.full((fy.size, fx.size), np.inf)
    true_fy, true_fx = np.zeros_like(radius_sq), np.zeros_like(radius_sq)
    for i in (0, -1, 1):
        for j in (0, -1, 1):
            ox, oy = i * b1 + j * b2
            dx, dy = np.broadcast_arrays(fx[None, :] - ox, fy[:, None] - oy)
            d = dx ** 2 + dy ** 2
            closer = d < radius_sq
            radius_sq[closer], true_fy[closer], true_fx[closer] = d[closer], dy[closer], dx[closer]
    return radius_sq, true_fy, true_fx
