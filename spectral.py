#!/usr/bin/env python3

from typing import Tuple

import math
import torch

from pathtracer import Camera

def lattice_geometry(camera: Camera) -> Tuple[float, float]:
    """Row pitch and odd-row shift in column pitches, per Camera.pixel_lattice."""
    if camera.pixel_lattice == "hexagonal": return math.sqrt(3.0) / 2.0, 0.5
    return 1.0, 0.0

def row_twiddle(shape: Tuple[int, int], row_shift: float, device: torch.device) -> torch.Tensor:
    """Per-row phase exp(-2 pi i fx shift(y)) that moves each row's DFT to its true sample positions."""
    shift = row_shift * (torch.arange(shape[0], device=device) % 2)
    return torch.exp(-2j * math.pi * torch.outer(shift, torch.fft.fftfreq(shape[1], device=device)))

def fft(img: torch.Tensor, twiddle: torch.Tensor) -> torch.Tensor:
    """DFT of (..., height, width) images over their pixel centres' lattice, in fft order."""
    return torch.fft.fft(torch.fft.fft(img, dim=-1) * twiddle, dim=-2)

def ifft(spectrum: torch.Tensor, twiddle: torch.Tensor) -> torch.Tensor:
    """Inverse of fft: (..., height, width) images on the pixel lattice from their lattice DFT in fft order."""
    return torch.fft.ifft(torch.fft.ifft(spectrum, dim=-2) * twiddle.conj(), dim=-1)

def bin_freq(shape: Tuple[int, int], row_pitch: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bin frequencies (fy, fx) in cycles per column pitch, in fft order."""
    return torch.fft.fftfreq(shape[0], d=row_pitch, device=device), torch.fft.fftfreq(shape[1], device=device)

def alias_radius_sq(fy: torch.Tensor, fx: torch.Tensor, row_pitch: float, row_shift: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each bin folded to its nearest reciprocal-lattice point: squared radius and (fy, fx), the bin's true spatial frequency."""
    # Reciprocal basis b1 = (1, -shift / pitch), b2 = (0, 1 / pitch); argmin keeps the first of tied images, order 0, -1, 1
    i, j = (t.reshape(-1, 1, 1) for t in torch.meshgrid(*2 * (torch.tensor([0.0, -1.0, 1.0], device=fy.device),), indexing="ij"))
    dy = (fy[:, None] + (i * row_shift - j) / row_pitch).expand(-1, -1, fx.numel())
    dx = (fx - i).expand(-1, fy.numel(), -1)
    radius_sq = dx ** 2 + dy ** 2
    nearest = radius_sq.argmin(0, keepdim=True)
    return radius_sq.gather(0, nearest)[0], dy.gather(0, nearest)[0], dx.gather(0, nearest)[0]
