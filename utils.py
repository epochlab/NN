#!/usr/bin/env python3

import torch
import numpy as np

def device_mapper() -> torch.device:
    """Fastest available torch device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def host(x: torch.Tensor) -> np.ndarray:
    """Tensor as a NumPy array on the CPU: the boundary to matplotlib, pandas and pathtracer."""
    return x.cpu().numpy()
