#!/usr/bin/env python3

import torch
import numpy as np

def device_mapper() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def host(x: torch.Tensor) -> np.ndarray: return x.cpu().numpy()