import math
from typing import Tuple

import torch

def pixel_distribution(img: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
    """Counts per value 0-255 (index, all, then each channel) and summary stats of a (height, width, 1 or 3) uint8 image."""
    height, width, channels = img.shape
    values = img.long()
    pixel_count = torch.stack([
        torch.arange(256, device=img.device),
        torch.bincount(values.flatten(), minlength=256),
        *(torch.bincount(values[..., c].flatten(), minlength=256) for c in range(channels)),
    ])

    counts = pixel_count[1]
    occupied = counts.nonzero().flatten()
    pixels = height * width
    median = (counts.cumsum(0) > counts.sum() / 2).int().argmax().item()
    nonzero = [occupied[0].item(), occupied[-1].item()]
    empty_bins = (counts == 0).sum().item()
    unique_colors = torch.unique(img.reshape(pixels, channels), dim=0).shape[0]
    least_frequent = occupied[counts[occupied].argmin()].item()
    level = img.float()

    return pixel_count, (pixels, img.min().item(), img.max().item(), least_frequent, counts.argmax().item(), round(level.mean().item(), 2), round(level.std(correction=0).item(), 2), median, nonzero, empty_bins, unique_colors)

def principal_components(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA of (..., c) samples: mean, variances and unit axes (columns) largest first, and each sample's (..., c) projection."""
    pixels = img.reshape(-1, img.shape[-1])
    mean = pixels.mean(0)
    variance, axes = torch.linalg.eigh(torch.cov(pixels.T))
    variance, axes = variance.flip(0), axes.flip(1)
    # eigh signs are arbitrary: make each axis's largest loading positive so projections are reproducible
    axes = axes * axes.gather(0, axes.abs().argmax(0, keepdim=True)).sign()
    return mean, variance, axes, ((pixels - mean) @ axes).reshape(img.shape)

def independent_components(z: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Symmetric FastICA with g = tanh (Hyvärinen 1999) of sphered (n, d) data: the orthogonal (d, d) unmixing, one component per row."""
    def decorrelate(w: torch.Tensor) -> torch.Tensor:
        # (W Wᵀ)^-1/2 W, the nearest orthogonal matrix: every component is updated in parallel, none favoured (Hyvärinen & Oja 2000 §6.2)
        s, u = torch.linalg.eigh(w @ w.T)
        return (u / s.sqrt()) @ u.T @ w

    w = torch.linalg.qr(torch.randn(z.shape[1], z.shape[1], generator=generator, device=z.device)).Q
    # Fixed point up to sign: each row's cosine with its predecessor within sqrt(eps) of 1, MINPACK's default tolerance
    tolerance = math.sqrt(torch.finfo(z.dtype).eps)
    while True:
        y = torch.tanh(z @ w.T)
        update = decorrelate(y.T @ z / len(z) - (1 - y ** 2).mean(0)[:, None] * w)
        converged = 1 - (update * w).sum(1).abs().min().item() < tolerance
        w = update
        if converged: return w

def sparse_coding(z: torch.Tensor, atoms: int, noise: float, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Olshausen & Field (1997) MAP sparse coding of whitened (n, d) data, noise per dimension: (d, atoms) dictionary, (n, atoms) codes."""
    d = z.shape[1]
    # Laplace scale b by moment matching: unit-variance z carries signal energy d(1 - noise) = atoms · 2b² over unit-norm atoms
    scale = math.sqrt(d * (1 - noise) / (2 * atoms))
    # MAP coefficients solve the lasso ½‖z - Φa‖² + (noise / b)‖a‖₁; dividing by noise gives the negative log posterior in nats
    penalty = noise / scale
    nats = lambda codes, dictionary: ((0.5 * ((z - codes @ dictionary.T) ** 2).sum() + penalty * codes.abs().sum()) / noise).item()
    dictionary = torch.randn(d, atoms, generator=generator, device=z.device)
    dictionary /= dictionary.norm(dim=0)
    codes = z.new_zeros(len(z), atoms)
    previous = math.inf
    while True:
        # FISTA (Beck & Teboulle 2009) warm-started from the last codes, step 1 / ‖Φ‖₂², the gradient's Lipschitz constant
        gram, target = dictionary.T @ dictionary, z @ dictionary
        step = 1 / torch.linalg.eigvalsh(gram)[-1]
        search, momentum, inner = codes, 1.0, nats(codes, dictionary)
        while True:
            update = search - step * (search @ gram - target)
            update = update.sign() * (update.abs() - step * penalty).clamp(min=0)
            accelerated = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
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
        for j in energy.diag().nonzero().flatten().tolist():
            atom = dictionary[:, j] + (correlation[:, j] - dictionary @ energy[:, j]) / energy[j, j]
            dictionary[:, j] = atom / atom.norm().clamp(min=1)
