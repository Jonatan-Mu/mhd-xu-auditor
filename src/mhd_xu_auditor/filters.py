from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class KGrid:
    KX: np.ndarray
    KY: np.ndarray
    k2: np.ndarray
    kmag: np.ndarray
    k0: float
    kx: np.ndarray
    ky: np.ndarray


def make_kgrid(N: int, Lx: float, Ly: float) -> KGrid:
    kx_int = np.fft.fftfreq(N) * N
    ky_int = np.fft.fftfreq(N) * N
    kx = (2 * np.pi / Lx) * kx_int
    ky = (2 * np.pi / Ly) * ky_int
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k2 = KX**2 + KY**2
    kmag = np.sqrt(k2)
    k0 = min(2 * np.pi / Lx, 2 * np.pi / Ly)
    return KGrid(KX=KX, KY=KY, k2=k2, kmag=kmag, k0=k0, kx=kx, ky=ky)


def hard_mask_rect(N: int, kc_frac: float) -> np.ndarray:
    """Rectangular hard mask using axis cutoffs (comparable to 2/3 or 0.9)."""
    kx_int = np.fft.fftfreq(N) * N
    ky_int = np.fft.fftfreq(N) * N
    kcx = int(np.floor(kc_frac * (N // 2)))
    kcy = int(np.floor(kc_frac * (N // 2)))
    mx = (np.abs(kx_int) <= kcx)
    my = (np.abs(ky_int) <= kcy)
    return (my[:, None] & mx[None, :]).astype(np.float64)


def smooth_rolloff_axis(N: int, Lx: float, Ly: float, kc_frac: float = 0.9, p: int = 8, alpha: float = 12.0) -> np.ndarray:
    """
    Smooth radial roll-off with cutoff defined from axis Nyquist:
        k_nyq_axis = max |kx|
        kc = kc_frac * k_nyq_axis
    Damping goes from kc to axis nyquist (clipped).
    """
    kg = make_kgrid(N, Lx, Ly)
    kmag = kg.kmag
    k_nyq_axis = np.max(np.abs(kg.kx))
    kc = kc_frac * k_nyq_axis

    filt = np.ones_like(kmag, dtype=np.float64)
    m = kmag > kc
    x = (kmag[m] - kc) / (k_nyq_axis - kc + 1e-30)
    x = np.clip(x, 0.0, 1.0)
    filt[m] = np.exp(-alpha * (x**p))
    return filt
