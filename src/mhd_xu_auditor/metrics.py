from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .filters import make_kgrid
from .spectra import omega_to_streamfunc_hat


@dataclass
class MetricResult:
    BI: float
    tailE: float
    rho: float
    edgeE: float
    midE: float
    kcut: float


def _band_mean(kk: np.ndarray, Ek: np.ndarray, a: float, b: float) -> float:
    m = (kk >= a) & (kk <= b)
    if np.count_nonzero(m) == 0:
        return 0.0
    return float(np.mean(Ek[m]))


def blocking_index(kk: np.ndarray, Ek: np.ndarray, kcut: float) -> Tuple[float, float, float]:
    """
    BI = mean(Ek in [0.9 kcut, kcut]) / mean(Ek in [0.55 kcut, 0.65 kcut]).
    Returns (BI, edge_mean, mid_mean).
    """
    edge = _band_mean(kk, Ek, 0.90 * kcut, 1.00 * kcut)
    mid = _band_mean(kk, Ek, 0.55 * kcut, 0.65 * kcut)
    BI = edge / (mid + 1e-300)
    return float(BI), float(edge), float(mid)


def tail_energy_above(kk: np.ndarray, Ek: np.ndarray, kcut: float) -> float:
    m = kk >= kcut
    return float(np.sum(Ek[m]))


def regularity_ratio_rho(omega_hat: np.ndarray, Lx: float, Ly: float) -> float:
    """
    Snapshot regularity ratio:
        rho = (Σ k^4 |ω̂|^2) / (Σ k^2 |ω̂|^2 + eps).
    This is deliberately high-k sensitive and avoids logs.
    """
    N = omega_hat.shape[0]
    kg = make_kgrid(N, Lx, Ly)
    k2 = kg.k2
    w2 = np.abs(omega_hat) ** 2

    num = float(np.sum((k2 ** 2) * w2))
    den = float(np.sum(k2 * w2)) + 1e-300
    return num / den


def sanity_check_field(arr: np.ndarray, name: str, min_nonzero: float = 1e-30) -> None:
    if arr is None:
        return
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    maxabs = float(np.max(np.abs(arr)))
    if maxabs < min_nonzero:
        raise ValueError(
            f"{name} appears ~zero (max|.|={maxabs:.3e}). "
            "This usually means the array was misinterpreted (spectral vs physical) or empty."
        )


def compute_metrics(
    kk: np.ndarray,
    Ek: np.ndarray,
    omega_hat: np.ndarray,
    Lx: float,
    Ly: float,
    kcut: float
) -> MetricResult:
    sanity_check_field(Ek, "Ek")
    BI, edge, mid = blocking_index(kk, Ek, kcut)
    tailE = tail_energy_above(kk, Ek, kcut)
    rho = regularity_ratio_rho(omega_hat, Lx, Ly)
    return MetricResult(BI=BI, tailE=tailE, rho=rho, edgeE=edge, midE=mid, kcut=kcut)
