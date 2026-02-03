from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .filters import make_kgrid


@dataclass
class MetricResult:
    BI: float
    tailE: float
    rho: float
    edgeE: float
    midE: float
    kcut: float


# -----------------------------
# Basic helpers
# -----------------------------
_TINY = np.finfo(np.float64).tiny  # ~2e-308
_EPS  = 1e-300                     # stable denom guard


def sanity_check_field(arr: np.ndarray, name: str) -> None:
    """Hard fail only for NaN/Inf. Do NOT fail for small magnitudes."""
    if arr is None:
        return
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf.")


def _as_hat(field_or_hat: np.ndarray) -> np.ndarray:
    """
    Accept either:
      - physical field (real ndarray) -> return normalized fft2(field)/N
      - spectral field (complex ndarray) -> return as complex128
    Convention: u_hat = fft2(u)/u.size
    """
    arr = np.asarray(field_or_hat)
    sanity_check_field(arr, "omega/A input")

    # If it's complex with nontrivial imag part, assume spectral.
    if np.iscomplexobj(arr) and np.nanmax(np.abs(arr.imag)) > 0:
        return arr.astype(np.complex128, copy=False)

    # Otherwise assume physical, FFT-normalize
    arr_real = np.asarray(arr, dtype=np.float64)
    return (np.fft.fft2(arr_real) / arr_real.size).astype(np.complex128, copy=False)


def _band_mean(kk: np.ndarray, Ek: np.ndarray, a: float, b: float) -> float:
    m = (kk >= a) & (kk <= b)
    if np.count_nonzero(m) == 0:
        return 0.0
    # Ek may contain exact zeros -> fine
    return float(np.mean(Ek[m]))


def blocking_index(kk: np.ndarray, Ek: np.ndarray, kcut: float) -> Tuple[float, float, float]:
    """
    BI = mean(Ek in [0.9 kcut, kcut]) / mean(Ek in [0.55 kcut, 0.65 kcut]).
    Returns (BI, edge_mean, mid_mean).
    """
    edge = _band_mean(kk, Ek, 0.90 * kcut, 1.00 * kcut)
    mid  = _band_mean(kk, Ek, 0.55 * kcut, 0.65 * kcut)
    BI = edge / (mid + _EPS)
    return float(BI), float(edge), float(mid)


def tail_energy_above(kk: np.ndarray, Ek: np.ndarray, kcut: float) -> float:
    m = kk >= kcut
    return float(np.sum(Ek[m]))


def regularity_ratio_rho(omega_or_hat: np.ndarray, Lx: float, Ly: float) -> float:
    """
    Snapshot regularity ratio (high-k sensitive, no logs):
        rho = (Σ k^4 |ω̂|^2) / (Σ k^2 |ω̂|^2 + eps)

    Accepts omega in physical space OR omega_hat in spectral space.
    """
    omega_hat = _as_hat(omega_or_hat)

    # (Assumes square N×N grid as in the rest of the repo.)
    N = omega_hat.shape[0]
    if omega_hat.ndim != 2 or omega_hat.shape[0] != omega_hat.shape[1]:
        raise ValueError(f"regularity_ratio_rho expects square 2D array; got shape={omega_hat.shape}")

    kg = make_kgrid(N, Lx, Ly)
    k2 = np.asarray(kg.k2, dtype=np.longdouble)

    # |omega_hat|^2
    w2 = (omega_hat.real.astype(np.longdouble)**2 + omega_hat.imag.astype(np.longdouble)**2)

    num = np.sum((k2**2) * w2)
    den = np.sum(k2 * w2)

    # If den is ~0 (e.g. omega ≈ 0), define rho = 0 (safe, not NaN)
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    if den <= _TINY:
        return 0.0
    return float(num / (den + _EPS))


def compute_metrics(
    kk: np.ndarray,
    Ek: np.ndarray,
    omega_or_hat: np.ndarray,
    Lx: float,
    Ly: float,
    kcut: float,
) -> MetricResult:
    kk = np.asarray(kk, dtype=np.float64)
    Ek = np.asarray(Ek, dtype=np.float64)

    sanity_check_field(kk, "kk")
    sanity_check_field(Ek, "Ek")

    if kcut <= 0 or not np.isfinite(kcut):
        raise ValueError(f"kcut must be positive finite. Got kcut={kcut}")

    # Guard: sometimes tiny negative Ek can appear from numerical noise; clip it
    if np.any(Ek < 0):
        Ek = np.maximum(Ek, 0.0)

    # If spectrum is effectively empty, don’t emit NaNs.
    Etot = float(np.sum(Ek))
    rho = regularity_ratio_rho(omega_or_hat, Lx, Ly)

    if (not np.isfinite(Etot)) or Etot <= _TINY:
        return MetricResult(BI=0.0, tailE=0.0, rho=float(rho), edgeE=0.0, midE=0.0, kcut=float(kcut))

    BI, edge, mid = blocking_index(kk, Ek, kcut)
    tailE = tail_energy_above(kk, Ek, kcut)

    return MetricResult(BI=float(BI), tailE=float(tailE), rho=float(rho),
                        edgeE=float(edge), midE=float(mid), kcut=float(kcut))
