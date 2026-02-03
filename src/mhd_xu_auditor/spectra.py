from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
from .filters import make_kgrid


def fft2n(u: np.ndarray) -> np.ndarray:
    return np.fft.fft2(u) / u.size


def ifft2n(uh: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(uh) * uh.size


def to_spectral(field: np.ndarray, kind: str) -> np.ndarray:
    """
    kind: 'physical' or 'spectral'
    physical (real) -> fft2n
    spectral -> returned as complex array
    """
    if kind == "spectral":
        return np.asarray(field, dtype=np.complex128)
    if kind == "physical":
        return np.asarray(fft2n(np.asarray(field, dtype=np.float64)), dtype=np.complex128)
    raise ValueError(f"Unknown kind={kind}")


def omega_to_streamfunc_hat(omega_hat: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """phi_hat from omega_hat via -k^2 phi_hat = omega_hat."""
    inv_k2 = np.zeros_like(k2, dtype=np.float64)
    inv_k2[k2 != 0] = 1.0 / k2[k2 != 0]
    phi_hat = -(omega_hat * inv_k2)
    phi_hat = np.asarray(phi_hat, dtype=np.complex128)
    phi_hat[k2 == 0] = 0.0 + 0.0j
    return phi_hat


def shell_spectrum_Ek(
    omega_hat: np.ndarray,
    A_hat: Optional[np.ndarray],
    Lx: float,
    Ly: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shell spectrum E(k) = 1/2 k^2 (|phi_hat|^2 + |A_hat|^2) * (area/N^2) binned into shells.
    If A_hat is None: uses kinetic only 1/2 k^2 |phi_hat|^2.
    """
    N = omega_hat.shape[0]
    area = Lx * Ly
    NxNy = N * N

    kg = make_kgrid(N, Lx, Ly)
    k2 = kg.k2
    k0 = kg.k0

    phi_hat = omega_to_streamfunc_hat(omega_hat, k2)

    Emode = 0.5 * (area / NxNy) * k2 * (np.abs(phi_hat) ** 2)
    if A_hat is not None:
        Emode = Emode + 0.5 * (area / NxNy) * k2 * (np.abs(A_hat) ** 2)

    shell = np.rint(kg.kmag / k0).astype(int)
    s = shell.ravel()
    e = Emode.ravel()
    Ek = np.bincount(s, weights=e, minlength=s.max() + 1)
    kk = np.arange(len(Ek)) * k0

    # drop k=0
    return kk[1:], Ek[1:]


def field_from_hat(field_hat: np.ndarray) -> np.ndarray:
    return np.real(ifft2n(field_hat))


def safe_positive(arr: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return np.maximum(arr, eps)
