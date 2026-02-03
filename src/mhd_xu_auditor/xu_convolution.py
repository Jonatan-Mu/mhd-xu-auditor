from __future__ import annotations

import numpy as np


def fft2n(u: np.ndarray) -> np.ndarray:
    return np.fft.fft2(u) / u.size


def ifft2n(uh: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(uh) * uh.size


def enforce_conj_sym(F: np.ndarray) -> np.ndarray:
    Ny, Nx = F.shape
    iy_neg = (-np.arange(Ny)) % Ny
    ix_neg = (-np.arange(Nx)) % Nx
    partner = np.conj(F[np.ix_(iy_neg, ix_neg)])
    G = 0.5 * (F + partner)

    self_y = [0] + ([Ny // 2] if Ny % 2 == 0 else [])
    self_x = [0] + ([Nx // 2] if Nx % 2 == 0 else [])
    for iy in self_y:
        for ix in self_x:
            G[iy, ix] = np.real(G[iy, ix]) + 0j
    return G


def apply_filter(F: np.ndarray, filt: np.ndarray) -> np.ndarray:
    return enforce_conj_sym(F * filt)


def make_k(N: int, L: float) -> np.ndarray:
    k_int = np.fft.fftfreq(N) * N
    return (2 * np.pi / L) * k_int


def make_kgrid(N: int, Lx: float, Ly: float):
    kx = make_k(N, Lx)
    ky = make_k(N, Ly)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k2 = KX**2 + KY**2
    inv_k2 = np.zeros_like(k2)
    inv_k2[k2 != 0] = 1.0 / k2[k2 != 0]
    return KX, KY, k2, inv_k2, kx, ky


def forcing_pseudospectral(zeta_hat: np.ndarray, A_hat: np.ndarray, Lx: float, Ly: float, filt: np.ndarray):
    """
    MHD in vorticity/potential:
      dζ/dt = -J(phi,ζ) + J(A,j)
      dA/dt = -J(phi,A)
    with ζ = -Δphi, j=-ΔA.
    """
    N = zeta_hat.shape[0]
    KX, KY, k2, invk, *_ = make_kgrid(N, Lx, Ly)
    KX = KX.astype(np.complex128)
    KY = KY.astype(np.complex128)
    k2c = k2.astype(np.complex128)

    zeta_hat = apply_filter(zeta_hat, filt)
    A_hat = apply_filter(A_hat, filt)

    phi_hat = -(zeta_hat * invk)
    phi_hat[k2 == 0] = 0.0 + 0.0j
    j_hat = -(k2c) * A_hat

    phi_x = ifft2n(1j * KX * phi_hat).real
    phi_y = ifft2n(1j * KY * phi_hat).real
    z_x = ifft2n(1j * KX * zeta_hat).real
    z_y = ifft2n(1j * KY * zeta_hat).real

    A_x = ifft2n(1j * KX * A_hat).real
    A_y = ifft2n(1j * KY * A_hat).real
    j_x = ifft2n(1j * KX * j_hat).real
    j_y = ifft2n(1j * KY * j_hat).real

    J_phi_z = phi_x * z_y - phi_y * z_x
    J_A_j = A_x * j_y - A_y * j_x
    J_phi_A = phi_x * A_y - phi_y * A_x

    Nzeta = -J_phi_z + J_A_j
    NA = -J_phi_A

    Nz_hat = fft2n(Nzeta).astype(np.complex128)
    NA_hat = fft2n(NA).astype(np.complex128)
    return apply_filter(Nz_hat, filt), apply_filter(NA_hat, filt)


def forcing_xu_convolution_fast(zeta_hat: np.ndarray, A_hat: np.ndarray, Lx: float, Ly: float, filt: np.ndarray):
    """
    Xu cyclic convolution implementation via roll:
      q = k - p  ==> array_q[k] = array_orig[k - p] = roll(+p)
    """
    N = zeta_hat.shape[0]
    KX, KY, k2, invk, kx, ky = make_kgrid(N, Lx, Ly)
    KX = KX.astype(np.complex128)
    KY = KY.astype(np.complex128)
    k2c = k2.astype(np.complex128)

    zeta_hat = apply_filter(zeta_hat, filt)
    A_hat = apply_filter(A_hat, filt)

    phi_hat = -(zeta_hat * invk)
    phi_hat[k2 == 0] = 0.0 + 0.0j
    j_hat = -(k2c) * A_hat

    Nz = np.zeros((N, N), dtype=np.complex128)
    NA = np.zeros((N, N), dtype=np.complex128)

    for py in range(N):
        p_y = float(ky[py])
        for px in range(N):
            p_x = float(kx[px])
            phi_p = phi_hat[py, px]
            A_p = A_hat[py, px]
            if phi_p == 0j and A_p == 0j:
                continue

            cross = (p_x * KY - p_y * KX)
            z_q = np.roll(zeta_hat, shift=(py, px), axis=(0, 1))
            j_q = np.roll(j_hat, shift=(py, px), axis=(0, 1))
            A_q = np.roll(A_hat, shift=(py, px), axis=(0, 1))

            Nz += cross * (phi_p * z_q - A_p * j_q)
            NA += cross * (phi_p * A_q)

    return apply_filter(Nz, filt), apply_filter(NA, filt)
