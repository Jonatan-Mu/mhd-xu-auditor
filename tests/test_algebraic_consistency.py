import numpy as np

from mhd_xu_auditor.spectra import fft2n, ifft2n
from mhd_xu_auditor.xu_convolution import enforce_conj_sym


def test_fft_roundtrip_and_parseval_scaling():
    """FFT normalization convention should be self-consistent.

    In this project we use:
      u_hat = fft2(u) / N
      u     = ifft2(u_hat) * N
    where N = u.size.

    With this normalization, discrete Parseval reads:
      sum|u|^2 = N * sum|u_hat|^2.
    """
    rng = np.random.default_rng(0)
    u = rng.standard_normal((64, 64))

    N = u.size
    u_hat = fft2n(u)
    u_rec = ifft2n(u_hat).real

    rel = np.linalg.norm(u_rec - u) / np.linalg.norm(u)
    assert rel < 1e-13

    e_phys = float(np.sum(np.abs(u) ** 2))
    e_spec = float(np.sum(np.abs(u_hat) ** 2))
    assert abs(e_phys - N * e_spec) / (e_phys + 1e-300) < 1e-13


def test_conjugate_symmetry_produces_real_ifft():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))

    z_sym = enforce_conj_sym(z)
    u = ifft2n(z_sym)

    # Imaginary part should be numerical noise.
    assert np.max(np.abs(u.imag)) < 1e-12
