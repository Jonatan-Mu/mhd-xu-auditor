# MHD Xu-Auditor

[![Python Tests](https://github.com/Jonatan-Mu/mhd-xu-auditor/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Jonatan-Mu/mhd-xu-auditor/actions/workflows/main.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18467205.svg)](https://doi.org/10.5281/zenodo.18467205)

**MHD Xu-Auditor** is a *spectral fidelity sensor* for 2D incompressible MHD: it audits high-frequency content in simulation or experimental snapshots and reports whether the data are being corrupted by spectral blocking / aliasing artifacts.

The core idea is to complement (or replace) traditional pseudo-spectral (FFT-based) diagnostics with **Xu-style modal auditing**, which is less dependent on repeated physical↔spectral round trips and enables *publication-grade* regularity monitoring.

## What it produces
For each snapshot \((\omega, A)\), Xu-Auditor generates:
- **Shell spectrum** \(E(k)\) and a vector PDF figure.
- **Blocking Index (BI)**: a quantitative spectral-blocking indicator near a user-chosen cutoff.
- **Tail energy (tailE)**: energy residing above the cutoff (high-frequency tail).
- **Regularity ratio \(\rho\)**: a stability/regularity diagnostic computed from consistent norms (see `docs/technical_notes.md`).
- Paper-grade field renders (vector PDF, optional PNG).
- A `metrics.csv` suitable for batching and plotting.

## Quick start (CLI)
Install in a clean environment:
```bash
pip install -r requirements.txt
pip install -e .
```

Run the diagnostic pipeline on an external file (`.npy` or `.npz`):
```bash
xu-auditor diagnose --input /path/to/snapshot.npz --outdir ./output --Lx 6.283185 --Ly 6.283185
```

Supported input formats:
- **`.npz`** containing `omega` (or `w`, `vorticity`) and optionally `A`.
- **`.npy`** containing a single array interpreted as `omega` (vorticity-only mode).

If `A` is missing, the auditor will still compute \(E_u(k)\) from \(\phi\) reconstructed from \(\omega\), render \(\omega\), and export partial metrics.

## Quick start (Python)
```python
import numpy as np
from mhd_xu_auditor.pipeline import run_xu_diagnostics_pipeline

omega = np.load("data/example_omega_only.npy")
run_xu_diagnostics_pipeline(
    omega=omega,
    A=None,
    outdir="./output",
    Lx=2*np.pi, Ly=2*np.pi,
    kcut_mode=28,  # e.g. 0.9*(N/2)
)
```

## Key metrics (interpretation)
- **Blocking Index (BI)** compares mean spectral energy in an *edge band* just below the cutoff to a *mid band* well below it.
  - `BI ≈ 1` suggests a clean tail.
  - `BI >> 1` indicates **spectral blocking** (pile-up / hook near the cutoff).
- **tailE** is the cumulative spectral energy beyond the cutoff.
- **\(\rho\)** is a regularity ratio built from consistent norms; spikes or high total variation are typically a warning of high-frequency contamination.

## Reproducibility
- All I/O uses explicit metadata (`N`, `Lx`, `Ly`, `fft_norm`) when saving `.npz`.
- Figures are generated as **vector PDFs** for direct inclusion in manuscripts.
- `tests/` includes algebraic consistency checks (Parseval & normalization) intended to reproduce a \(\sim10^{-14}\) identity on standard hardware.

## Repository layout
```
mhd-xu-auditor/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── src/mhd_xu_auditor/
│   ├── __init__.py
│   ├── io.py
│   ├── filters.py
│   ├── spectra.py
│   ├── metrics.py
│   ├── rendering.py
│   ├── pipeline.py
│   ├── xu_convolution.py
│   └── cli.py
├── data/
│   ├── example_snapshot.npz
│   └── example_omega_only.npy
├── docs/
│   ├── technical_notes.md
│   └── figures/
└── tests/
    ├── test_algebraic_consistency.py
    └── test_io_roundtrip.py
```

## License
MIT. See `LICENSE`.

## Citation / DOI
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18467205.svg)](https://doi.org/10.5281/zenodo.18467205)

