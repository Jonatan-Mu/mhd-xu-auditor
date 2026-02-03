from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .io import save_snapshot_npz, save_metrics_csv
from .spectra import to_spectral, shell_spectrum_Ek, field_from_hat
from .metrics import compute_metrics, sanity_check_field
from .filters import make_kgrid
from .rendering import render_field_pdf, render_spectrum_pdf


def _auto_kind(arr: np.ndarray) -> str:
    return "spectral" if np.iscomplexobj(arr) else "physical"


def run_xu_diagnostics_pipeline(
    omega: np.ndarray,
    A: Optional[np.ndarray],
    outdir: str,
    Lx: float = 2*np.pi,
    Ly: float = 2*np.pi,
    input_kind_omega: str = "auto",
    input_kind_A: Optional[str] = "auto",
    dpi: int = 300,
    cmap_omega: str = "viridis",
    cmap_A: str = "inferno",
    vlines: Sequence[float] = (21.0, 28.0),
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """
    Load arrays → compute shell spectrum E(k) → compute metrics (BI, tailE, rho)
    → export vector PDFs + CSV + snapshot npz.
    Handles omega-only inputs (A=None) without crashing.

    Returns dict with paths.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    spec_dir = out / "spectra"
    img_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    md = {} if metadata is None else dict(metadata)

    # Decide kinds
    if input_kind_omega == "auto":
        input_kind_omega = _auto_kind(omega)
    if A is not None and (input_kind_A == "auto"):
        input_kind_A = _auto_kind(A)

    # Sanity on raw arrays
    sanity_check_field(omega, "omega")
    if A is not None:
        sanity_check_field(A, "A")

    # Convert to spectral
    omega_hat = to_spectral(omega, input_kind_omega)
    A_hat = None if A is None else to_spectral(A, input_kind_A or "auto")

    # More safety: if spectral arrays are all tiny, likely misinterpreted
    sanity_check_field(omega_hat, "omega_hat")

    N = omega_hat.shape[0]
    if omega_hat.ndim != 2 or omega_hat.shape[0] != omega_hat.shape[1]:
        raise ValueError(f"omega must be NxN 2D array; got {omega_hat.shape}")

    # Compute physical omega for plots (robust to kind)
    omega_phys = omega if input_kind_omega == "physical" else field_from_hat(omega_hat)
    A_phys = None
    if A_hat is not None:
        A_phys = A if (input_kind_A == "physical") else field_from_hat(A_hat)

    # Spectrum
    kk, Ek = shell_spectrum_Ek(omega_hat, A_hat, Lx=Lx, Ly=Ly)

    # Choose cutoff for metrics:
    # Use last vline if provided, else axis-nyquist*0.9.
    if len(vlines) > 0:
        kcut = float(vlines[-1])
    else:
        kg = make_kgrid(N, Lx, Ly)
        k_nyq_axis = float(np.max(np.abs(kg.kx)))
        kcut = 0.9 * k_nyq_axis

    # Metrics
    m = compute_metrics(kk=kk, Ek=Ek, omega_hat=omega_hat, Lx=Lx, Ly=Ly, kcut=kcut)

    # Filenames (time metadata if present)
    t = md.get("t", md.get("time", None))
    t_tag = "unknown"
    if t is not None:
        try:
            t_tag = f"{float(t):.6f}".replace(".", "p")
        except Exception:
            t_tag = "unknown"

    # Save snapshot
    snap_path = out / f"snapshot_t{t_tag}.npz"
    save_snapshot_npz(snap_path, omega=omega, A=A, metadata=md)

    # Render fields (vector PDFs)
    omega_pdf = img_dir / f"omega_t{t_tag}.pdf"
    omega_png = img_dir / f"omega_t{t_tag}.png"
    render_field_pdf(omega_phys, str(omega_pdf), title=r"$\omega(x,y)$", Lx=Lx, Ly=Ly,
                     cmap=cmap_omega, dpi=dpi, out_png=str(omega_png))

    A_pdf = None
    A_png = None
    if A_phys is not None:
        A_pdf = img_dir / f"A_t{t_tag}.pdf"
        A_png = img_dir / f"A_t{t_tag}.png"
        render_field_pdf(A_phys, str(A_pdf), title=r"$A(x,y)$", Lx=Lx, Ly=Ly,
                         cmap=cmap_A, dpi=dpi, out_png=str(A_png))

    # Render spectrum
    spec_pdf = spec_dir / f"spectrum_t{t_tag}.pdf"
    spec_png = spec_dir / f"spectrum_t{t_tag}.png"
    render_spectrum_pdf(
        spectra={"E(k)": (kk, Ek)},
        out_pdf=str(spec_pdf),
        title=r"Shell spectrum $E(k)$",
        vlines=vlines,
        dpi=dpi,
        out_png=str(spec_png),
    )

    # CSV metrics
    metrics = {
        "t": float(t) if t is not None else np.nan,
        "N": int(N),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "kcut": float(m.kcut),
        "BI": float(m.BI),
        "tailE": float(m.tailE),
        "rho": float(m.rho),
        "edgeE_mean": float(m.edgeE),
        "midE_mean": float(m.midE),
        "omega_kind": str(input_kind_omega),
        "A_present": bool(A_hat is not None),
    }
    # Merge safe metadata scalars (optional)
    for k in ["status", "input_file"]:
        if k in md:
            metrics[k] = md[k]

    csv_path = out / "metrics.csv"
    save_metrics_csv(csv_path, metrics)

    return {
        "snapshot_npz": str(snap_path),
        "metrics_csv": str(csv_path),
        "omega_pdf": str(omega_pdf),
        "omega_png": str(omega_png),
        "A_pdf": str(A_pdf) if A_pdf is not None else None,
        "A_png": str(A_png) if A_png is not None else None,
        "spectrum_pdf": str(spec_pdf),
        "spectrum_png": str(spec_png),
    }
