from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict
import numpy as np
import matplotlib.pyplot as plt


def render_field_pdf(
    field: np.ndarray,
    out_pdf: str,
    title: str,
    Lx: float,
    Ly: float,
    cmap: str = "viridis",
    dpi: int = 300,
    out_png: Optional[str] = None,
) -> None:
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.5, 5.3))
    ax = plt.gca()
    im = ax.imshow(
        field,
        origin="lower",
        extent=(0, Lx, 0, Ly),
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(title, rotation=90)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    fig.tight_layout()

    # Vector PDF
    fig.savefig(out_pdf, format="pdf")

    # Optional raster PNG
    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, format="png")

    plt.close(fig)


def render_spectrum_pdf(
    spectra: Dict[str, tuple],
    out_pdf: str,
    title: str = "Shell spectrum",
    vlines: Sequence[float] = (),
    dpi: int = 300,
    out_png: Optional[str] = None,
) -> None:
    """
    spectra: dict name -> (kk, Ek)
    """
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 4.8))
    ax = plt.gca()

    for name, (kk, Ek) in spectra.items():
        ax.loglog(kk, Ek + 1e-300, label=name)

    for v in vlines:
        ax.axvline(v, linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_pdf, format="pdf")
    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, format="png")

    plt.close(fig)
