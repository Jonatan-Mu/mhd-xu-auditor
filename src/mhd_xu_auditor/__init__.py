"""MHD Xu-Auditor

Publication-grade diagnostics for 2D MHD snapshots.

The package is organized as a small library plus a CLI.

Key entrypoints
--------------

- :func:`mhd_xu_auditor.pipeline.run_xu_diagnostics_pipeline`
- :func:`mhd_xu_auditor.io.load_snapshot`
- :func:`mhd_xu_auditor.io.save_snapshot_npz`

"""

from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import run_xu_diagnostics_pipeline
from .io import load_snapshot, save_snapshot_npz

__all__ = [
    "__version__",
    "run_xu_diagnostics_pipeline",
    "load_snapshot",
    "save_snapshot_npz",
]
