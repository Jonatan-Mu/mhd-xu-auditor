from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile
import numpy as np

from .io import load_snapshot
from .pipeline import run_xu_diagnostics_pipeline


def zip_folder(folder: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            root_p = Path(root)
            for f in files:
                fp = root_p / f
                rel = fp.relative_to(folder)
                zf.write(fp, arcname=str(rel))


def main():
    p = argparse.ArgumentParser(prog="xu-auditor")
    p.add_argument("--input", required=True, help="Path to .npy/.npz/.h5 snapshot (omega required; A optional).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--Lx", type=float, default=2*np.pi)
    p.add_argument("--Ly", type=float, default=2*np.pi)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap_omega", default="viridis")
    p.add_argument("--cmap_A", default="inferno")
    p.add_argument("--vlines", nargs="*", type=float, default=[21.0, 28.0])
    p.add_argument("--mode", choices=["auto", "physical", "spectral"], default="auto",
                   help="Interpret arrays: auto (complex->spectral), or force physical/spectral.")
    args = p.parse_args()

    snap = load_snapshot(args.input)
    omega = snap.omega
    A = snap.A
    md = snap.metadata

    def decide(arr):
        if arr is None:
            return None
        if args.mode == "auto":
            return "spectral" if np.iscomplexobj(arr) else "physical"
        return args.mode

    res = run_xu_diagnostics_pipeline(
        omega=omega,
        A=A,
        outdir=args.outdir,
        Lx=args.Lx,
        Ly=args.Ly,
        input_kind_omega=decide(omega) or "physical",
        input_kind_A=decide(A) if A is not None else None,
        dpi=args.dpi,
        cmap_omega=args.cmap_omega,
        cmap_A=args.cmap_A,
        vlines=tuple(args.vlines),
        metadata=md,
    )

    outdir = Path(args.outdir)
    zip_path = outdir / "Xu_Diagnostic_Results.zip"
    zip_folder(outdir, zip_path)
    print("[OK] Output:", outdir)
    print("[OK] ZIP:", zip_path)
    for k, v in res.items():
        if v:
            print(f"[OK] {k}: {v}")


if __name__ == "__main__":
    main()
