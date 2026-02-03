from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import h5py
import pandas as pd


@dataclass
class Snapshot:
    omega: np.ndarray
    A: Optional[np.ndarray]
    metadata: Dict[str, Any]


def save_snapshot_npz(path: Union[str, Path], omega: np.ndarray, A: Optional[np.ndarray] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
    path = str(path)
    md = {} if metadata is None else dict(metadata)
    # np.savez_compressed only stores array-like; store metadata as a JSON-ish string dict
    np.savez_compressed(path, omega=omega, A=A if A is not None else np.array([], dtype=np.float64), metadata=np.array([repr(md)]))
    return path


def load_snapshot(path: Union[str, Path]) -> Snapshot:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Snapshot not found: {p}")

    suffix = p.suffix.lower()

    if suffix == ".npy":
        omega = np.load(p, allow_pickle=False)
        return Snapshot(omega=omega, A=None, metadata={"input_file": str(p), "format": "npy"})

    if suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        keys = set(z.files)

        def pick(keys_pref):
            for k in keys_pref:
                if k in keys:
                    return k
            return None

        k_omega = pick(["omega", "zeta", "w", "vorticity", "omega_hat", "zeta_hat"])
        if k_omega is None:
            # fallback: take first ndarray key
            arr_keys = [k for k in keys if isinstance(z[k], np.ndarray)]
            if len(arr_keys) == 1:
                k_omega = arr_keys[0]
            else:
                raise ValueError(f"Could not detect omega in NPZ keys={sorted(keys)}")

        omega = z[k_omega]

        k_A = pick(["A", "a", "potential", "A_hat"])
        A = z[k_A] if k_A is not None else None

        md = {"input_file": str(p), "format": "npz", "keys": sorted(keys), "omega_key": k_omega}
        if k_A is not None:
            md["A_key"] = k_A

        # metadata field (optional)
        if "metadata" in keys:
            try:
                md_str = z["metadata"]
                if isinstance(md_str, np.ndarray) and md_str.size > 0:
                    md.update(eval(str(md_str.flat[0])))
            except Exception:
                pass

        # treat empty A as None
        if isinstance(A, np.ndarray) and A.size == 0:
            A = None

        return Snapshot(omega=omega, A=A, metadata=md)

    if suffix in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            if "omega" not in f:
                raise ValueError("HDF5 must contain dataset 'omega'.")
            omega = f["omega"][()]
            A = f["A"][()] if "A" in f else None
            md = {"input_file": str(p), "format": suffix}
            if "metadata" in f:
                # stored as attributes or dataset
                try:
                    md.update(dict(f["metadata"].attrs))
                except Exception:
                    pass
        return Snapshot(omega=omega, A=A, metadata=md)

    raise ValueError(f"Unsupported snapshot format: {suffix}. Use .npy, .npz, .h5/.hdf5")


def save_metrics_csv(path: Union[str, Path], metrics: Dict[str, Any]) -> str:
    path = str(path)
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    return path


def save_snapshot_hdf5(path: Union[str, Path], omega: np.ndarray, A: Optional[np.ndarray] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    md = {} if metadata is None else dict(metadata)
    with h5py.File(p, "w") as f:
        f.create_dataset("omega", data=omega)
        if A is not None:
            f.create_dataset("A", data=A)
        g = f.create_group("metadata")
        for k, v in md.items():
            try:
                g.attrs[str(k)] = str(v)
            except Exception:
                pass
    return str(p)
