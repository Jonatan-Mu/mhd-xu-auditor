import numpy as np

from mhd_xu_auditor.io import save_snapshot_npz, load_snapshot


def test_npz_roundtrip(tmp_path):
    omega = np.arange(16, dtype=np.float64).reshape(4, 4)
    A = (omega * 0.5).astype(np.float64)

    f = tmp_path / "snap.npz"
    save_snapshot_npz(f, omega=omega, A=A, metadata={"Lx": 2*np.pi, "Ly": 2*np.pi})

    snap = load_snapshot(f)
    assert np.allclose(snap.omega, omega)
    assert np.allclose(snap.A, A)
    assert float(snap.metadata["Lx"]) == float(2*np.pi)
