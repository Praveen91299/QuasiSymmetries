import numpy as np

from quasisymmetries.block2_qubit_benchmark import (
    _compress_pyblock_mps_with_svd_fallback,
)


class _FakePyblockMPS:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.singular_values = None

    def compress(self, k, cutoff, left):
        _u, self.singular_values, _vh = np.linalg.svd(
            self.matrix, full_matrices=False
        )
        return 0.0


def test_compression_retries_failed_numpy_svd_with_gesvd(monkeypatch):
    original = np.linalg.svd
    calls = 0

    def fail_once(matrix, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise np.linalg.LinAlgError("synthetic nonconvergence")
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", fail_once)
    mps = _FakePyblockMPS([[1e-23, 0.0], [0.0, 0.95]])

    error, fallback_count = _compress_pyblock_mps_with_svd_fallback(
        mps, k=-1, cutoff=1e-13, left=True
    )

    assert error == 0.0
    assert fallback_count == 1
    np.testing.assert_allclose(mps.singular_values, [0.95, 1e-23])
