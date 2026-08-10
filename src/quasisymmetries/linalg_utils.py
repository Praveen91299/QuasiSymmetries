"""Small numerical-linear-algebra helpers shared by MPS code paths."""

from __future__ import annotations

import numpy as np


def robust_svd(
    matrix,
    *,
    full_matrices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Compute an SVD and retry with LAPACK ``gesvd`` if NumPy fails.

    Parameters
    ----------
    matrix
        Finite two-dimensional real or complex array to factorize.
    full_matrices
        Whether to return full-sized unitary factors. MPS splits normally use
        the default ``False`` economy-sized factorization.

    Returns
    -------
    u, singular_values, vh, used_fallback
        SVD factors satisfying ``matrix = u @ diag(singular_values) @ vh``
        up to floating-point precision, plus a flag indicating whether the
        rescaled SciPy/LAPACK ``gesvd`` retry was required.

    Notes
    -----
    NumPy normally uses the faster divide-and-conquer ``gesdd`` driver. Very
    ill-conditioned MPS tensors can occasionally make that driver report
    nonconvergence even though their entries are finite. Rescaling and using
    the more conservative ``gesvd`` driver changes the factorization method,
    not the requested MPS truncation.
    """
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("cannot factorize a matrix containing non-finite values")
    try:
        u, singular_values, vh = np.linalg.svd(
            array, full_matrices=bool(full_matrices)
        )
        return u, singular_values, vh, False
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd

        scale = float(np.max(np.abs(array))) if array.size else 0.0
        if scale == 0.0:
            # A zero matrix should never reach the fallback in practice, but
            # SciPy still provides a deterministic factorization if it does.
            scale = 1.0
        u, singular_values, vh = scipy_svd(
            array / scale,
            full_matrices=bool(full_matrices),
            compute_uv=True,
            check_finite=True,
            lapack_driver="gesvd",
        )
        return u, singular_values * scale, vh, True
