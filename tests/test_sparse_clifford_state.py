import numpy as np

from quasisymmetries.clifford_symmetry_optimized import Clifford
from quasisymmetries.state_utils import SparseQubitState


def test_sparse_clifford_state_matches_dense_path():
    rng = np.random.default_rng(729)
    dense = rng.normal(size=16) + 1j * rng.normal(size=16)
    dense /= np.linalg.norm(dense)
    sparse = SparseQubitState.from_dense(dense, threshold=0.2)
    sparse.normalize()
    clifford = Clifford(
        4,
        [
            ("H", 1),
            ("S", 3),
            ("CNOT", 1, 2),
            ("X", 0),
            ("Sdg", 2),
            ("H", 3),
        ],
        permutation=[2, 0, 3, 1],
    )

    actual = clifford.transform_sparse_state(sparse).to_dense()
    expected = clifford.transform_state(sparse.to_dense())

    assert np.allclose(actual, expected)
    assert np.isclose(np.linalg.norm(actual), 1.0)
