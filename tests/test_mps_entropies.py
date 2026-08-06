import numpy as np

from quasisymmetries.block2_qubit_benchmark import (
    load_qubit_mps_arrays,
    save_qubit_mps_arrays,
)
from quasisymmetries.fiedler import (
    fiedler_order_from_mps,
    qubit_mps_cut_entropies,
    qubit_mutual_information_matrix,
    qubit_mutual_information_matrix_mps,
)


def statevector_to_mps(state, n_qubits):
    state = np.asarray(state, dtype=complex).reshape(1, -1)
    tensors = []
    left_dim = 1
    for _site in range(n_qubits - 1):
        matrix = state.reshape(left_dim * 2, -1)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        right_dim = len(singular_values)
        tensors.append(u.reshape(left_dim, 2, right_dim))
        state = singular_values[:, None] * vh
        left_dim = right_dim
    tensors.append(state.reshape(left_dim, 2, 1))
    return tensors


def dense_cut_entropies(state, n_qubits):
    values = []
    for cut in range(1, n_qubits):
        singular_values = np.linalg.svd(
            state.reshape(1 << cut, -1), compute_uv=False
        )
        probabilities = singular_values**2
        probabilities = probabilities[probabilities > 1e-12]
        values.append(float(-np.sum(probabilities * np.log2(probabilities))))
    return np.asarray(values)


def test_mps_entropies_match_dense_random_state():
    rng = np.random.default_rng(18)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    tensors = statevector_to_mps(state, 4)

    expected_mi, expected_s1, expected_s2 = (
        qubit_mutual_information_matrix(state, n_qubits=4)
    )
    actual_mi, actual_s1, actual_s2 = (
        qubit_mutual_information_matrix_mps(tensors)
    )

    assert np.allclose(actual_mi, expected_mi, atol=1e-11)
    assert np.allclose(actual_s1, expected_s1, atol=1e-11)
    assert np.allclose(actual_s2, expected_s2, atol=1e-11)
    assert np.allclose(
        qubit_mps_cut_entropies(tensors),
        dense_cut_entropies(state, 4),
        atol=1e-11,
    )


def test_mps_fiedler_path_marks_mps_source():
    state = np.zeros(8)
    state[0] = state[-1] = 1 / np.sqrt(2)
    info = fiedler_order_from_mps(statevector_to_mps(state, 3))

    assert sorted(info["ordering"]) == [0, 1, 2]
    assert info["mutual_information_method"] == "mps"
    assert np.allclose(info["one_qubit_entropies"], 1.0)


def test_portable_mps_archive_round_trip(tmp_path):
    state = np.arange(8, dtype=float)
    state /= np.linalg.norm(state)
    tensors = statevector_to_mps(state, 3)
    path = tmp_path / "reference_mps.npz"

    metadata = save_qubit_mps_arrays(tensors, path)
    restored = load_qubit_mps_arrays(path)

    assert metadata["n_sites"] == 3
    assert len(restored) == len(tensors)
    for expected, actual in zip(tensors, restored):
        assert np.array_equal(actual, expected)
