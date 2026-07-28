import numpy as np
from openfermion import QubitOperator, get_sparse_operator

from quasisymmetries.fiedler import (
    do_fiedler_reordering,
    fiedler_order_from_state,
    qubit_mutual_information_matrix,
    reorder_statevector_axes,
    reduced_density_matrix_statevector,
)
from quasisymmetries.metrics import get_entropies_at_cuts
from quasisymmetries.state_utils import (
    PauliActionMask,
    SparseQubitState,
    action_mask_to_pauli_mask,
    pauli_mask_to_action_mask,
)
from quasisymmetries.bs.utils import mask_to_qubit_operator, term_to_masks


def _sample_state(n_qubits=4):
    dense = np.zeros(1 << n_qubits, dtype=complex)
    dense[0b0000] = 0.35
    dense[0b0011] = -0.2j
    dense[0b1010] = 0.4 + 0.1j
    dense[0b1111] = -0.3
    dense /= np.linalg.norm(dense)
    return dense


def test_sparse_qubit_state_round_trip_and_normalization():
    dense = _sample_state()

    sparse = SparseQubitState.from_dense(dense)

    assert sparse.n_qubits == 4
    assert sparse.nnz == 4
    assert np.allclose(sparse.to_dense(), dense)
    assert np.isclose(sparse.norm(), 1.0)

    unnormalized = SparseQubitState.from_dict(
        {0b0000: 2.0, 0b1111: 2.0j},
        n_qubits=4,
    )
    normalized = unnormalized.normalized()

    assert np.isclose(normalized.norm(), 1.0)
    assert np.isclose(unnormalized.norm(), np.sqrt(8.0))


def test_sparse_qubit_state_qubit_operator_expectation_and_apply():
    dense = _sample_state()
    sparse = SparseQubitState.from_dense(dense)
    op = (
        0.7 * QubitOperator("X0")
        - 0.3j * QubitOperator("Y1 Z3")
        + 1.2 * QubitOperator("Z0 Z2")
        - 0.4 * QubitOperator(())
    )
    matrix = get_sparse_operator(op, n_qubits=4)

    expected_state = matrix @ dense
    actual_state = sparse.apply_qubit_operator(op).to_dense()
    assert np.allclose(actual_state, expected_state)

    expected_exp = np.vdot(dense, expected_state)
    actual_exp = sparse.expectation_qubit_operator(op)
    assert np.allclose(actual_exp, expected_exp)


def test_pauli_action_mask_converts_to_and_from_bs_pauli_mask():
    n_qubits = 4
    term = ((0, "X"), (1, "Y"), (3, "Z"))
    bs_mask = term_to_masks(term, n_qubits)

    action_mask = pauli_mask_to_action_mask(bs_mask, n_qubits)

    # Action masks use OpenFermion/statevector bit order: qubit 0 is MSB.
    assert action_mask.flip_mask == (1 << 3) | (1 << 2)
    assert action_mask.sign_mask == (1 << 2) | (1 << 0)

    assert action_mask_to_pauli_mask(action_mask, n_qubits) == bs_mask
    assert action_mask.to_pauli_mask(n_qubits) == bs_mask


def test_pauli_mask_to_action_mask_default_is_hermitian_pauli_action():
    n_qubits = 3
    bs_mask = term_to_masks(((0, "Y"), (2, "X")), n_qubits)
    action_mask = PauliActionMask.from_pauli_mask(bs_mask, n_qubits)
    op = mask_to_qubit_operator(bs_mask, n_qubits)

    dense = np.zeros(1 << n_qubits, dtype=complex)
    dense[0b001] = 0.3
    dense[0b110] = -0.4j
    dense /= np.linalg.norm(dense)
    sparse = SparseQubitState.from_dense(dense)

    expected_state = get_sparse_operator(op, n_qubits=n_qubits) @ dense
    actual_state = sparse.apply_mask(action_mask).to_dense()

    assert np.allclose(actual_state, expected_state)
    assert np.allclose(
        sparse.expectation_mask(action_mask),
        np.vdot(dense, expected_state),
    )


def test_sparse_qubit_state_sparse_operator_apply_expectation_and_variance():
    dense = _sample_state()
    sparse = SparseQubitState.from_dense(dense)
    op = 0.6 * QubitOperator("X0 X2") - 1.1 * QubitOperator("Z1")
    matrix = get_sparse_operator(op, n_qubits=4)

    expected_state = matrix @ dense
    actual_state = sparse.apply_sparse_operator(matrix).to_dense()
    assert np.allclose(actual_state, expected_state)

    expected_exp = np.vdot(dense, expected_state)
    assert np.allclose(sparse.expectation_sparse_operator(matrix), expected_exp)

    expected_var = np.vdot(expected_state, expected_state) - abs(expected_exp) ** 2
    assert np.allclose(sparse.variance_qubit_operator(op), expected_var)
    assert np.allclose(sparse.variance_sparse_operator(matrix), expected_var)


def test_sparse_qubit_state_cut_entropies_match_dense_svd():
    dense = _sample_state()
    sparse = SparseQubitState.from_dense(dense)

    expected = get_entropies_at_cuts(dense, n_qubits=4, log_base=np.e)
    actual = sparse.entropies_at_cuts(log_base=np.e)

    assert np.allclose(actual, expected)


def test_sparse_qubit_state_bloch_rdms_match_statevector_trace():
    dense = _sample_state()
    sparse = SparseQubitState.from_dense(dense)

    for qubit in range(4):
        expected = reduced_density_matrix_statevector(dense, [qubit], 4)
        actual = sparse.one_qubit_rdm(qubit)
        assert np.allclose(actual, expected)

    for pair in [(0, 1), (0, 3), (2, 3)]:
        expected = reduced_density_matrix_statevector(dense, list(pair), 4)
        actual = sparse.two_qubit_rdm(*pair)
        assert np.allclose(actual, expected)


def test_sparse_bloch_fiedler_path_matches_statevector_path():
    dense = _sample_state()
    sparse = SparseQubitState.from_dense(dense)

    mi_dense, s1_dense, s2_dense = qubit_mutual_information_matrix(
        dense,
        n_qubits=4,
        base=np.e,
    )
    mi_sparse, s1_sparse, s2_sparse = sparse.qubit_mutual_information_matrix(
        base=np.e,
    )

    assert np.allclose(mi_sparse, mi_dense)
    assert np.allclose(s1_sparse, s1_dense)
    assert np.allclose(s2_sparse, s2_dense)

    dense_info = fiedler_order_from_state(
        dense,
        n_qubits=4,
        base=np.e,
        component_order="index",
    )
    sparse_info = fiedler_order_from_state(
        sparse,
        n_qubits=4,
        base=np.e,
        component_order="index",
        mutual_information_method="sparse_bloch",
    )

    assert np.allclose(
        sparse_info["mutual_information"],
        dense_info["mutual_information"],
    )
    assert sparse_info["ordering"] == dense_info["ordering"]


def test_do_fiedler_reordering_sparse_path_keeps_sparse_state_and_cut_entropies():
    dense = _sample_state()
    sparse = SparseQubitState.from_dense(dense)
    hamiltonian = QubitOperator("Z0") + 0.2 * QubitOperator("X1 X3")

    ent_sparse, h_sparse, psi_sparse_reord, info_sparse = do_fiedler_reordering(
        hamiltonian,
        sparse,
        n_qubits=4,
        verbose=False,
        component_order="index",
        log_base=np.e,
        mutual_information_method="sparse_bloch",
    )

    assert isinstance(psi_sparse_reord, SparseQubitState)

    dense_reord = reorder_statevector_axes(
        dense,
        info_sparse["ordering"],
        n_qubits=4,
    )
    ent_dense = get_entropies_at_cuts(dense_reord, n_qubits=4, log_base=np.e)

    assert np.allclose(ent_sparse, ent_dense)
    assert h_sparse.terms
