import numpy as np
from openfermion import QubitOperator, get_sparse_operator

from quasisymmetries.metrics import (
    GroupedSparsePauliCommutatorEvaluator,
    PauliTermOverlapCommutatorEvaluator,
    comm_sq_exp_fast,
)
from quasisymmetries.state_utils import SparseQubitState


def test_term_overlap_commutator_matches_full_sparse_matrix():
    hamiltonian = (
        -0.3 * QubitOperator(())
        + 0.7 * QubitOperator("Z0")
        - 0.2 * QubitOperator("X0 X1")
        + 0.4 * QubitOperator("Y1 Z2")
        + 0.1 * QubitOperator("Z0 X2")
    )
    state = SparseQubitState(
        [0, 1, 3, 6],
        [0.5, -0.25j, 0.4, 0.3j],
        n_qubits=3,
    ).normalize()
    symmetries = [
        QubitOperator("X0 Z1"),
        -QubitOperator("Y1 Y2"),
        QubitOperator("Z2"),
    ]

    evaluator = PauliTermOverlapCommutatorEvaluator(hamiltonian, state)
    expected = comm_sq_exp_fast(
        symmetries,
        get_sparse_operator(hamiltonian, n_qubits=3),
        state.to_dense(),
        3,
    )

    assert np.isclose(evaluator.cost(symmetries), expected, atol=1e-11)
    assert np.isclose(
        evaluator.cost(symmetries),
        sum(evaluator.cost([symmetry]) for symmetry in symmetries),
        atol=1e-12,
    )


def test_grouped_sparse_commutator_matches_full_sparse_matrix():
    hamiltonian = (
        -0.3 * QubitOperator(())
        + 0.7 * QubitOperator("Z0")
        - 0.2 * QubitOperator("X0 X1")
        + 0.4 * QubitOperator("Y1 Z2")
        + 0.1 * QubitOperator("Z0 X2")
        - 0.17 * QubitOperator("Y0 Y1")
    )
    state = SparseQubitState(
        [0, 1, 3, 6],
        [0.5, -0.25j, 0.4, 0.3j],
        n_qubits=3,
    ).normalize()
    symmetries = [
        QubitOperator("Z0 Z2"),
        QubitOperator("X0 Z1"),
        -QubitOperator("Y1 Y2"),
    ]

    evaluator = GroupedSparsePauliCommutatorEvaluator(hamiltonian, state)
    sparse_hamiltonian = get_sparse_operator(hamiltonian, n_qubits=3)
    for symmetry in symmetries:
        expected = comm_sq_exp_fast(
            [symmetry], sparse_hamiltonian, state.to_dense(), 3
        )
        assert np.isclose(evaluator.cost([symmetry]), expected, atol=1e-11)
    assert np.isclose(
        evaluator.cost(symmetries),
        comm_sq_exp_fast(
            symmetries, sparse_hamiltonian, state.to_dense(), 3
        ),
        atol=1e-11,
    )
