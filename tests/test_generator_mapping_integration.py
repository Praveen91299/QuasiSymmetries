import numpy as np
from openfermion import (
    QubitOperator,
    get_ground_state,
    get_sparse_operator,
)

from quasisymmetries.metrics import get_permuted_bipartite_entanglement
from quasisymmetries.op_utils import (
    permute_sym_to_start,
    taper_hamiltonian,
    taper_symmetries,
)


def oq_equal(left, right, atol=1e-10):
    difference = left - right
    difference.compress(abs_tol=atol)
    return not difference.terms


def example_problem():
    symmetries = [
        QubitOperator("X1"),
        QubitOperator("X0 X1"),
    ]
    hamiltonian = (
        -QubitOperator("X1")
        - 0.4 * QubitOperator("X0 X1")
        - 0.2 * QubitOperator("X0")
    )
    return hamiltonian, symmetries


def test_permute_default_is_legacy_and_positive_z_is_opt_in():
    hamiltonian, symmetries = example_problem()

    _, legacy, _ = permute_sym_to_start(
        hamiltonian,
        symmetries,
        2,
        return_clifford_perm=True,
    )
    _, positive, _ = permute_sym_to_start(
        hamiltonian,
        symmetries,
        2,
        return_clifford_perm=True,
        generator_mapping="positive_z",
    )

    assert legacy.generator_mapping == "row_reduced"
    assert oq_equal(legacy.transform(symmetries[0]), QubitOperator("Z0"))
    assert oq_equal(
        legacy.transform(symmetries[1]),
        QubitOperator("Z0 Z1"),
    )
    assert positive.generator_mapping == "positive_z"
    assert all(
        oq_equal(positive.transform(symmetry), QubitOperator(f"Z{i}"))
        for i, symmetry in enumerate(symmetries)
    )


def test_entanglement_helper_forwards_generator_mapping():
    hamiltonian, symmetries = example_problem()
    energy, ground_state = get_ground_state(
        get_sparse_operator(hamiltonian, n_qubits=2)
    )

    legacy = get_permuted_bipartite_entanglement(
        symmetries,
        hamiltonian,
        2,
        fci_energy=energy,
        fci_gs=ground_state,
        return_state=True,
        return_clifford=True,
    )
    positive = get_permuted_bipartite_entanglement(
        symmetries,
        hamiltonian,
        2,
        fci_energy=energy,
        fci_gs=ground_state,
        return_state=True,
        return_clifford=True,
        generator_mapping="positive_z",
    )

    _, legacy_h, legacy_clifford, legacy_state = legacy
    _, positive_h, positive_clifford, positive_state = positive
    assert legacy_clifford.generator_mapping == "row_reduced"
    assert positive_clifford.generator_mapping == "positive_z"
    assert np.allclose(
        legacy_state,
        legacy_clifford.transform_state(ground_state),
    )
    assert np.allclose(
        positive_state,
        positive_clifford.transform_state(ground_state),
    )
    assert np.allclose(
        np.linalg.eigvalsh(get_sparse_operator(legacy_h, 2).toarray()),
        np.linalg.eigvalsh(get_sparse_operator(positive_h, 2).toarray()),
    )


def test_taper_symmetries_defaults_to_original_positive_z_labels():
    hamiltonian, symmetries = example_problem()
    labels = [1, 1]

    default_tapered = taper_symmetries(
        hamiltonian,
        symmetries,
        labels,
        labels,
        2,
    )
    positive_h = permute_sym_to_start(
        hamiltonian,
        symmetries,
        2,
        generator_mapping="positive_z",
    )
    expected = taper_hamiltonian(positive_h, labels, labels)
    legacy_tapered = taper_symmetries(
        hamiltonian,
        symmetries,
        labels,
        labels,
        2,
        generator_mapping="row_reduced",
    )

    assert oq_equal(default_tapered, expected)
    assert not oq_equal(default_tapered, legacy_tapered)
