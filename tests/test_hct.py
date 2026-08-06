import numpy as np
from openfermion import QubitOperator

from quasisymmetries.gf2_utils import (
    gf2_check_commuting,
    gf2_greedy_coset_representative,
    gf2_maximal_isotropic_subspace,
    gf2_rank,
    gf2_symplectic_gram_schmidt,
)
from quasisymmetries.sym import HCT


def test_symplectic_gram_schmidt_finds_radical_and_hyperbolic_pair():
    # Z0 is radical; X1 and Z1 form one hyperbolic pair.
    space = np.asarray(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    radical, pairs = gf2_symplectic_gram_schmidt(space, n_qubits=2)

    assert gf2_rank(radical) == 1
    assert len(pairs) == 1
    assert gf2_check_commuting(radical, space, n_qubits=2)

    isotropic = gf2_maximal_isotropic_subspace(space, n_qubits=2)
    assert gf2_rank(isotropic) == 2
    assert gf2_check_commuting(isotropic, isotropic, n_qubits=2)


def test_maximal_isotropic_subspace_ranks_all_hyperbolic_directions():
    space = np.asarray(
        [
            [1, 0],  # X0
            [0, 1],  # Z0
        ],
        dtype=np.uint8,
    )
    _, pairs = gf2_symplectic_gram_schmidt(space, n_qubits=1)
    first, second = pairs[0]
    product_direction = first ^ second

    isotropic = gf2_maximal_isotropic_subspace(
        space,
        n_qubits=1,
        vector_cost=lambda vector: (
            0 if np.array_equal(vector, product_direction) else 1
        ),
    )

    assert np.array_equal(isotropic, np.asarray([product_direction]))
    assert gf2_check_commuting(isotropic, isotropic, n_qubits=1)


def test_hct_metric_selects_best_hyperbolic_direction():
    hamiltonian = QubitOperator("X0", 1.0) + QubitOperator("Z0", 1.0)

    def prefer_y(symmetry):
        return 0 if symmetry == QubitOperator("Y0") else 1

    symmetries, _ = HCT(
        hamiltonian,
        n_sym=1,
        sym_metric_func=prefer_y,
        use_coeffs_eps=True,
        verbose=False,
    )

    assert symmetries == [QubitOperator("Y0")]


def test_coset_representative_removes_seeded_support_lexicographically():
    # X0 X1 differs from the shorter X1 by the seeded X0 direction.
    vector = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    seeded = np.asarray([[1, 0, 0, 0]], dtype=np.uint8)

    def score(row):
        weight = int(np.count_nonzero(row[:2] | row[2:]))
        packed = sum(int(bit) << index for index, bit in enumerate(row))
        return (0, weight, packed)

    reduced = gf2_greedy_coset_representative(vector, seeded, score)

    assert np.array_equal(reduced, np.asarray([0, 1, 0, 0]))


def test_seeded_quotient_sgs_keeps_seed_out_of_new_generator():
    seeded = np.asarray([[1, 0, 0, 0]], dtype=np.uint8)  # X0
    quotient = np.asarray(
        [
            [1, 1, 0, 0],  # X0 X1, equivalent to X1 modulo the seed
            [0, 0, 0, 1],  # Z1
        ],
        dtype=np.uint8,
    )

    def score(row):
        weight = int(np.count_nonzero(row[:2] | row[2:]))
        packed = sum(int(bit) << index for index, bit in enumerate(row))
        return (0, weight, packed)

    isotropic = gf2_maximal_isotropic_subspace(
        quotient,
        n_qubits=2,
        vector_cost=score,
        representative_func=lambda row: gf2_greedy_coset_representative(
            row, seeded, score
        ),
    )

    assert gf2_check_commuting(seeded, isotropic, n_qubits=2)
    assert gf2_check_commuting(isotropic, isotropic, n_qubits=2)
    assert np.array_equal(isotropic, np.asarray([[0, 1, 0, 0]]))


def test_hct_returns_maximal_commuting_exact_symmetries():
    hamiltonian = QubitOperator("Z0 Z1", 1.0)
    symmetries, thresholds = HCT(
        hamiltonian, n_sym=2, use_coeffs_eps=True, verbose=False
    )

    assert thresholds == [0.0, 0.0]
    for first in symmetries:
        assert not (first * hamiltonian - hamiltonian * first).terms
        for second in symmetries:
            assert not (first * second - second * first).terms


def test_hct_threshold_sweep_preserves_prior_generators():
    hamiltonian = QubitOperator("X0", 1.0) + QubitOperator("Z0", 0.5)
    symmetries, thresholds = HCT(
        hamiltonian,
        n_sym=1,
        num_intervals=3,
        eps_max=1.000001,
        verbose=False,
    )

    assert len(symmetries) == 1
    assert len(thresholds) == 1
    assert thresholds[0] > 0.5


def test_hct_coefficient_schedule_drops_equal_magnitude_terms():
    hamiltonian = QubitOperator("X0", 1.0) + QubitOperator("Z0", 1.0)
    symmetries, thresholds = HCT(
        hamiltonian, n_sym=1, use_coeffs_eps=True, verbose=False
    )

    assert len(symmetries) == 1
    assert thresholds[0] > 1.0
