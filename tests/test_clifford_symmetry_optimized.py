import importlib.util
import itertools
import os
import random
import sys

import numpy as np
import pytest
from scipy import sparse
from openfermion import QubitOperator, get_sparse_operator


MODULE_PATH = os.environ.get("MODULE_UNDER_TEST")
if MODULE_PATH:
    spec = importlib.util.spec_from_file_location(
        "clifford_symmetry_optimized", MODULE_PATH
    )
    cs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cs
    spec.loader.exec_module(cs)
else:
    import quasisymmetries.clifford_symmetry_optimized as cs


def oq_equal(a, b, atol=1e-10):
    aa = QubitOperator()
    aa += a
    aa -= b
    aa.compress(abs_tol=atol)
    return len(aa.terms) == 0


def mat(op, n):
    return get_sparse_operator(op, n_qubits=n).toarray()


def dense_conjugate_term(term_op, factor_descriptions, n):
    U = mat(QubitOperator(()), n)
    for desc in factor_descriptions:
        gate = cs.factor_from_parsed_gate(cs.parse_factor_description(desc))
        U = mat(gate, n) @ U
    return U @ mat(term_op, n) @ U.conj().T


def assert_operator_matches_dense(op, dense, n, atol=1e-10):
    got = mat(op, n)
    assert np.allclose(got, dense, atol=atol)


# ---------------- basic utilities ----------------


def test_qubit_operator_num_qubits_identity_and_nontrivial():
    assert cs.qubit_operator_num_qubits(QubitOperator(())) == 0
    assert cs.qubit_operator_num_qubits(QubitOperator("X3 Y5")) == 6


def test_single_pauli_term_rejects_multiple_terms():
    with pytest.raises(ValueError):
        cs.single_pauli_term(QubitOperator("X0") + QubitOperator("Z1"))


def test_pauli_dict_roundtrip_binary():
    pmap = {0: "X", 2: "Y", 4: "Z"}
    x, z = cs.binary_from_pauli_map(pmap, 5)
    assert cs.pauli_map_from_binary(x, z) == pmap
    assert oq_equal(cs.pauli_dict_to_qubit_operator(pmap, coeff=-2), -2 * QubitOperator("X0 Y2 Z4"))


def test_binary_symplectic_commutes():
    x0, z0 = cs.binary_from_pauli_map({0: "X"}, 1)
    x1, z1 = cs.binary_from_pauli_map({0: "Z"}, 1)
    assert not cs.binary_symplectic_commutes(x0, z0, x1, z1)
    assert cs.binary_symplectic_commutes(x0, z0, x0, z0)


def test_term_mask_roundtrip():
    term = ((0, "X"), (3, "Y"), (5, "Z"))
    x, z = cs.term_to_masks(term)
    assert cs.masks_to_term(x, z, 6) == term


# ---------------- exact gate rules ----------------


@pytest.mark.parametrize(
    "desc,input_op,expected",
    [
        ("H(0)", QubitOperator("X0"), QubitOperator("Z0")),
        ("H(0)", QubitOperator("Z0"), QubitOperator("X0")),
        ("H(0)", QubitOperator("Y0"), -QubitOperator("Y0")),
        ("Sdg(0)", QubitOperator("X0"), -QubitOperator("Y0")),
        ("Sdg(0)", QubitOperator("Y0"), QubitOperator("X0")),
        ("Sdg(0)", QubitOperator("Z0"), QubitOperator("Z0")),
        ("S(0)", QubitOperator("X0"), QubitOperator("Y0")),
        ("S(0)", QubitOperator("Y0"), -QubitOperator("X0")),
    ],
)
def test_single_qubit_exact_rules(desc, input_op, expected):
    got = cs.conjugate_single_pauli_by_factor_sequence_exact(input_op, [desc], n_qubits=1)
    assert oq_equal(got, expected)


@pytest.mark.parametrize("pc", ["I", "X", "Y", "Z"])
@pytest.mark.parametrize("pt", ["I", "X", "Y", "Z"])
def test_cnot_exact_against_dense_for_all_local_paulis(pc, pt):
    pieces = []
    if pc != "I":
        pieces.append(f"{pc}0")
    if pt != "I":
        pieces.append(f"{pt}1")
    op = QubitOperator(" ".join(pieces)) if pieces else QubitOperator(())
    got = cs.conjugate_single_pauli_by_factor_sequence_exact(op, ["CNOT(0->1)"], n_qubits=2)
    dense = dense_conjugate_term(op, ["CNOT(0->1)"], 2)
    assert_operator_matches_dense(got, dense, 2)


def test_composite_sequence_exact_against_dense():
    op = 0.7 * QubitOperator("X0 Y1 Z2")
    seq = ["Sdg(1)", "H(2)", "CNOT(0->2)", "H(0)", "S(2)"]
    got = cs.conjugate_qubit_operator_by_clifford_factors_exact(op, seq, n_qubits=3)
    dense = dense_conjugate_term(op, seq, 3)
    assert_operator_matches_dense(got, dense, 3)


def test_aliases_are_exact_not_phase_dropping():
    got = cs.conjugate_qubit_operator_by_clifford_factors(QubitOperator("Y0"), ["H(0)"], n_qubits=1)
    assert oq_equal(got, -QubitOperator("Y0"))


def test_parsed_gate_path_matches_string_path():
    seq = ["H(0)", "Sdg(1)", "CNOT(0->1)"]
    parsed = cs.parse_factor_descriptions(seq)
    op = QubitOperator("X0 Z1", 2.0)
    assert oq_equal(
        cs.conjugate_qubit_operator_by_clifford_factors_exact(op, seq, n_qubits=2),
        cs.conjugate_qubit_operator_by_clifford_factors_exact(op, parsed, n_qubits=2),
    )


# ---------------- synthesis ----------------


def test_synthesis_simple_maps_to_single_z():
    res = cs.synthesize_ordered_symmetry_clifford([QubitOperator("X0 X1")], n_qubits=2)
    assert res.mapped_qubits == [0]
    assert oq_equal(res.transformed_generators[0], QubitOperator("Z0"))
    direct = cs.conjugate_qubit_operator_by_clifford_factors_exact(
        QubitOperator("X0 X1"), res.parsed_gates, n_qubits=2
    )
    assert oq_equal(direct, QubitOperator("Z0"))


def test_z_native_synthesis_keeps_z_strings_in_z_basis():
    res = cs.synthesize_ordered_symmetry_clifford(
        [QubitOperator("Z0 Z1")],
        n_qubits=2,
        synthesis_basis="Z",
    )
    assert res.factor_descriptions == ("CNOT(1->0)",)
    assert oq_equal(res.transform(QubitOperator("Z0 Z1")), QubitOperator("Z0"))


def test_synthesis_basis_validation():
    with pytest.raises(ValueError, match="either 'X' or 'Z'"):
        cs.Clifford.from_symmetries(
            [QubitOperator("Z0")],
            synthesis_basis="Y",
        )


def test_synthesis_rejects_noncommuting_symmetries():
    with pytest.raises(ValueError, match="do not commute"):
        cs.synthesize_ordered_symmetry_clifford([QubitOperator("X0"), QubitOperator("Z0")], n_qubits=1)


def test_synthesis_rejects_dependent_symmetries():
    with pytest.raises(ValueError, match="dependent"):
        cs.synthesize_ordered_symmetry_clifford([QubitOperator("X0"), QubitOperator("X0")], n_qubits=1)


def test_synthesis_rejects_nonhermitian_unit_coefficients():
    with pytest.raises(ValueError, match="real"):
        cs.synthesize_ordered_symmetry_clifford([1j * QubitOperator("X0")], n_qubits=1)


def test_synthesis_row_reduced_generator_contract():
    # The original second symmetry maps to Z0 Z1, while the row-reduced second
    # generator maps to Z0. This codifies the intended row-reduced behavior.
    syms = [QubitOperator("X1"), QubitOperator("X0 X1")]
    res = cs.synthesize_ordered_symmetry_clifford(syms, n_qubits=2)
    assert [str(g) for g in res.transformed_generators] == [str(QubitOperator("Z1")), str(QubitOperator("Z0"))]
    direct_second = cs.conjugate_qubit_operator_by_clifford_factors_exact(syms[1], res.parsed_gates, n_qubits=2)
    assert oq_equal(direct_second, QubitOperator("Z0 Z1"))


def test_full_clifford_optional_and_matches_sequence():
    res = cs.synthesize_ordered_symmetry_clifford(
        [QubitOperator("Y0 Z1")], n_qubits=2, return_full_clifford=True
    )
    assert res.full_clifford is not None
    direct = cs.conjugate_qubit_operator_by_clifford_factors_exact(
        QubitOperator("Y0 Z1"), res.parsed_gates, n_qubits=2
    )
    U = mat(res.full_clifford, 2)
    dense = U @ mat(QubitOperator("Y0 Z1"), 2) @ U.conj().T
    assert_operator_matches_dense(direct, dense, 2)


def test_inverse_clifford_factor_sequence_recovers_pauli():
    op = -QubitOperator("Y0 X1 Z2")
    sequence = ["Sdg(0)", "H(1)", "CNOT(1->2)", "H(0)"]
    transformed = cs.conjugate_qubit_operator_by_clifford_factors_exact(
        op, sequence, n_qubits=3
    )
    recovered = cs.inverse_conjugate_qubit_operator_by_clifford_factors_exact(
        transformed, sequence, n_qubits=3
    )
    assert oq_equal(recovered, op)


# ---------------- Hamiltonian and spectra ----------------


def random_pauli_term(n, rng):
    labels = []
    for q in range(n):
        p = rng.choice(["I", "X", "Y", "Z"])
        if p != "I":
            labels.append(f"{p}{q}")
    return " ".join(labels)


def random_qubit_operator(n, n_terms, seed=0):
    rng = random.Random(seed)
    op = QubitOperator()
    for _ in range(n_terms):
        term = random_pauli_term(n, rng)
        coeff = rng.uniform(-1.0, 1.0)
        op += coeff * (QubitOperator(term) if term else QubitOperator(()))
    op.compress(abs_tol=1e-12)
    return op


def test_random_hamiltonian_exact_conjugation_matches_dense():
    n = 4
    h = random_qubit_operator(n, 20, seed=123)
    seq = ["H(0)", "Sdg(1)", "CNOT(0->2)", "H(3)", "CNOT(3->1)", "S(2)"]
    got = cs.conjugate_qubit_operator_by_clifford_factors_exact(h, seq, n_qubits=n)

    U = np.eye(1 << n, dtype=complex)
    for desc in seq:
        gate = cs.factor_from_parsed_gate(cs.parse_factor_description(desc))
        U = mat(gate, n) @ U
    dense = U @ mat(h, n) @ U.conj().T
    assert_operator_matches_dense(got, dense, n)


def test_spectrum_preserved_for_synthesized_clifford():
    n = 3
    h = QubitOperator("X0 X1", 0.5) + QubitOperator("Y0 Y1", -0.25) + QubitOperator("Z2", 1.2)
    syms = [QubitOperator("Z0 Z1")]
    res = cs.synthesize_ordered_symmetry_clifford(syms, n_qubits=n)
    h2 = cs.conjugate_qubit_operator_by_clifford_factors_exact(h, res.parsed_gates, n_qubits=n)
    assert cs.check_same_spectrum(h, h2, n_qubits=n)


# ---------------- sector ordering and sparse matrix handling ----------------


def test_sector_ordering_from_symmetry_qubits():
    order, sector_to_indices, sectors = cs.sector_ordering_from_symmetry_qubits(3, [2, 0])
    assert sorted(order) == list(range(8))
    assert sectors == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert sector_to_indices[(0, 0)] == [0, 2]


def test_permutation_matrix_sparse_by_default():
    P = cs.permutation_matrix_from_order([2, 0, 1])
    assert sparse.issparse(P)
    dense = P.toarray()
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert np.allclose(dense, expected)


def test_reordered_matrix_by_sector_returns_sparse_and_correct_reindexing():
    h = QubitOperator("Z0") + 0.3 * QubitOperator("X1")
    res = cs.reordered_matrix_by_sector(h, symmetry_qubits=[0], factor_descriptions=[], n_qubits=2)
    assert sparse.issparse(res.transformed_matrix)
    assert sparse.issparse(res.reordered_matrix)
    idx = np.asarray(res.basis_order)
    dense_expected = mat(h, 2)[np.ix_(idx, idx)]
    assert np.allclose(res.reordered_matrix.toarray(), dense_expected)


def test_block_structure_has_zero_off_sector_blocks_for_commuting_hamiltonian():
    n = 2
    h = QubitOperator("Z0") + 0.5 * QubitOperator("X1")
    syms = [QubitOperator("Z0")]
    out = cs.build_symmetry_block_structure(h, syms, n_qubits=n)
    H = out.reordered_result.reordered_matrix.toarray()
    b = out.reordered_result.sector_boundaries[0]
    assert np.allclose(H[:b, b:], 0.0)
    assert np.allclose(H[b:, :b], 0.0)


# ---------------- permutation helpers and packed pipeline ----------------


def test_invert_permutation_and_validation():
    assert cs.invert_permutation([2, 0, 1]) == [1, 2, 0]
    with pytest.raises(ValueError):
        cs.invert_permutation([0, 0, 1])
    with pytest.raises(ValueError):
        cs.invert_permutation([0, 1, 3])


def test_permute_qubits_in_qubit_operator():
    op = QubitOperator("X0 Y2", 2.0) + QubitOperator("Z1", -1.0)
    got = cs.permute_qubits_in_qubit_operator(op, [2, 0, 1])
    expected = 2.0 * QubitOperator("X2 Y1") - QubitOperator("Z0")
    assert oq_equal(got, expected)


def test_make_symmetry_qubits_last_permutation():
    perm, new_sym = cs.make_symmetry_qubits_last_permutation(5, [1, 3])
    assert perm == [0, 3, 1, 4, 2]
    assert new_sym == [3, 4]


def test_move_symmetry_qubits_to_end():
    h = QubitOperator("Z1") + QubitOperator("X0")
    packed = cs.move_symmetry_qubits_to_end(h, mapped_qubits=[1], n_qubits=3)
    assert packed.qubit_permutation == [0, 2, 1]
    assert packed.permuted_symmetry_qubits == [2]
    assert oq_equal(packed.permuted_hamiltonian, QubitOperator("Z2") + QubitOperator("X0"))


def test_build_symmetry_block_structure_with_packed_qubits():
    h = QubitOperator("Z0") + QubitOperator("X1")
    syms = [QubitOperator("Z0")]
    out = cs.build_symmetry_block_structure_with_packed_qubits(h, syms, n_qubits=2, reorder_sector=True)
    assert out.packed_symmetry_qubits == [1]
    assert sparse.issparse(out.reordered_matrix)
    assert out.ordered_sectors == [(0,), (1,)]


# ---------------- small exhaustive property tests ----------------


def all_nonidentity_paulis(n):
    labels = ["I", "X", "Y", "Z"]
    for ps in itertools.product(labels, repeat=n):
        if all(p == "I" for p in ps):
            continue
        parts = [f"{p}{q}" for q, p in enumerate(ps) if p != "I"]
        yield QubitOperator(" ".join(parts))


def commutes(op1, op2, n):
    _, p1 = cs.single_pauli_term(op1)
    _, p2 = cs.single_pauli_term(op2)
    x1, z1 = cs.binary_from_pauli_map(p1, n)
    x2, z2 = cs.binary_from_pauli_map(p2, n)
    return cs.binary_symplectic_commutes(x1, z1, x2, z2)


@pytest.mark.parametrize("synthesis_basis", ["X", "Z"])
def test_exhaustive_single_generator_synthesis_for_two_qubits(
    synthesis_basis,
):
    n = 2
    for sym in all_nonidentity_paulis(n):
        res = cs.synthesize_ordered_symmetry_clifford(
            [sym],
            n_qubits=n,
            synthesis_basis=synthesis_basis,
        )
        direct = cs.conjugate_qubit_operator_by_clifford_factors_exact(sym, res.parsed_gates, n_qubits=n)
        assert oq_equal(direct, QubitOperator(f"Z{res.mapped_qubits[0]}"))


@pytest.mark.parametrize("synthesis_basis", ["X", "Z"])
def test_exhaustive_commuting_independent_pairs_two_qubits(
    synthesis_basis,
):
    n = 2
    paulis = list(all_nonidentity_paulis(n))
    checked = 0
    for a, b in itertools.combinations(paulis, 2):
        if not commutes(a, b, n):
            continue
        try:
            res = cs.synthesize_ordered_symmetry_clifford(
                [a, b],
                n_qubits=n,
                synthesis_basis=synthesis_basis,
            )
        except ValueError as exc:
            # Dependent pairs are allowed to be rejected.
            assert "dependent" in str(exc)
            continue
        assert len(res.mapped_qubits) == 2
        for i, q in enumerate(res.mapped_qubits):
            assert oq_equal(res.transformed_generators[i], QubitOperator(f"Z{q}"))
        for symmetry in (a, b):
            transformed = res.transform(symmetry)
            assert all(
                pauli == "Z"
                for term in transformed.terms
                for _, pauli in term
            )
            assert oq_equal(res.inverse_transform(transformed), symmetry)
        checked += 1
    assert checked > 0


# ---------------- unified Clifford class ----------------


def test_clifford_binary_transform_matches_factors_exhaustively():
    n = 2
    sequence = ["Sdg(0)", "H(1)", "CNOT(0->1)", "H(0)"]
    permutation = [1, 0]
    clifford = cs.Clifford(n, sequence, permutation)

    for op in [QubitOperator(())] + list(all_nonidentity_paulis(n)):
        expected = cs.conjugate_qubit_operator_by_clifford_factors_exact(
            op, sequence, n_qubits=n
        )
        expected = cs.permute_qubits_in_qubit_operator(expected, permutation)
        actual = clifford.transform(op)
        assert oq_equal(actual, expected)
        assert oq_equal(clifford.inverse_transform(actual), op)


def test_clifford_transform_matches_sparse_matrix_for_operator_sum():
    n = 3
    op = (
        0.3 * QubitOperator(())
        - 0.7 * QubitOperator("Y0 X2")
        + (0.2 + 0.4j) * QubitOperator("Z1")
    )
    clifford = cs.Clifford(
        n,
        ["H(0)", "Sdg(2)", "CNOT(0->1)", "CNOT(2->0)"],
        [2, 0, 1],
    )
    transformed = clifford.transform(op)
    assert_operator_matches_dense(
        transformed,
        clifford.sparse_matrix @ mat(op, n) @ clifford.sparse_matrix.getH(),
        n,
    )
    recovered = clifford.inverse_transform(transformed)
    assert oq_equal(recovered, op)


def test_clifford_sparse_matrix_is_cached_and_invalidated():
    clifford = cs.Clifford(2, ["H(0)"])
    first = clifford.sparse_matrix
    assert clifford.sparse_matrix is first

    clifford.set_permutation([1, 0])
    second = clifford.sparse_matrix
    assert second is not first
    assert clifford.sparse_matrix is second

    clifford.set_factor_descriptions(["Sdg(1)"])
    assert clifford.sparse_matrix is not second


@pytest.mark.parametrize("synthesis_basis", ["X", "Z"])
def test_clifford_from_symmetries_maps_row_reduced_generators_to_front_zs(
    synthesis_basis,
):
    syms = [QubitOperator("X1"), QubitOperator("X0 X1")]
    clifford = cs.Clifford.from_symmetries(
        syms,
        n_qubits=3,
        synthesis_basis=synthesis_basis,
    )
    assert clifford.synthesis_basis == synthesis_basis
    assert clifford.symmetry_qubits == (0, 1)
    for i, generator in enumerate(clifford.canonical_generators):
        assert oq_equal(generator, QubitOperator(f"Z{i}"))
    assert all(
        oq_equal(actual, clifford.transform(source))
        for actual, source in zip(clifford.transformed_symmetries, syms)
    )


def test_clifford_list_transform_and_inverse():
    op = QubitOperator("Y0 X1")
    cliffords = [
        cs.Clifford(2, ["H(0)"], [1, 0]),
        cs.Clifford(2, ["Sdg(1)", "CNOT(1->0)"]),
    ]
    expected = cliffords[1].transform(cliffords[0].transform(op))
    transformed = cs.Clifford.transform_by_cliffords(op, cliffords)
    assert oq_equal(transformed, expected)
    assert oq_equal(
        cs.Clifford.inverse_transform_by_cliffords(transformed, cliffords),
        op,
    )

    op_sparse = get_sparse_operator(op, n_qubits=2).tocsr()
    transformed_sparse = cs.Clifford.transform_sparse_by_cliffords(
        op_sparse, cliffords
    )
    assert np.allclose(transformed_sparse.toarray(), mat(transformed, 2))
    recovered_sparse = cs.Clifford.inverse_transform_sparse_by_cliffords(
        transformed_sparse, cliffords
    )
    assert np.allclose(recovered_sparse.toarray(), op_sparse.toarray())

    state = np.arange(4, dtype=complex)
    state /= np.linalg.norm(state)
    transformed_state = cs.Clifford.transform_state_by_cliffords(
        state, cliffords
    )
    recovered_state = cs.Clifford.inverse_transform_state_by_cliffords(
        transformed_state, cliffords
    )
    assert np.allclose(recovered_state, state)


def test_clifford_portable_description_roundtrip():
    original = cs.Clifford(
        3,
        ["Sdg(0)", "H(2)", "CNOT(2->1)"],
        [1, 2, 0],
        synthesis_basis="Z",
        mapped_qubits=[2],
    )
    restored = cs.Clifford.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    op = QubitOperator("Y0 X1 Z2")
    assert oq_equal(restored.transform(op), original.transform(op))


def test_clifford_random_three_qubit_tableaus_match_factor_path():
    rng = random.Random(8172)
    n = 3
    paulis = [QubitOperator(())] + list(all_nonidentity_paulis(n))
    for _ in range(8):
        sequence = []
        for _ in range(10):
            gate = rng.choice(["H", "S", "Sdg", "CNOT"])
            if gate == "CNOT":
                control, target = rng.sample(range(n), 2)
                sequence.append(f"CNOT({control}->{target})")
            else:
                sequence.append(f"{gate}({rng.randrange(n)})")
        permutation = list(range(n))
        rng.shuffle(permutation)
        clifford = cs.Clifford(n, sequence, permutation)

        for op in paulis:
            expected = cs.conjugate_qubit_operator_by_clifford_factors_exact(
                op, sequence, n_qubits=n
            )
            expected = cs.permute_qubits_in_qubit_operator(
                expected, permutation
            )
            assert oq_equal(clifford.transform(op), expected)
            assert oq_equal(clifford.inverse_transform(expected), op)


def test_clifford_from_symmetries_matches_previous_pipeline_exactly():
    n = 3
    syms = [QubitOperator("Y0 Y1"), QubitOperator("X0 X1 Z2")]
    hamiltonian = (
        0.2 * QubitOperator(())
        - 0.4 * QubitOperator("X0 Y2")
        + 0.7 * QubitOperator("Y0 Z1 X2")
    )

    old = cs.synthesize_ordered_symmetry_clifford(syms, n_qubits=n)
    old_transformed = cs.conjugate_qubit_operator_by_clifford_factors_exact(
        hamiltonian, old.parsed_gates, n_qubits=n
    )
    mapped_set = set(old.mapped_qubits)
    old_order = old.mapped_qubits + [
        q for q in range(n) if q not in mapped_set
    ]
    old_permutation = [0] * n
    for new_q, old_q in enumerate(old_order):
        old_permutation[old_q] = new_q
    old_transformed = cs.permute_qubits_in_qubit_operator(
        old_transformed, old_permutation
    )

    clifford = cs.Clifford.from_symmetries(syms, n_qubits=n)
    assert list(clifford.permutation) == old_permutation
    assert oq_equal(clifford.transform(hamiltonian), old_transformed)


def test_clifford_reorders_transformed_operator_by_symmetry_sectors():
    hamiltonian = QubitOperator("Z0") + 0.5 * QubitOperator("X1")
    clifford = cs.Clifford.from_symmetries(
        [QubitOperator("Z0")], n_qubits=2
    )
    reordered = clifford.reorder_operator_by_symmetry_sectors(hamiltonian)
    boundary = reordered.sector_boundaries[0]
    matrix = reordered.reordered_matrix.toarray()
    assert np.allclose(matrix[:boundary, boundary:], 0.0)
    assert np.allclose(matrix[boundary:, :boundary], 0.0)
