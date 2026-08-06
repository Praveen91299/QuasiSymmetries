import numpy as np
import pytest

from quasisymmetries.block2_qubit_benchmark import (
    _unitaries_need_complex_driver,
)
from quasisymmetries.clifford_symmetry_optimized import Clifford
from quasisymmetries.mps_unitary import (
    OrbitalRotationUnitary,
    PermutationUnitary,
    compose_mps_unitaries,
    mps_configuration_probabilities,
    mps_prefix_configurations_for_probability_mass,
    mps_prefix_configurations_above_probability,
    mps_reduced_density_matrix,
    project_mps_onto_prefix_configurations,
    transform_qubit_mps_arrays,
)


def _statevector_to_mps(state, n_qubits):
    work = np.asarray(state).reshape(1, -1)
    arrays = []
    left_dim = 1
    for _site in range(n_qubits - 1):
        u, singular_values, vh = np.linalg.svd(
            work.reshape(left_dim * 2, -1), full_matrices=False
        )
        right_dim = len(singular_values)
        arrays.append(u.reshape(left_dim, 2, right_dim))
        work = singular_values[:, None] * vh
        left_dim = right_dim
    arrays.append(work.reshape(left_dim, 2, 1))
    return arrays


def _mps_to_statevector(arrays):
    state = arrays[0]
    for array in arrays[1:]:
        state = np.tensordot(state, array, axes=(-1, 0))
    return state.reshape(-1)


def test_clifford_exposes_mps_unitary_interface():
    clifford = Clifford(3, ["H(0)", "CNOT(0->2)"], [2, 0, 1])
    assert clifford.get_parsed_gates() == clifford.parsed_gates
    assert clifford.get_permutation() == clifford.permutation


def test_composed_cliffords_match_sequential_state_application():
    first = Clifford(3, ["H(0)", "CNOT(0->2)"], [2, 0, 1])
    second = Clifford(3, ["Sdg(1)", "CNOT(2->0)"], [1, 2, 0])
    third = PermutationUnitary([0, 2, 1])
    components = [first, second, third]
    composed = compose_mps_unitaries(components)

    rng = np.random.default_rng(9182)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state /= np.linalg.norm(state)
    expected = state
    for unitary in components:
        if isinstance(unitary, Clifford):
            expected = unitary.transform_state(expected)
        else:
            expected = np.transpose(
                expected.reshape((2,) * unitary.n_qubits),
                axes=np.argsort(unitary.permutation),
            ).reshape(-1)

    combined_clifford = Clifford(
        composed.n_qubits,
        composed.parsed_gates,
        composed.permutation,
    )
    assert np.allclose(combined_clifford.transform_state(state), expected)

    transformed, metadata = transform_qubit_mps_arrays(
        _statevector_to_mps(state, 3),
        unitaries=components,
        cutoff=1e-14,
    )
    actual = _mps_to_statevector(transformed)
    phase = np.vdot(expected, actual)
    assert np.isclose(abs(phase), 1.0)
    assert np.allclose(actual * np.exp(-1j * np.angle(phase)), expected)
    assert metadata["number_of_composed_gates"] == len(
        composed.parsed_gates
    )


def test_orbital_rotation_is_an_mps_unitary():
    theta = 0.37
    rotation = np.array(
        [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    )
    unitary = OrbitalRotationUnitary(rotation)
    assert unitary.n_qubits == 4
    assert unitary.get_permutation() == (0, 1, 2, 3)
    assert [gate[0] for gate in unitary.get_parsed_gates()] == [
        "FSWAP",
        "FGIVENS",
        "FGIVENS",
        "FSWAP",
    ]


def test_composition_rejects_mismatched_sizes():
    with pytest.raises(ValueError, match="acts on 4 qubits"):
        compose_mps_unitaries(
            [PermutationUnitary([0, 1]), OrbitalRotationUnitary(np.eye(2))]
        )


def test_complex_driver_detection_includes_clifford_phase_gates():
    assert _unitaries_need_complex_driver(
        (Clifford(3, ["H(0)", "Sdg(1)", "CNOT(1->2)"]),),
        3,
    )
    assert not _unitaries_need_complex_driver(
        (Clifford(3, ["H(0)", "X(1)", "CNOT(1->2)"]),),
        3,
    )


class _PhaseUnitary:
    n_qubits = 2

    def __init__(self, exponent):
        self.exponent = exponent

    def get_parsed_gates(self):
        return (("PHASE", 0, self.exponent),)

    def get_permutation(self):
        return (0, 1)


def test_complex_driver_detection_checks_general_phase_gate():
    assert _unitaries_need_complex_driver((_PhaseUnitary(0.25j),), 2)
    assert not _unitaries_need_complex_driver(
        (_PhaseUnitary(1j * np.pi),), 2
    )


def test_mps_partial_trace_and_configuration_probabilities():
    # (|000> + |101> + i|110>) / sqrt(3)
    state = np.zeros(8, dtype=complex)
    state[[0, 5, 6]] = [1, 1, 1j]
    state /= np.linalg.norm(state)
    mps = _statevector_to_mps(state, 3)

    matrix = state.reshape(4, 2)
    expected_rdm = matrix @ matrix.conj().T
    actual_rdm = mps_reduced_density_matrix(mps, keep_sites=(0, 1))
    assert np.allclose(actual_rdm, expected_rdm)

    probabilities = mps_configuration_probabilities(
        mps, keep_sites=(0, 1)
    )
    assert set(probabilities) == {"00", "01", "10", "11"}
    assert np.isclose(probabilities["00"], 1 / 3)
    assert np.isclose(probabilities["10"], 1 / 3)
    assert np.isclose(probabilities["11"], 1 / 3)
    assert np.isclose(probabilities["01"], 0.0)


def test_configuration_probabilities_support_noncontiguous_sites():
    state = np.zeros(8)
    state[[0, 5]] = 1 / np.sqrt(2)
    probabilities = mps_configuration_probabilities(
        _statevector_to_mps(state, 3), keep_sites=(0, 2)
    )
    assert np.isclose(probabilities["00"], 0.5)
    assert np.isclose(probabilities["11"], 0.5)
    assert np.isclose(sum(probabilities.values()), 1.0)


def test_thresholded_prefix_search_matches_exhaustive_marginal():
    rng = np.random.default_rng(812)
    state = rng.normal(size=32) + 1j * rng.normal(size=32)
    state /= np.linalg.norm(state)
    mps = _statevector_to_mps(state, 5)
    exhaustive = mps_configuration_probabilities(mps, range(4))
    epsilon = 0.07
    expected = {
        bits: probability
        for bits, probability in exhaustive.items()
        if probability > epsilon
    }
    actual = mps_prefix_configurations_above_probability(
        mps, n_prefix_sites=4, epsilon=epsilon
    )
    assert actual.keys() == expected.keys()
    assert all(np.isclose(actual[key], expected[key]) for key in expected)


def test_thresholded_prefix_search_prunes_zero_probability_subtrees():
    state = np.zeros(1 << 8)
    state[int("00000000", 2)] = np.sqrt(0.6)
    state[int("11100011", 2)] = np.sqrt(0.4)
    result = mps_prefix_configurations_above_probability(
        _statevector_to_mps(state, 8),
        n_prefix_sites=5,
        epsilon=0.1,
    )
    assert result == pytest.approx({"00000": 0.6, "11100": 0.4})


def test_probability_mass_search_bounds_total_omitted_probability():
    state = np.zeros(1 << 8)
    state[int("00000000", 2)] = np.sqrt(0.6)
    state[int("11100011", 2)] = np.sqrt(0.4)
    result, metadata = mps_prefix_configurations_for_probability_mass(
        _statevector_to_mps(state, 8),
        n_prefix_sites=5,
        max_omitted_probability=0.41,
    )
    assert result == pytest.approx({"00000": 0.6})
    assert metadata["omitted_probability_bound"] == pytest.approx(0.4)
    assert metadata["retained_probability"] == pytest.approx(0.6)


def test_exact_prefix_trie_projection_matches_dense_projection():
    rng = np.random.default_rng(114)
    state = rng.normal(size=64) + 1j * rng.normal(size=64)
    state /= np.linalg.norm(state)
    configurations = {"000", "101", "111"}
    expected = state.reshape(8, 8).copy()
    for index in range(8):
        if format(index, "03b") not in configurations:
            expected[index] = 0
    probability = float(np.vdot(expected, expected).real)
    expected = expected.reshape(-1) / np.sqrt(probability)

    projected, metadata = project_mps_onto_prefix_configurations(
        _statevector_to_mps(state, 6),
        configurations,
        n_prefix_sites=3,
    )
    actual = _mps_to_statevector(projected)
    assert np.allclose(actual, expected)
    assert metadata["retained_probability"] == pytest.approx(probability)
    assert metadata["svd_compression_used"] is False
    assert metadata["bond_dimension_cap"] is None
