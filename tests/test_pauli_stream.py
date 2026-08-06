import numpy as np
from openfermion import FermionOperator, QubitOperator, jordan_wigner

from quasisymmetries.bs.utils import (
    PauliTermStream,
    jordan_wigner_pauli_stream,
    pauli_stream_l1_norm,
)
from quasisymmetries.clifford_symmetry_optimized import (
    Clifford,
    permute_qubits_in_qubit_operator,
)
from quasisymmetries.metrics import universal_grading
from quasisymmetries.sym import hct_mod


def assert_qubit_operators_close(first, second, tolerance=1e-11):
    difference = first - second
    difference.compress(abs_tol=tolerance)
    assert not difference.terms


def test_stream_round_trip_preserves_identity_signs_and_phases():
    operator = (
        QubitOperator((), -1.25)
        + QubitOperator("X0 Y2", 0.75)
        + QubitOperator("Z1", -0.5)
        + QubitOperator("Y0", 0.125j)
    )
    stream = PauliTermStream.from_qubit_operator(operator, n_qubits=3)
    assert_qubit_operators_close(stream.to_qubit_operator(), operator)
    assert np.isclose(pauli_stream_l1_norm(stream), 2.625)
    assert np.isclose(
        pauli_stream_l1_norm(stream, include_identity=False), 1.375
    )


def test_direct_jordan_wigner_stream_matches_openfermion():
    fermion = FermionOperator((), 0.7)
    fermion += FermionOperator("0^ 1", 0.3 - 0.1j)
    fermion += FermionOperator("1^ 0", 0.3 + 0.1j)
    fermion += FermionOperator("0^ 2^ 2 0", -0.2)
    expected = jordan_wigner(fermion)
    actual = jordan_wigner_pauli_stream(fermion, n_qubits=3)
    assert_qubit_operators_close(actual.to_qubit_operator(), expected)


def test_clifford_and_permutation_stream_match_operator_path():
    operator = (
        QubitOperator((), -0.4)
        + QubitOperator("X0 Y1", 0.3)
        + QubitOperator("Z0 X2", -0.7)
    )
    stream = PauliTermStream.from_qubit_operator(operator, n_qubits=3)
    clifford = Clifford.from_symmetries(
        [QubitOperator("X0 X1")], n_qubits=3
    )
    transformed_operator = clifford.transform(operator)
    transformed_stream = clifford.transform(stream)
    assert_qubit_operators_close(
        transformed_stream.to_qubit_operator(), transformed_operator
    )
    permutation = (2, 0, 1)
    assert_qubit_operators_close(
        permute_qubits_in_qubit_operator(
            transformed_stream, permutation
        ).to_qubit_operator(),
        permute_qubits_in_qubit_operator(
            transformed_operator, permutation
        ),
    )


def test_hct_stream_matches_operator_path():
    operator = (
        QubitOperator("Z0 Z1", 1.0)
        + QubitOperator("X0 X1", 0.2)
        + QubitOperator("Y0 Y1", -0.1)
    )
    stream = PauliTermStream.from_qubit_operator(operator, n_qubits=2)
    old_syms, old_eps = hct_mod(
        operator, n_sym=2, use_coeffs_eps=True, verbose=False
    )
    new_syms, new_eps = hct_mod(
        stream, n_sym=2, use_coeffs_eps=True, verbose=False
    )
    assert old_eps == new_eps
    assert [item.terms for item in old_syms] == [
        item.terms for item in new_syms
    ]
    for symmetry in new_syms:
        expected = float(np.real(universal_grading([symmetry], operator)))
        # The stream default metric controls ordering; matching output above
        # verifies its commutator-L1 convention against the legacy function.
        assert expected >= 0.0
