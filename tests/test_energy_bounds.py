import numpy as np
import pytest
from openfermion import QubitOperator

from quasisymmetries.bs.utils import PauliTermStream
from quasisymmetries.energy_bounds import (
    anticommuting_grouped_pauli_norm,
    maximum_omitted_probability_for_block_energy_error,
    mutually_anticommuting_pauli_groups,
    symmetry_block_anticommuting_pauli_norms,
    symmetry_block_energy_error_bound,
    symmetry_block_pauli_l1_norms,
)


def test_symmetry_block_l1_classification_uses_x_support():
    operator = (
        QubitOperator((), 4.0)
        + QubitOperator("Z0 Z2", -0.7)
        + QubitOperator("X0", 0.2)
        + QubitOperator("Y1 Z2", -0.3)
        + QubitOperator("X2", 0.5)
    )
    stream = PauliTermStream.from_qubit_operator(operator, n_qubits=3)
    diagonal, off_diagonal = symmetry_block_pauli_l1_norms(
        stream, symmetry_sites=(0, 1)
    )
    # Identity is excluded; X/Y on sites 0 or 1 changes the block. X2 does not.
    assert diagonal == pytest.approx(1.2)
    assert off_diagonal == pytest.approx(0.5)


def test_maximum_omitted_probability_saturates_bound():
    diagonal = 2.5
    off_diagonal = 0.4
    tolerance = 1.6e-3
    delta = maximum_omitted_probability_for_block_energy_error(
        diagonal, off_diagonal, tolerance
    )
    assert 0 < delta < 0.5
    assert symmetry_block_energy_error_bound(
        delta, diagonal, off_diagonal
    ) <= tolerance
    assert symmetry_block_energy_error_bound(
        np.nextafter(delta, 1.0), diagonal, off_diagonal
    ) <= tolerance + 1e-12


def test_probability_cap_is_returned_when_already_safe():
    assert maximum_omitted_probability_for_block_energy_error(
        0.0, 0.0, 1.6e-3, probability_cap=0.25
    ) == pytest.approx(0.25)


def test_mutually_anticommuting_group_has_euclidean_norm():
    operator = (
        QubitOperator("X0", 3.0)
        + QubitOperator("Y0", 4.0)
        + QubitOperator("Z0", 12.0)
    )
    stream = PauliTermStream.from_qubit_operator(operator, n_qubits=1)
    bound, groups, group_norms = anticommuting_grouped_pauli_norm(
        stream.terms, return_groups=True
    )
    assert len(groups) == 1
    assert group_norms == pytest.approx((13.0,))
    assert bound == pytest.approx(13.0)


def test_commuting_terms_cannot_share_anticommuting_group():
    operator = QubitOperator("X0", 3.0) + QubitOperator("X1", 4.0)
    stream = PauliTermStream.from_qubit_operator(operator, n_qubits=2)
    groups = mutually_anticommuting_pauli_groups(stream.terms)
    assert len(groups) == 2
    assert anticommuting_grouped_pauli_norm(stream.terms) == pytest.approx(7.0)


def test_block_grouped_norms_tighten_ungrouped_bounds():
    operator = (
        QubitOperator("X0", 3.0)
        + QubitOperator("Y0", 4.0)
        + QubitOperator("Z0 X1", 5.0)
        + QubitOperator("Z0 Y1", 12.0)
    )
    stream = PauliTermStream.from_qubit_operator(operator, n_qubits=2)
    diagonal, off_diagonal, diagnostics = (
        symmetry_block_anticommuting_pauli_norms(
            stream,
            symmetry_sites=(0,),
            return_diagnostics=True,
        )
    )
    # X0/Y0 change the symmetry bit and anticommute: sqrt(3^2 + 4^2) = 5.
    assert off_diagonal == pytest.approx(5.0)
    # Z0X1/Z0Y1 preserve it and anticommute: sqrt(5^2 + 12^2) = 13.
    assert diagonal == pytest.approx(13.0)
    assert diagnostics["off_diagonal_ungrouped_pauli_l1"] == pytest.approx(7.0)
    assert diagnostics["diagonal_ungrouped_pauli_l1"] == pytest.approx(17.0)
