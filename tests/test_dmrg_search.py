import pytest

from quasisymmetries.dmrg_search import next_binary_bond_dimension


def _row(bond_dimension, accepted):
    return {
        "bond_dim": bond_dimension,
        "accepted_converged_bond_dimension": accepted,
    }


def test_binary_search_finds_exact_integer_threshold():
    rows = []
    expected_sequence = [55, 78, 66, 60, 63, 61, 62]
    for expected in expected_sequence:
        assert next_binary_bond_dimension(rows, 10, 100) == expected
        rows.append(_row(expected, expected >= 62))
    assert next_binary_bond_dimension(rows, 10, 100) is None


def test_binary_search_handles_endpoint_outcomes():
    assert next_binary_bond_dimension([], 10, 100) == 55
    assert next_binary_bond_dimension([_row(10, True)], 10, 100) is None
    assert next_binary_bond_dimension([_row(10, False)], 10, 100) == 55
    assert (
        next_binary_bond_dimension(
            [_row(10, False), _row(100, False)], 10, 100
        )
        is None
    )


def test_midpoint_first_resume_reuses_endpoint_observation():
    assert next_binary_bond_dimension([_row(1, False)], 1, 200) == 100
    assert (
        next_binary_bond_dimension(
            [_row(1, False), _row(100, False)], 1, 200
        )
        == 150
    )
    assert (
        next_binary_bond_dimension(
            [_row(1, False), _row(100, True)], 1, 200
        )
        == 50
    )


def test_binary_search_rejects_nonmonotone_results():
    with pytest.raises(RuntimeError, match="nonmonotone"):
        next_binary_bond_dimension(
            [_row(20, True), _row(30, False)], 10, 100
        )
