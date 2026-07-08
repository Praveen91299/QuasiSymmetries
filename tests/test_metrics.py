from openfermion import QubitOperator

from quasisymmetries.metrics import find_commuting_paulis


def _sum_paulis(paulis):
    total = QubitOperator.zero()
    for pauli in paulis:
        total += pauli
    total.compress()
    return total


def test_find_commuting_paulis_excludes_constant_and_preserves_coefficients():
    H = (
        7.0 * QubitOperator(())
        + 2.0 * QubitOperator("Z0")
        + 3.0 * QubitOperator("X0")
        - 5.0 * QubitOperator("Z0 Z1")
        + 11.0 * QubitOperator("X1")
    )

    result = find_commuting_paulis(H, [QubitOperator("Z0")])

    assert _sum_paulis(result) == (
        2.0 * QubitOperator("Z0")
        - 5.0 * QubitOperator("Z0 Z1")
        + 11.0 * QubitOperator("X1")
    )


def test_find_commuting_paulis_requires_commutation_with_all_symmetries():
    H = (
        QubitOperator("Z0")
        + 2.0 * QubitOperator("X1")
        + 3.0 * QubitOperator("Z0 Z1")
        + 4.0 * QubitOperator("X0 X1")
    )

    result = find_commuting_paulis(H, [QubitOperator("Z0"), QubitOperator("X1")])

    assert _sum_paulis(result) == (
        QubitOperator("Z0")
        + 2.0 * QubitOperator("X1")
    )


def test_find_commuting_paulis_with_no_symmetries_returns_all_nonconstant_terms():
    H = (
        3.0 * QubitOperator(())
        + QubitOperator("X0")
        - 2.0 * QubitOperator("Y0 Z1")
    )

    result = find_commuting_paulis(H, [])

    assert _sum_paulis(result) == (
        QubitOperator("X0")
        - 2.0 * QubitOperator("Y0 Z1")
    )
