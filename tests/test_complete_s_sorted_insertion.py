import pytest
from openfermion import QubitOperator

from quasisymmetries.pt import complete_S_sorted_insertion


def test_inserts_by_coefficient_while_enforcing_pairwise_commutation():
    hamiltonian = (
        10.0 * QubitOperator(())
        + 5.0 * QubitOperator("X0")
        + 4.0 * QubitOperator("Z0")
        + 3.0 * QubitOperator("Z1")
    )

    result = complete_S_sorted_insertion(
        [],
        hamiltonian,
        n_qubits=2,
        target_rank=2,
    )

    # Z0 is stronger than Z1 but anticommutes with the already selected X0.
    assert result == [QubitOperator("X0"), QubitOperator("Z1")]


def test_skips_dependent_terms_and_normalizes_inserted_coefficients():
    initial = [-QubitOperator("Z0")]
    original = list(initial)
    hamiltonian = (
        9.0 * QubitOperator("Z0")
        - 4.5 * QubitOperator("Z1")
    )

    result = complete_S_sorted_insertion(
        initial,
        hamiltonian,
        n_qubits=2,
        target_rank=2,
    )

    assert initial == original
    assert result[0] == -QubitOperator("Z0")
    assert result[1] == QubitOperator("Z1")


def test_skips_products_dependent_on_generators_selected_so_far():
    initial = [QubitOperator("Z0")]
    hamiltonian = (
        5.0 * QubitOperator("Z1")
        + 4.0 * QubitOperator("Z0 Z1")
        + 3.0 * QubitOperator("X2")
    )

    result = complete_S_sorted_insertion(
        initial,
        hamiltonian,
        n_qubits=3,
        target_rank=3,
    )

    assert result == [
        QubitOperator("Z0"),
        QubitOperator("Z1"),
        QubitOperator("X2"),
    ]


def test_default_target_rank_and_insufficient_candidates(capsys):
    result = complete_S_sorted_insertion(
        [],
        2.0 * QubitOperator("Z0"),
        n_qubits=2,
    )

    assert result == [QubitOperator("Z0")]
    assert "Insufficient generators identified:  1" in capsys.readouterr().out


def test_rejects_invalid_initial_generator_sets():
    with pytest.raises(ValueError, match="mutually commute"):
        complete_S_sorted_insertion(
            [QubitOperator("X0"), QubitOperator("Z0")],
            QubitOperator("Z1"),
            n_qubits=2,
        )

    with pytest.raises(ValueError, match="independent"):
        complete_S_sorted_insertion(
            [QubitOperator("Z0"), -QubitOperator("Z0")],
            QubitOperator("Z1"),
            n_qubits=2,
        )

    with pytest.raises(ValueError, match="identity"):
        complete_S_sorted_insertion(
            [QubitOperator(())],
            QubitOperator("Z0"),
            n_qubits=1,
        )


@pytest.mark.parametrize("target_rank", [-1, 3])
def test_rejects_invalid_target_rank(target_rank):
    with pytest.raises(ValueError, match="target_rank"):
        complete_S_sorted_insertion(
            [],
            QubitOperator("Z0"),
            n_qubits=2,
            target_rank=target_rank,
        )
