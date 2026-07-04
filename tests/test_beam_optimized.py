from collections import Counter

import pytest
from openfermion import QubitOperator

from quasisymmetries.bs.beam import (
    BeamSearch_Symmetries,
    beam_search_symmetries,
    build_candidate_pool,
    local_swap_refine,
)
from quasisymmetries.bs.utils import qubit_operator_terms, qubitops_to_masks


def example_problem():
    hamiltonian = (
        QubitOperator("Z0")
        + 0.8 * QubitOperator("X0 X1")
        + 0.7 * QubitOperator("Y1 Y2")
        + 0.6 * QubitOperator("Z2 Z3")
        + 0.5 * QubitOperator("X0 Z2 X3")
    )
    n_qubits, terms = qubit_operator_terms(hamiltonian, 4)
    candidate_pool = build_candidate_pool(
        terms,
        n_qubits,
        include_pairwise_products=True,
    )
    return hamiltonian, n_qubits, candidate_pool


def separable_score(symmetries):
    return -sum(
        sum(q + 1 for term in symmetry.terms for q, _pauli in term)
        for symmetry in symmetries
    )


class CountingSeparableScore:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.calls = Counter()

    def __call__(self, symmetries):
        for mask in qubitops_to_masks(symmetries, self.n_qubits):
            self.calls[mask] += 1
        return separable_score(symmetries)


def run_search(
    *,
    score_is_separable=False,
    n_processes=1,
    score_func=separable_score,
    cache=None,
):
    hamiltonian, n_qubits, candidate_pool = example_problem()
    result = beam_search_symmetries(
        hamiltonian,
        candidate_pool,
        target_rank=4,
        n_qubits=n_qubits,
        beam_width=8,
        score_func=score_func,
        score_is_separable=score_is_separable,
        separable_score_cache=cache,
        n_processes=n_processes,
        mp_start_method="spawn" if n_processes > 1 else None,
    )
    return result, candidate_pool


def test_separable_cache_matches_original_output_and_baseline():
    original, candidate_pool = run_search()
    cache = {}
    optimized, _ = run_search(
        score_is_separable=True,
        cache=cache,
    )

    assert optimized == original
    assert [str(symmetry) for symmetry in optimized] == [
        "1.0 [Z0]",
        "1.0 [X1]",
        "1.0 [X2]",
        "1.0 [X3]",
    ]
    assert set(cache) == set(candidate_pool)


def test_each_pool_singleton_is_evaluated_only_once():
    _hamiltonian, n_qubits, candidate_pool = example_problem()
    counting_score = CountingSeparableScore(n_qubits)
    cache = {}

    result, _ = run_search(
        score_is_separable=True,
        score_func=counting_score,
        cache=cache,
    )

    assert len(result) == 4
    assert set(counting_score.calls) == set(candidate_pool)
    assert all(count == 1 for count in counting_score.calls.values())
    assert sum(counting_score.calls.values()) == len(candidate_pool)


def test_shared_cache_covers_search_and_local_refinement_once():
    hamiltonian, n_qubits, candidate_pool = example_problem()
    counting_score = CountingSeparableScore(n_qubits)
    cache = {}

    symmetries = beam_search_symmetries(
        hamiltonian,
        candidate_pool,
        target_rank=4,
        n_qubits=n_qubits,
        beam_width=8,
        score_func=counting_score,
        score_is_separable=True,
        separable_score_cache=cache,
    )
    refined = local_swap_refine(
        hamiltonian,
        symmetries,
        candidate_pool,
        n_qubits=n_qubits,
        score_func=counting_score,
        score_is_separable=True,
        separable_score_cache=cache,
    )

    assert refined == symmetries
    assert all(count == 1 for count in counting_score.calls.values())
    assert set(candidate_pool).issubset(cache)


@pytest.mark.parametrize("score_is_separable", [False, True])
def test_parallel_extension_matches_serial(score_is_separable):
    serial, _ = run_search(
        score_is_separable=score_is_separable,
        n_processes=1,
    )
    parallel, _ = run_search(
        score_is_separable=score_is_separable,
        n_processes=2,
    )
    assert parallel == serial


def test_parallel_local_refinement_matches_serial():
    hamiltonian, n_qubits, candidate_pool = example_problem()
    symmetries, _ = run_search(score_is_separable=True)

    serial = local_swap_refine(
        hamiltonian,
        symmetries,
        candidate_pool,
        n_qubits=n_qubits,
        score_func=separable_score,
        score_is_separable=True,
        n_processes=1,
    )
    parallel = local_swap_refine(
        hamiltonian,
        symmetries,
        candidate_pool,
        n_qubits=n_qubits,
        score_func=separable_score,
        score_is_separable=True,
        n_processes=2,
        mp_start_method="spawn",
    )
    assert parallel == serial


def test_top_level_optimized_modes_match():
    hamiltonian, _n_qubits, _candidate_pool = example_problem()
    common = dict(
        target_rank=4,
        n_qubits=4,
        beam_width=8,
        include_hct_symmetries=False,
        include_pairwise_products=True,
        do_local_refine=True,
        score_func=separable_score,
        score_is_separable=True,
    )
    serial = BeamSearch_Symmetries(hamiltonian, n_processes=1, **common)
    parallel = BeamSearch_Symmetries(
        hamiltonian,
        n_processes=2,
        mp_start_method="spawn",
        **common,
    )
    assert parallel == serial


def test_invalid_optimization_options_raise():
    hamiltonian, n_qubits, candidate_pool = example_problem()
    with pytest.raises(ValueError, match="requires a score_func"):
        beam_search_symmetries(
            hamiltonian,
            candidate_pool,
            n_qubits=n_qubits,
            score_is_separable=True,
        )
    with pytest.raises(ValueError, match="at least 1"):
        beam_search_symmetries(
            hamiltonian,
            candidate_pool,
            n_qubits=n_qubits,
            n_processes=0,
        )
