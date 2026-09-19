"""Adaptive searches for the smallest sufficient DMRG bond dimension."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def next_binary_bond_dimension(
    completed_rows: Sequence[Mapping],
    minimum_bond_dimension: int,
    maximum_bond_dimension: int,
    *,
    acceptance_key: str = "accepted_converged_bond_dimension",
) -> int | None:
    """Return the next integer bond dimension in a midpoint-first search.

    Parameters
    ----------
    completed_rows
        DMRG result records from earlier points in the same search. Each row
        must contain an integer ``bond_dim`` and a Boolean value under
        ``acceptance_key``. A value is accepted only when the complete
        scientific convergence criterion has passed.
    minimum_bond_dimension, maximum_bond_dimension
        Inclusive integer search interval. The midpoint is tested first. A
        failed midpoint removes it and every smaller value; a successful
        midpoint removes every larger value while retaining that midpoint as
        the current upper bound. The remaining integer interval is repeatedly
        bisected until no untested candidate remains.
    acceptance_key
        Result-row key defining whether a bond dimension is sufficient. The
        default requires the combined energy-accuracy and DMRG sweep-
        convergence flag used by the benchmark code.

    Returns
    -------
    bond_dimension
        Next untested integer bond dimension, or ``None`` when the minimum
        sufficient integer has been identified or the maximum has failed.

    Notes
    -----
    Binary search assumes sufficiency is monotone in the allowed maximum bond
    dimension. DMRG optimization can violate this assumption because of local
    convergence behavior. A detected success below a failure therefore raises
    an error instead of silently reporting a scientifically ambiguous result.
    Previously completed in-range points may be supplied in any order, making
    interrupted searches resumable.
    """
    minimum = int(minimum_bond_dimension)
    maximum = int(maximum_bond_dimension)
    if minimum < 1:
        raise ValueError("minimum_bond_dimension must be positive")
    if maximum < minimum:
        raise ValueError(
            "maximum_bond_dimension must not be smaller than the minimum"
        )

    outcomes: dict[int, bool] = {}
    for row in completed_rows:
        bond_dimension = int(row["bond_dim"])
        if not minimum <= bond_dimension <= maximum:
            raise ValueError(
                f"completed bond dimension {bond_dimension} lies outside "
                f"the search interval [{minimum}, {maximum}]"
            )
        if bond_dimension in outcomes:
            raise ValueError(
                f"duplicate completed bond dimension {bond_dimension}"
            )
        accepted = row.get(acceptance_key)
        if not isinstance(accepted, bool):
            raise ValueError(
                f"{acceptance_key!r} must be Boolean for bond dimension "
                f"{bond_dimension}; got {accepted!r}"
            )
        outcomes[bond_dimension] = accepted

    successful = sorted(
        bond_dimension
        for bond_dimension, accepted in outcomes.items()
        if accepted
    )
    failed = sorted(
        bond_dimension
        for bond_dimension, accepted in outcomes.items()
        if not accepted
    )
    if successful and failed and min(successful) < max(failed):
        raise RuntimeError(
            "DMRG acceptance is nonmonotone: a smaller bond dimension passed "
            "while a larger one failed. Binary search cannot determine a "
            "reliable threshold from these results."
        )

    lower_candidate = minimum
    if failed:
        lower_candidate = max(failed) + 1
    upper_candidate = maximum
    if successful:
        upper_candidate = min(successful) - 1
    if lower_candidate > upper_candidate:
        return None
    initial_midpoint = (minimum + maximum) // 2
    if (
        lower_candidate <= initial_midpoint <= upper_candidate
        and initial_midpoint not in outcomes
    ):
        return initial_midpoint
    midpoint = (lower_candidate + upper_candidate) // 2
    if midpoint in outcomes:
        raise RuntimeError(
            "binary-search midpoint was already evaluated; saved results do "
            "not define a valid search bracket"
        )
    return midpoint


def lowest_accepted_bond_dimension(
    completed_rows: Sequence[Mapping],
    *,
    acceptance_key: str = "accepted_converged_bond_dimension",
) -> int | None:
    """Return the smallest accepted bond dimension in completed DMRG rows.

    Parameters
    ----------
    completed_rows
        Result rows containing ``bond_dim`` and the Boolean acceptance field.
    acceptance_key
        Field defining whether the corresponding calculation was sufficient.

    Returns
    -------
    bond_dimension
        Smallest accepted dimension, or ``None`` if no row was accepted.
    """
    accepted = [
        int(row["bond_dim"])
        for row in completed_rows
        if row.get(acceptance_key) is True
    ]
    return min(accepted, default=None)
