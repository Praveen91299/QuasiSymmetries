"""Energy-error bounds for truncating Clifford symmetry configurations."""

from __future__ import annotations

import numpy as np

from .bs.utils import as_pauli_term_stream, symplectic_commutes


def _validated_symmetry_mask(stream, symmetry_sites):
    sites = tuple(int(site) for site in symmetry_sites)
    if len(set(sites)) != len(sites):
        raise ValueError("symmetry_sites contains duplicates")
    if any(site < 0 or site >= stream.n_qubits for site in sites):
        raise ValueError("symmetry_sites contains an out-of-range site")
    return sum(1 << site for site in sites)


def _symmetry_block_terms(hamiltonian, symmetry_sites):
    stream = as_pauli_term_stream(hamiltonian)
    symmetry_mask = _validated_symmetry_mask(stream, symmetry_sites)
    diagonal_terms = []
    off_diagonal_terms = []
    for term in stream.terms:
        if term.mask == (0, 0):
            continue
        target = (
            off_diagonal_terms
            if term.mask[0] & symmetry_mask
            else diagonal_terms
        )
        target.append(term)
    return diagonal_terms, off_diagonal_terms


def mutually_anticommuting_pauli_groups(terms, *, tolerance=1e-10):
    """Greedily partition weighted Pauli terms into anticommuting groups.

    Parameters
    ----------
    terms
        Iterable of ``WeightedTerm`` objects representing nonzero Pauli terms.
    tolerance
        Maximum allowed imaginary coefficient magnitude; the anticommuting
        norm identity assumes a Hermitian expansion with real coefficients.

    Returns
    -------
    groups
        Tuple of groups, each itself a tuple of ``WeightedTerm`` objects. Every
        pair of masks within one group anticommutes.

    Notes
    -----
    Terms are considered in descending coefficient magnitude.  Among all
    compatible groups, a term is placed where it causes the smallest increase
    in the sum of group 2-norms.  The partition is deterministic but heuristic;
    finding an optimal weighted anticommuting clique cover is combinatorial.
    """
    ordered = sorted(
        tuple(terms),
        key=lambda term: (-term.abs_coeff, term.mask[0], term.mask[1]),
    )
    groups = []
    squared_norms = []
    for term in ordered:
        if abs(term.signed_coefficient.imag) > tolerance:
            raise ValueError(
                "anticommuting Pauli grouping requires a Hermitian expansion "
                "with real coefficients"
            )
        compatible = [
            index
            for index, group in enumerate(groups)
            if all(
                not symplectic_commutes(term.mask, member.mask)
                for member in group
            )
        ]
        if compatible:
            coefficient_squared = term.abs_coeff**2
            group_index = min(
                compatible,
                key=lambda index: (
                    np.sqrt(squared_norms[index] + coefficient_squared)
                    - np.sqrt(squared_norms[index]),
                    index,
                ),
            )
            groups[group_index].append(term)
            squared_norms[group_index] += coefficient_squared
        else:
            groups.append([term])
            squared_norms.append(term.abs_coeff**2)
    return tuple(tuple(group) for group in groups)


def anticommuting_grouped_pauli_norm(terms, *, return_groups=False):
    """Bound an operator norm using mutually anticommuting Pauli groups.

    Parameters
    ----------
    terms
        Iterable of ``WeightedTerm`` Pauli terms to group.
    return_groups
        If false, return only the scalar norm bound. If true, additionally
        return the term groups and individual group norms.

    Returns
    -------
    bound or (bound, groups, group_norms)
        ``bound`` is the sum of group Euclidean coefficient norms. ``groups``
        contains the grouped terms, and ``group_norms[g]`` equals
        ``sqrt(sum_j |h_j|**2)`` for group ``g``.
    """
    groups = mutually_anticommuting_pauli_groups(terms)
    group_norms = tuple(
        float(np.sqrt(sum(term.abs_coeff**2 for term in group)))
        for group in groups
    )
    bound = float(sum(group_norms))
    if return_groups:
        return bound, groups, group_norms
    return bound


def symmetry_block_anticommuting_pauli_norms(
    hamiltonian,
    symmetry_sites,
    *,
    return_diagnostics=False,
):
    """Return grouped norm bounds for block-preserving/changing terms.

    Parameters
    ----------
    hamiltonian
        ``PauliTermStream`` or OpenFermion ``QubitOperator`` to classify.
    symmetry_sites
        Qubit sites whose computational-basis bit strings label the retained
        configurations.
    return_diagnostics
        If true, also return the explicit groups and ungrouped Pauli 1-norms.

    Returns
    -------
    diagonal_bound, off_diagonal_bound[, diagnostics]
        Grouped operator-norm upper bounds for terms that preserve or change
        the symmetry-site bit string. The optional diagnostics dictionary
        describes every group and compares grouped and ungrouped bounds.

    Notes
    -----
    These values can be passed directly as the diagonal and off-diagonal norm
    arguments to :func:`maximum_omitted_probability_for_block_energy_error`.
    """
    diagonal_terms, off_diagonal_terms = _symmetry_block_terms(
        hamiltonian, symmetry_sites
    )
    diagonal, diagonal_groups, diagonal_group_norms = (
        anticommuting_grouped_pauli_norm(
            diagonal_terms, return_groups=True
        )
    )
    off_diagonal, off_diagonal_groups, off_diagonal_group_norms = (
        anticommuting_grouped_pauli_norm(
            off_diagonal_terms, return_groups=True
        )
    )
    if not return_diagnostics:
        return diagonal, off_diagonal

    def describe(groups, norms):
        return [
            {
                "group_norm": norm,
                "term_count": len(group),
                "terms": [
                    {
                        "x_mask": int(term.mask[0]),
                        "z_mask": int(term.mask[1]),
                        "coefficient": float(term.signed_coefficient.real),
                    }
                    for term in group
                ],
            }
            for group, norm in zip(groups, norms)
        ]

    diagnostics = {
        "method": "largest_coefficient_first_weighted_best_fit",
        "diagonal_ungrouped_pauli_l1": float(
            sum(term.abs_coeff for term in diagonal_terms)
        ),
        "off_diagonal_ungrouped_pauli_l1": float(
            sum(term.abs_coeff for term in off_diagonal_terms)
        ),
        "diagonal_grouped_bound": diagonal,
        "off_diagonal_grouped_bound": off_diagonal,
        "diagonal_groups": describe(diagonal_groups, diagonal_group_norms),
        "off_diagonal_groups": describe(
            off_diagonal_groups, off_diagonal_group_norms
        ),
    }
    return diagonal, off_diagonal, diagnostics


def symmetry_block_pauli_l1_norms(
    hamiltonian,
    symmetry_sites,
) -> tuple[float, float]:
    """Return block-preserving and block-changing Pauli coefficient norms.

    Parameters
    ----------
    hamiltonian
        ``PauliTermStream`` or OpenFermion ``QubitOperator`` to classify.
    symmetry_sites
        Qubit sites defining the computational-basis symmetry configurations.

    Returns
    -------
    diagonal_l1, off_diagonal_l1
        Sums of absolute coefficients for block-preserving and block-changing
        nonidentity Pauli terms, respectively.

    Notes
    -----
    A Pauli term changes a computational-basis configuration on the selected
    symmetry sites iff it contains X or Y on at least one of those sites.  The
    scalar identity is omitted from the block-preserving norm because it
    cancels from normalized-state energy differences.
    """
    diagonal_terms, off_diagonal_terms = _symmetry_block_terms(
        hamiltonian, symmetry_sites
    )
    return (
        float(sum(term.abs_coeff for term in diagonal_terms)),
        float(sum(term.abs_coeff for term in off_diagonal_terms)),
    )


def symmetry_block_energy_error_bound(
    omitted_probability: float,
    diagonal_pauli_l1: float,
    off_diagonal_pauli_l1: float,
) -> float:
    """Evaluate the rigorous block-aware Pauli-L1 energy bound.

    Parameters
    ----------
    omitted_probability
        Total probability ``delta`` removed by the normalized projection.
    diagonal_pauli_l1
        Operator-norm upper bound for the block-preserving Hamiltonian part;
        this may be an ungrouped Pauli 1-norm or tighter grouped bound.
    off_diagonal_pauli_l1
        Operator-norm upper bound for the block-changing Hamiltonian part.

    Returns
    -------
    error_bound
        Upper bound ``2*diagonal*delta + 2*off_diagonal*sqrt(delta)`` on the
        energy change between the original and normalized projected states.

    Notes
    -----
    The block-preserving contribution scales as ``delta`` because it has no
    retained/discarded cross matrix elements.  The inexpensive X/Y-support
    classification does not imply that the block-changing part is purely a
    retained/discarded cross operator, so its safe trace-distance contribution
    is ``2 * Λoff * sqrt(delta)``.
    """
    delta = float(omitted_probability)
    diagonal = float(diagonal_pauli_l1)
    off_diagonal = float(off_diagonal_pauli_l1)
    if not 0 <= delta <= 1:
        raise ValueError("omitted_probability must lie in [0, 1]")
    if diagonal < 0 or off_diagonal < 0:
        raise ValueError("Pauli 1-norms must be nonnegative")
    return float(
        2.0 * diagonal * delta
        + 2.0 * off_diagonal * np.sqrt(delta)
    )


def maximum_omitted_probability_for_block_energy_error(
    diagonal_pauli_l1: float,
    off_diagonal_pauli_l1: float,
    energy_tolerance: float,
    *,
    probability_cap: float = 0.5,
    bisection_tolerance: float = 1e-15,
) -> float:
    """Solve the block-aware bound for a safe omitted probability.

    Parameters
    ----------
    diagonal_pauli_l1
        Operator-norm upper bound for the block-preserving Hamiltonian part.
    off_diagonal_pauli_l1
        Operator-norm upper bound for the block-changing Hamiltonian part.
    energy_tolerance
        Maximum allowed energy change in the Hamiltonian's energy units.
    probability_cap
        Maximum omitted probability the solver may return.
    bisection_tolerance
        Absolute stopping tolerance for the scalar probability bisection.

    Returns
    -------
    maximum_omitted_probability
        Largest ``delta`` within ``[0, probability_cap]`` whose error bound is
        no greater than ``energy_tolerance``.

    Notes
    -----
    The returned value is the largest allowed ``delta`` in
    ``[0, probability_cap]``.  The default cap selects the physically relevant
    monotone branch of the bound and prevents a truncation from discarding a
    majority of the probability even in unusual limiting cases.
    """
    diagonal = float(diagonal_pauli_l1)
    off_diagonal = float(off_diagonal_pauli_l1)
    tolerance = float(energy_tolerance)
    probability_cap = float(probability_cap)
    if diagonal < 0 or off_diagonal < 0:
        raise ValueError("Pauli 1-norms must be nonnegative")
    if tolerance <= 0:
        raise ValueError("energy_tolerance must be positive")
    if not 0 < probability_cap <= 0.5:
        raise ValueError("probability_cap must lie in (0, 0.5]")
    if bisection_tolerance <= 0:
        raise ValueError("bisection_tolerance must be positive")

    def bound(delta):
        return symmetry_block_energy_error_bound(
            delta, diagonal, off_diagonal
        )

    if bound(probability_cap) <= tolerance:
        return probability_cap

    low = 0.0
    high = probability_cap
    while high - low > bisection_tolerance:
        middle = 0.5 * (low + high)
        if bound(middle) <= tolerance:
            low = middle
        else:
            high = middle
    return low
