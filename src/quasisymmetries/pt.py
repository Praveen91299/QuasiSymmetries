"""
utils for PT based sector identification

"""
from typing import List, Optional, Sequence, Set, Tuple

from openfermion import QubitOperator

from .bs.utils import (
    PauliMask,
    combine_mask,
    symplectic_commutes,
    term_to_masks,
    try_add_to_span,
)
from .metrics import find_commuting_paulis

def coupled_computational_basis_states(
    op: QubitOperator,
    reference_state: Sequence[int],
    include_diagonal: bool = True,
) -> Set[Tuple[int, ...]]:
    """
    Return computational basis states coupled to a reference state by a QubitOperator.

    Each Pauli string maps a computational basis state to another basis state by
    flipping exactly the qubits where the Pauli string has X or Y. Z operators
    only contribute phases, so they do not change the bitstring.

    Args:
        op:
            OpenFermion QubitOperator.
        reference_state:
            Computational basis state as a list/tuple of 0s and 1s.
        include_diagonal:
            If True, diagonal Pauli strings with only I/Z operators include the
            unchanged reference state in the returned set.

    Returns:
        Set of coupled basis states as tuples of 0s and 1s. Tuples are used
        because lists cannot be stored in a Python set.
    """
    ref = tuple(reference_state)
    if any(bit not in (0, 1) for bit in ref):
        raise ValueError("reference_state must contain only 0 and 1.")

    coupled_states = set()
    n_qubits = len(ref)

    for term, coeff in op.terms.items():
        if coeff == 0:
            continue

        flipped = list(ref)
        has_flip = False

        for q, pauli in term:
            if q >= n_qubits:
                raise ValueError(
                    f"Pauli term acts on qubit {q}, but reference_state has "
                    f"length {n_qubits}."
                )

            if pauli in ("X", "Y"):
                flipped[q] = 1 - flipped[q]
                has_flip = True
            elif pauli != "Z":
                raise ValueError(f"Unknown Pauli operator {pauli!r} on qubit {q}.")

        if has_flip or include_diagonal:
            coupled_states.add(tuple(flipped))

    return coupled_states

def computational_basis_matrix_element(
    bra_state: Sequence[int],
    op: QubitOperator,
    ket_state: Sequence[int],
) -> complex:
    """
    Compute <bra_state| op |ket_state> without constructing a matrix.

    Args:
        bra_state:
            Computational basis bra as 0/1 bits.
        op:
            OpenFermion QubitOperator.
        ket_state:
            Computational basis ket as 0/1 bits.

    Returns:
        Complex matrix element <bra_state| op |ket_state>.
    """
    bra = tuple(bra_state)
    ket = tuple(ket_state)

    if len(bra) != len(ket):
        raise ValueError("bra_state and ket_state must have the same length.")
    if any(bit not in (0, 1) for bit in bra):
        raise ValueError("bra_state must contain only 0 and 1.")
    if any(bit not in (0, 1) for bit in ket):
        raise ValueError("ket_state must contain only 0 and 1.")

    n_qubits = len(ket)
    matrix_element = 0.0 + 0.0j

    for term, coeff in op.terms.items():
        if coeff == 0:
            continue

        phase = 1.0 + 0.0j
        transformed_ket = list(ket)

        for q, pauli in term:
            if q >= n_qubits:
                raise ValueError(
                    f"Pauli term acts on qubit {q}, but basis states have "
                    f"length {n_qubits}."
                )

            bit = transformed_ket[q]
            if pauli == "X":
                transformed_ket[q] = 1 - bit
            elif pauli == "Y":
                phase *= 1.0j if bit == 0 else -1.0j
                transformed_ket[q] = 1 - bit
            elif pauli == "Z":
                phase *= 1.0 if bit == 0 else -1.0
            else:
                raise ValueError(f"Unknown Pauli operator {pauli!r} on qubit {q}.")

        if tuple(transformed_ket) == bra:
            matrix_element += coeff * phase

    return matrix_element

def complete_S_sorted_insertion(
    symmetries: Sequence[QubitOperator],
    HQ: QubitOperator,
    n_qubits: int,
    target_rank: Optional[int] = None,
) -> List[QubitOperator]:
    """Extend a commuting Pauli basis with large commuting terms from ``HQ``.

    Hamiltonian terms are considered in descending absolute-coefficient order.
    A term is inserted only when it commutes with every generator selected so
    far and increases their binary symplectic GF(2) rank. Inserted terms are
    normalized to coefficient ``+1``.

    The input generators must already be nonidentity, mutually commuting, and
    independent. The input sequence itself is never mutated.
    """
    if n_qubits < 0:
        raise ValueError("n_qubits must be nonnegative.")
    if target_rank is None:
        target_rank = n_qubits
    if not 0 <= target_rank <= n_qubits:
        raise ValueError(
            "target_rank must satisfy 0 <= target_rank <= n_qubits."
        )

    def single_mask(
        operator: QubitOperator,
        *,
        label: str,
    ) -> Tuple[Tuple[Tuple[int, str], ...], PauliMask]:
        if len(operator.terms) != 1:
            raise ValueError(f"{label} must be a single Pauli string.")
        (term, coefficient), = operator.terms.items()
        if abs(complex(coefficient)) <= 1e-12:
            raise ValueError(f"{label} must have a nonzero coefficient.")
        if any(q < 0 or q >= n_qubits for q, _ in term):
            raise ValueError(
                f"{label} acts outside n_qubits={n_qubits}."
            )
        mask = term_to_masks(term, n_qubits)
        if mask == (0, 0):
            raise ValueError(f"{label} cannot be the identity.")
        return term, mask

    extended_symmetries = list(symmetries)
    selected_masks: List[PauliMask] = []
    rref_rows: List[int] = []
    n_bits = 2 * n_qubits

    for index, symmetry in enumerate(extended_symmetries):
        _, mask = single_mask(symmetry, label=f"symmetries[{index}]")
        if not all(
            symplectic_commutes(mask, previous)
            for previous in selected_masks
        ):
            raise ValueError("Input symmetries must mutually commute.")
        new_rows = try_add_to_span(
            combine_mask(mask, n_qubits),
            rref_rows,
            n_bits,
        )
        if new_rows is None:
            raise ValueError("Input symmetries must be independent.")
        rref_rows = new_rows
        selected_masks.append(mask)

    current_rank = len(rref_rows)
    if current_rank >= target_rank:
        return extended_symmetries

    commuting_terms = find_commuting_paulis(
        HQ,
        extended_symmetries,
        verbose=False,
    )
    commuting_terms.sort(
        key=lambda operator: sum(
            abs(coefficient) for coefficient in operator.terms.values()
        ),
        reverse=True,
    )

    for candidate_index, candidate in enumerate(commuting_terms):
        if len(candidate.terms) == 1:
            (candidate_term, _), = candidate.terms.items()
            if not candidate_term:
                continue
        term, mask = single_mask(
            candidate,
            label=f"Hamiltonian candidate {candidate_index}",
        )
        if not all(
            symplectic_commutes(mask, selected)
            for selected in selected_masks
        ):
            continue

        new_rows = try_add_to_span(
            combine_mask(mask, n_qubits),
            rref_rows,
            n_bits,
        )
        if new_rows is None:
            continue

        extended_symmetries.append(QubitOperator(term, 1.0))
        selected_masks.append(mask)
        rref_rows = new_rows
        current_rank += 1
        if current_rank == target_rank:
            return extended_symmetries

    print("Insufficient generators identified: ", current_rank)
    return extended_symmetries
