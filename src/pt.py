"""
utils for PT based sector identification

"""
from typing import Sequence, Set, Tuple
from openfermion import QubitOperator
import numpy as np

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