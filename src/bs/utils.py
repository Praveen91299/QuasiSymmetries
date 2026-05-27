from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openfermion import QubitOperator
from src.gf2_utils import (
    gf2_int_in_span,
    gf2_int_msb_pos,
    gf2_int_nullspace_basis,
    gf2_int_reduce_by_rref,
    gf2_int_rref,
    gf2_int_try_add_to_span,
)


# ============================================================
# Basic Pauli / symplectic utilities
# ============================================================

PauliMask = Tuple[int, int]  # (x_mask, z_mask)

def popcount(x: int) -> int:
    """
    Count nonzero bits in bin(x)
    """
    return bin(x).count("1")

def infer_n_qubits(op: QubitOperator) -> int:
    """
    Returns number of qubits, Same as count_qubits
    """
    n = 0
    for term in op.terms:
        for q, _ in term:
            n = max(n, q + 1)
    return n


def term_to_masks(term: Tuple[Tuple[int, str], ...], n_qubits: int) -> PauliMask:
    """
    QubitOperator descriptor to mask
    """
    x = 0
    z = 0
    for q, p in term:
        bit = 1 << q
        if p == "X":
            x ^= bit
        elif p == "Y":
            x ^= bit
            z ^= bit
        elif p == "Z":
            z ^= bit
        else:
            raise ValueError(f"Unsupported Pauli label {p!r}")
    return x, z


def masks_to_term(mask: PauliMask, n_qubits: int) -> Tuple[Tuple[int, str], ...]:
    x, z = mask
    out = []
    for q in range(n_qubits):
        xb = (x >> q) & 1
        zb = (z >> q) & 1
        if xb and zb:
            out.append((q, "Y"))
        elif xb:
            out.append((q, "X"))
        elif zb:
            out.append((q, "Z"))
    return tuple(out)


def mask_to_qubit_operator(mask: PauliMask, n_qubits: int) -> QubitOperator:
    return QubitOperator(masks_to_term(mask, n_qubits), 1.0)


def combine_mask(mask: PauliMask, n_qubits: int) -> int:
    """
    Concatenate x, z bits, as (z|x)

    """
    x, z = mask
    return x | (z << n_qubits)


def split_mask(vec: int, n_qubits: int) -> PauliMask:
    lo = (1 << n_qubits) - 1
    x = vec & lo
    z = vec >> n_qubits
    return x, z


def symplectic_commutes(a: PauliMask, b: PauliMask) -> bool:
    ax, az = a
    bx, bz = b
    return ((popcount(ax & bz) + popcount(az & bx)) & 1) == 0


def pauli_product_mod_phase(a: PauliMask, b: PauliMask) -> PauliMask:
    ax, az = a
    bx, bz = b
    return ax ^ bx, az ^ bz


def pauli_weight(mask: PauliMask) -> int:
    x, z = mask
    return popcount(x | z)

# ============================================================
# Hamiltonian term handling
# ============================================================

@dataclass(frozen=True)
class WeightedTerm:
    mask: PauliMask
    abs_coeff: float
    term: Tuple[Tuple[int, str], ...]


def qubit_operator_terms(
    op: QubitOperator,
    n_qubits: Optional[int] = None,
) -> Tuple[int, List[WeightedTerm]]:
    if n_qubits is None:
        n_qubits = infer_n_qubits(op)

    terms: List[WeightedTerm] = []
    for term, coeff in op.terms.items():
        c = complex(coeff)
        if abs(c.imag) > 1e-12:
            raise ValueError("Hamiltonian coefficients must be real.")
        w = abs(c.real)
        if w <= 0.0:
            continue

        mask = term_to_masks(term, n_qubits)

        # Identity term does not constrain the symmetry search.
        if mask == (0, 0):
            continue

        terms.append(WeightedTerm(mask=mask, abs_coeff=w, term=term))

    return n_qubits, terms

def terms_to_HQ(terms):
    "Takes in weighted turns it into HQ: note that this doesnt get the same H rather it destroys sign and all I paulis"
    op = QubitOperator()
    for t in terms:
        op += QubitOperator(t.term, t.abs_coeff)
    return op

def heavy_core(terms: Sequence[WeightedTerm], fraction: float) -> List[WeightedTerm]:
    """
    Extract from terms, items with abs coeffs upto fraction
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("heavy_core_fraction must be in (0, 1].")

    ordered = sorted(terms, key=lambda t: t.abs_coeff, reverse=True)
    total = sum(t.abs_coeff for t in ordered)
    cutoff = fraction * total

    out = []
    acc = 0.0
    for t in ordered:
        out.append(t)
        acc += t.abs_coeff
        if acc >= cutoff:
            break

    return out

def qubitops_to_masks(
    ops: Sequence[QubitOperator],
    n_qubits: int,
    tol = 1e-12
) -> List[PauliMask]:
    masks: List[PauliMask] = []
    for op in ops:
        if len(op.terms) != 1:
            raise ValueError("Each generator must be a single Pauli string QubitOperator.")
        ((term, coeff),) = op.terms.items()
        if abs(complex(coeff) - 1.0) > tol:
            raise ValueError("Each generator must have coefficient 1.0.")
        masks.append(term_to_masks(term, n_qubits))
    return masks


# ============================================================
# GF(2) linear algebra
# ============================================================

def msb_pos(x: int) -> int:
    """
    Leading bit position, starting count from 0

    """
    return gf2_int_msb_pos(x)


def rref(rows: Sequence[int], n_bits: int) -> Tuple[List[int], Dict[int, int]]:
    """
    Reduced row echelon form over GF(2), represented as ints.
    Returns:
        (rref_rows, pivot_col_to_row_index)
    """
    return gf2_int_rref(rows, n_bits)


def reduce_by_rref(vec: int, rref_rows: Sequence[int]) -> int:
    """
    Remove support of rref_rows in vec, returns remainder

    """
    return gf2_int_reduce_by_rref(vec, rref_rows)


def in_span(vec: int, rref_rows: Sequence[int]) -> bool:
    return gf2_int_in_span(vec, rref_rows)


def try_add_to_span(vec: int, rref_rows: Sequence[int], n_bits: int) -> Optional[List[int]]:
    return gf2_int_try_add_to_span(vec, rref_rows, n_bits)


def nullspace_basis(rows: Sequence[int], n_bits: int) -> List[int]:
    """
    Nullspace of the GF(2) matrix whose rows are `rows`, represented as ints.
    Returns a basis as a list of ints.
    """
    return gf2_int_nullspace_basis(rows, n_bits)

# ============================================================
# Exact Pauli symmetries of the Hamiltonian
# ============================================================

def exact_pauli_symmetry_basis(
    hamiltonian: QubitOperator,
    *,
    n_qubits: Optional[int] = None,
) -> List[QubitOperator]:
    """
    Compute an independent basis of exact Pauli symmetries of H.

    A Pauli g is an exact symmetry if it commutes with every Hamiltonian term.
    If a term has symplectic vector p_i = (x_i | z_i), then the condition
        p_i ⊙ g = 0
    is equivalent to
        (z_i | x_i) · g = 0   over GF(2).

    So the exact symmetry space is the nullspace of the matrix whose rows are
    (z_i | x_i) for all Hamiltonian terms.
    """
    n_qubits, terms = qubit_operator_terms(hamiltonian, n_qubits)
    n_bits = 2 * n_qubits

    if not terms:
        return []

    constraints: List[int] = []
    for t in terms:
        x, z = t.mask
        constraints.append(z | (x << n_qubits))

    basis_vecs = nullspace_basis(constraints, n_bits)

    indep_rows: List[int] = []
    masks: List[PauliMask] = []
    for v in basis_vecs:
        if v == 0:
            continue
        new_rows = try_add_to_span(v, indep_rows, n_bits)
        if new_rows is None:
            continue
        indep_rows = new_rows
        masks.append(split_mask(v, n_qubits))

    return [mask_to_qubit_operator(m, n_qubits) for m in masks]

def complete_basis_any(
    basis: List[PauliMask],
    n_qubits: int,
    target_rank: int,
) -> List[PauliMask]:
    """
    Complete an isotropic basis by using arbitrary directions in S^⊥,
    not restricted to the heuristic candidate pool.
    """
    n_bits = 2 * n_qubits
    current = basis[:]
    rref_rows, _ = rref([combine_mask(g, n_qubits) for g in current], n_bits)

    while len(current) < target_rank:
        constraints = []
        for g in current:
            x, z = g
            constraints.append(z | (x << n_qubits))

        null_basis = nullspace_basis(constraints, n_bits)

        added = False
        for vec in null_basis:
            if vec == 0 or in_span(vec, rref_rows):
                continue
            g = split_mask(vec, n_qubits)
            if all(symplectic_commutes(g, h) for h in current):
                current.append(g)
                rref_rows, _ = rref([combine_mask(h, n_qubits) for h in current], n_bits)
                added = True
                break

        if not added:
            raise RuntimeError("Failed to complete isotropic basis.")

    return current

def bs_tests():
    """
    Some tests
    
    """
    assert popcount(1) == 1
    assert popcount(5) == 2

    op = QubitOperator("X1") + QubitOperator("Y0", -1)
    assert infer_n_qubits(op) == 2

    op = QubitOperator("Y1 X0", 1.0)
    assert term_to_masks(list(op.terms.keys())[0], 2) == (3, 2)

    op = QubitOperator("Y1 X0", 1.0)
    m = term_to_masks(list(op.terms.keys())[0], 2)
    assert mask_to_qubit_operator(m, 2) == op

    a = (1, 0)
    b = (0, 2)
    assert symplectic_commutes(a, b)

    a = (1, 0)
    b = (0, 3)
    assert not symplectic_commutes(a, b)

    op = QubitOperator("X0 Y2", 1.0)
    assert pauli_weight(term_to_masks(list(op.terms.keys())[0], 3)) == 2

    x = 2
    assert msb_pos(x) == 1

    assert rref([3, 1, 1], 2) == ([2, 1], {1: 0, 0: 1})
    assert rref([4, 1], 3) == ([4, 1], {2:0, 0:1})

    return True
