from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openfermion import QubitOperator
from ..gf2_utils import (
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
    term: Tuple[Tuple[int, str], ...] = ()
    coefficient: complex | None = None

    @property
    def signed_coefficient(self) -> complex:
        """Return the physical coefficient when it is available.

        Older Beam-search callers only supplied ``abs_coeff``.  Keeping that
        field preserves their inexpensive scoring representation, while the
        streamed Hamiltonian path additionally retains signs and phases.
        """
        if self.coefficient is None:
            return complex(self.abs_coeff)
        return complex(self.coefficient)


@dataclass(frozen=True)
class PauliTermStream:
    """Packed Pauli Hamiltonian without an OpenFermion operator dictionary.

    Pauli words use the existing ``(x_mask, z_mask)`` convention.  Terms are
    combined by mask on construction and include the identity, so this form is
    suitable both for symmetry search and direct pyblock2 MPO construction.
    """

    n_qubits: int
    terms: Tuple[WeightedTerm, ...]

    @classmethod
    def from_terms(
        cls,
        n_qubits: int,
        terms: Iterable[tuple[PauliMask, complex]],
        *,
        tolerance: float = 0.0,
    ) -> "PauliTermStream":
        n_qubits = int(n_qubits)
        if n_qubits < 0:
            raise ValueError("n_qubits must be nonnegative")
        limit = (1 << n_qubits) - 1
        combined: Dict[PauliMask, complex] = {}
        for mask, coefficient in terms:
            x, z = int(mask[0]), int(mask[1])
            if x < 0 or z < 0 or (x | z) & ~limit:
                raise ValueError(f"Pauli mask {(x, z)} exceeds n_qubits={n_qubits}")
            combined[(x, z)] = combined.get((x, z), 0.0j) + complex(coefficient)
        packed = []
        for mask, coefficient in combined.items():
            if abs(coefficient) <= tolerance:
                continue
            packed.append(
                WeightedTerm(
                    mask=mask,
                    abs_coeff=float(abs(coefficient)),
                    coefficient=coefficient,
                )
            )
        return cls(n_qubits=n_qubits, terms=tuple(packed))

    @classmethod
    def from_qubit_operator(
        cls,
        op: QubitOperator,
        n_qubits: Optional[int] = None,
        *,
        tolerance: float = 0.0,
    ) -> "PauliTermStream":
        if n_qubits is None:
            n_qubits = infer_n_qubits(op)
        return cls.from_terms(
            n_qubits,
            (
                (term_to_masks(term, n_qubits), complex(coefficient))
                for term, coefficient in op.terms.items()
            ),
            tolerance=tolerance,
        )

    def without_identity(self) -> List[WeightedTerm]:
        return [term for term in self.terms if term.mask != (0, 0)]

    def truncated(self, threshold: float) -> "PauliTermStream":
        return PauliTermStream(
            self.n_qubits,
            tuple(term for term in self.terms if term.abs_coeff >= threshold),
        )

    def to_qubit_operator(self) -> QubitOperator:
        """Materialize an OpenFermion operator only at compatibility boundaries."""
        op = QubitOperator()
        for item in self.terms:
            op += QubitOperator(
                masks_to_term(item.mask, self.n_qubits),
                item.signed_coefficient,
            )
        op.compress()
        return op


def as_pauli_term_stream(
    operator,
    n_qubits: Optional[int] = None,
) -> PauliTermStream:
    """Return a packed Pauli stream for a supported Hamiltonian object.

    Parameters
    ----------
    operator
        Existing ``PauliTermStream`` or OpenFermion ``QubitOperator``.
    n_qubits
        Optional required qubit count. It is used during conversion and checked
        against an existing stream.

    Returns
    -------
    stream
        ``PauliTermStream`` preserving identity terms and signed coefficients.
    """
    if isinstance(operator, PauliTermStream):
        if n_qubits is not None and int(n_qubits) != operator.n_qubits:
            raise ValueError(
                f"stream has {operator.n_qubits} qubits, requested {n_qubits}"
            )
        return operator
    if isinstance(operator, QubitOperator):
        return PauliTermStream.from_qubit_operator(operator, n_qubits)
    raise TypeError(
        "expected an OpenFermion QubitOperator or PauliTermStream, got "
        f"{type(operator).__name__}"
    )


def pauli_stream_l1_norm(operator, *, include_identity: bool = True) -> float:
    """Return the coefficient 1-norm of a packed Pauli expansion.

    Parameters
    ----------
    operator
        ``PauliTermStream`` or OpenFermion ``QubitOperator``.
    include_identity
        Whether the scalar identity coefficient contributes to the sum.

    Returns
    -------
    norm
        Sum of absolute coefficients of the selected Pauli terms.

    Notes
    -----
    Excluding the identity gives a tighter expectation-difference bound,
    because a scalar energy shift cancels between normalized states.
    """
    stream = as_pauli_term_stream(operator)
    return float(
        sum(
            item.abs_coeff
            for item in stream.terms
            if include_identity or item.mask != (0, 0)
        )
    )


def multiply_pauli_masks(
    first: PauliMask,
    second: PauliMask,
) -> tuple[PauliMask, complex]:
    """Multiply two Hermitian Pauli words represented by binary masks.

    Parameters
    ----------
    first, second
        ``(x_mask, z_mask)`` representations of Hermitian Pauli products.

    Returns
    -------
    product_mask, phase
        Mask of the product after extracting its phase, and a phase in
        ``{1, 1j, -1, -1j}`` satisfying ``P(first)P(second)=phase*P(product)``.
    """
    ax, az = first
    bx, bz = second
    out = (ax ^ bx, az ^ bz)
    exponent = (
        popcount(ax & az)
        + popcount(bx & bz)
        - popcount(out[0] & out[1])
        + 2 * popcount(az & bx)
    ) % 4
    return out, (1.0, 1.0j, -1.0, -1.0j)[exponent]


def jordan_wigner_pauli_stream(
    fermion_operator,
    n_qubits: Optional[int] = None,
    *,
    tolerance: float = 1e-12,
) -> PauliTermStream:
    """Jordan--Wigner map directly into packed masks and coefficients.

    Parameters
    ----------
    fermion_operator
        OpenFermion ``FermionOperator`` to transform.
    n_qubits
        Number of Jordan--Wigner modes/qubits. If omitted, it is inferred from
        the largest occupied mode index.
    tolerance
        Combined Pauli coefficients with magnitude at or below this value are
        removed.

    Returns
    -------
    stream
        Combined ``PauliTermStream`` containing the Jordan--Wigner image.

    Notes
    -----
    This avoids constructing OpenFermion ``QubitOperator`` term objects.  A
    coefficient dictionary is still used to combine equal Pauli words, which
    is necessary before coefficient-threshold HCT and MPO compression.
    """
    if n_qubits is None:
        n_qubits = 0
        for term in fermion_operator.terms:
            for mode, _action in term:
                n_qubits = max(n_qubits, int(mode) + 1)
    n_qubits = int(n_qubits)
    output: Dict[PauliMask, complex] = {}
    for fermion_term, fermion_coefficient in fermion_operator.terms.items():
        expansion: Dict[PauliMask, complex] = {
            (0, 0): complex(fermion_coefficient)
        }
        for mode, action in fermion_term:
            mode = int(mode)
            if not 0 <= mode < n_qubits:
                raise ValueError("fermionic mode exceeds n_qubits")
            if action not in (0, 1):
                raise ValueError(f"invalid ladder action {action!r}")
            z_prefix = (1 << mode) - 1
            ladder_terms = (
                ((1 << mode, z_prefix), 0.5),
                (
                    (1 << mode, z_prefix | (1 << mode)),
                    -0.5j if action == 1 else 0.5j,
                ),
            )
            updated: Dict[PauliMask, complex] = {}
            for left_mask, left_coefficient in expansion.items():
                for right_mask, right_coefficient in ladder_terms:
                    mask, phase = multiply_pauli_masks(left_mask, right_mask)
                    updated[mask] = updated.get(mask, 0.0j) + (
                        left_coefficient * right_coefficient * phase
                    )
            expansion = updated
        for mask, coefficient in expansion.items():
            output[mask] = output.get(mask, 0.0j) + coefficient
    return PauliTermStream.from_terms(
        n_qubits,
        output.items(),
        tolerance=tolerance,
    )


def qubit_operator_terms(
    op,
    n_qubits: Optional[int] = None,
) -> Tuple[int, List[WeightedTerm]]:
    if isinstance(op, PauliTermStream):
        stream = as_pauli_term_stream(op, n_qubits)
        return stream.n_qubits, stream.without_identity()

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

        terms.append(
            WeightedTerm(
                mask=mask,
                abs_coeff=w,
                term=term,
                coefficient=c,
            )
        )

    return n_qubits, terms

def terms_to_HQ(terms):
    """Compatibility conversion; prefer :class:`PauliTermStream`."""
    op = QubitOperator()
    for t in terms:
        term = t.term
        if not term and t.mask != (0, 0):
            n_qubits = max(t.mask[0].bit_length(), t.mask[1].bit_length())
            term = masks_to_term(t.mask, n_qubits)
        op += QubitOperator(term, t.signed_coefficient)
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
