### AI code, UNVERIFIED!!!
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import itertools
import numpy as np
import matplotlib.pyplot as plt

from openfermion import QubitOperator, get_sparse_operator

def check_same_spectrum(
    h1: QubitOperator,
    h2: QubitOperator,
    n_qubits: int,
    atol: float = 1e-10,
) -> bool:
    """
    Compare spectra of two Hermitian QubitOperators.
    """
    m1 = get_sparse_operator(h1, n_qubits=n_qubits).toarray()
    m2 = get_sparse_operator(h2, n_qubits=n_qubits).toarray()

    e1 = np.linalg.eigvalsh(m1)
    e2 = np.linalg.eigvalsh(m2)

    ok = np.allclose(e1, e2, atol=atol, rtol=0.0)
    if not ok:
        print("Max eigval diff:", np.max(np.abs(e1 - e2)))
    return ok


# ============================================================
# Basic Pauli / QubitOperator utilities
# ============================================================

def qubit_operator_num_qubits(op: QubitOperator) -> int:
    """Infer the number of qubits touched by a QubitOperator."""
    max_q = -1
    for term in op.terms:
        for q, _ in term:
            max_q = max(max_q, q)
    return max_q + 1


def single_pauli_term(op: QubitOperator) -> Tuple[complex, Dict[int, str]]:
    """
    Parse a QubitOperator expected to contain exactly one Pauli string term.
    Returns:
        coeff, pauli_map
    """
    if len(op.terms) != 1:
        raise ValueError("Expected a single Pauli string term.")
    (term, coeff), = op.terms.items()
    pauli_map = {q: p for q, p in term}
    return coeff, pauli_map


def pauli_dict_to_qubit_operator(pauli_map: Dict[int, str], coeff: complex = 1.0) -> QubitOperator:
    """Convert {qubit: 'X'/'Y'/'Z'} to a QubitOperator."""
    if not pauli_map:
        return QubitOperator((), coeff)
    term = tuple(sorted(pauli_map.items()))
    return QubitOperator(term, coeff)


def binary_from_pauli_map(pauli_map: Dict[int, str], n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a Pauli string to binary symplectic form (x | z).
    """
    x = np.zeros(n_qubits, dtype=np.uint8)
    z = np.zeros(n_qubits, dtype=np.uint8)
    for q, p in pauli_map.items():
        if p == 'X':
            x[q] = 1
        elif p == 'Y':
            x[q] = 1
            z[q] = 1
        elif p == 'Z':
            z[q] = 1
        else:
            raise ValueError(f"Unsupported Pauli {p}")
    return x, z


def pauli_map_from_binary(x: np.ndarray, z: np.ndarray) -> Dict[int, str]:
    """
    Convert binary symplectic form (x | z) to {qubit: Pauli}.
    Phase is ignored.
    """
    pauli_map = {}
    for q in range(len(x)):
        if x[q] == 0 and z[q] == 0:
            continue
        if x[q] == 1 and z[q] == 0:
            pauli_map[q] = 'X'
        elif x[q] == 0 and z[q] == 1:
            pauli_map[q] = 'Z'
        elif x[q] == 1 and z[q] == 1:
            pauli_map[q] = 'Y'
        else:
            raise RuntimeError("Invalid symplectic Pauli entry.")
    return pauli_map


def binary_symplectic_commutes(x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray) -> bool:
    """Check commutation via symplectic inner product."""
    val = (np.dot(x1, z2) + np.dot(z1, x2)) % 2
    return val == 0


# ============================================================
# Elementary Clifford factors as QubitOperator sums
# ============================================================

def I_op() -> QubitOperator:
    return QubitOperator(())

def X_op(q: int) -> QubitOperator:
    return QubitOperator(((q, 'X'),))

def Y_op(q: int) -> QubitOperator:
    return QubitOperator(((q, 'Y'),))

def Z_op(q: int) -> QubitOperator:
    return QubitOperator(((q, 'Z'),))


def H_factor(q: int) -> QubitOperator:
    """
    Hadamard on qubit q:
        H = (X + Z) / sqrt(2)
    """
    return (X_op(q) + Z_op(q)) / np.sqrt(2.0)


def S_factor(q: int) -> QubitOperator:
    """
    Phase gate S = diag(1, i):
        S = ((1+i)/2) I + ((1-i)/2) Z
    """
    return ((1.0 + 1.0j) / 2.0) * I_op() + ((1.0 - 1.0j) / 2.0) * Z_op(q)


def Sdg_factor(q: int) -> QubitOperator:
    """
    S^\dagger = diag(1, -i):
        Sdg = ((1-i)/2) I + ((1+i)/2) Z
    """
    return ((1.0 - 1.0j) / 2.0) * I_op() + ((1.0 + 1.0j) / 2.0) * Z_op(q)


def CNOT_factor(control: int, target: int) -> QubitOperator:
    """
    CNOT(control -> target):
        |0><0|_c ⊗ I_t + |1><1|_c ⊗ X_t
      = 1/2 (I + Z_c + X_t - Z_c X_t)
    """
    return 0.5 * (
        I_op()
        + Z_op(control)
        + X_op(target)
        - Z_op(control) * X_op(target)
    )


# ============================================================
# Binary conjugation of rows under elementary Cliffords
# ============================================================

def apply_H_to_rows(xs: np.ndarray, zs: np.ndarray, q: int) -> None:
    xs[:, q], zs[:, q] = zs[:, q].copy(), xs[:, q].copy()


def apply_Sdg_to_rows(xs: np.ndarray, zs: np.ndarray, q: int) -> None:
    """
    In binary symplectic form, S and S^\dagger have the same x/z action
    (they differ only by phases, which we ignore here):
        x' = x
        z' = z xor x
    """
    zs[:, q] ^= xs[:, q]


def apply_S_to_rows(xs: np.ndarray, zs: np.ndarray, q: int) -> None:
    """
    Same binary action as S^\dagger when phase is ignored.
    """
    zs[:, q] ^= xs[:, q]


def apply_CNOT_to_rows(xs: np.ndarray, zs: np.ndarray, control: int, target: int) -> None:
    """
    Conjugate all rows by CNOT(control -> target).

    Binary update rule:
        x_t' = x_t xor x_c
        z_c' = z_c xor z_t
    """
    xs[:, target] ^= xs[:, control]
    zs[:, control] ^= zs[:, target]


def apply_H_to_pauli(x: np.ndarray, z: np.ndarray, q: int) -> None:
    x[q], z[q] = z[q].copy(), x[q].copy()


def apply_Sdg_to_pauli(x: np.ndarray, z: np.ndarray, q: int) -> None:
    z[q] ^= x[q]


def apply_S_to_pauli(x: np.ndarray, z: np.ndarray, q: int) -> None:
    z[q] ^= x[q]


def apply_CNOT_to_pauli(x: np.ndarray, z: np.ndarray, control: int, target: int) -> None:
    x[target] ^= x[control]
    z[control] ^= z[target]

# ============================================================
# Exact Pauli-term manipulation
# ============================================================

def _term_to_pauli_dict(term: Tuple[Tuple[int, str], ...]) -> Dict[int, str]:
    return dict(term)


def _pauli_dict_to_term(pauli_dict: Dict[int, str]) -> Tuple[Tuple[int, str], ...]:
    return tuple(sorted((q, p) for q, p in pauli_dict.items() if p != "I"))


def _get_local_pauli(pauli_dict: Dict[int, str], q: int) -> str:
    return pauli_dict.get(q, "I")


def _set_local_pauli(pauli_dict: Dict[int, str], q: int, p: str) -> None:
    if p == "I":
        pauli_dict.pop(q, None)
    else:
        pauli_dict[q] = p


# ============================================================
# Exact single-gate conjugation rules
# ============================================================

def conjugate_term_by_H(
    term: Tuple[Tuple[int, str], ...],
    coeff: complex,
    q: int,
) -> Tuple[Tuple[Tuple[int, str], ...], complex]:
    """
    Exact conjugation by H(q):
        H X H = Z
        H Z H = X
        H Y H = -Y
    """
    pauli_dict = _term_to_pauli_dict(term)
    p = _get_local_pauli(pauli_dict, q)

    if p == "I":
        pass
    elif p == "X":
        _set_local_pauli(pauli_dict, q, "Z")
    elif p == "Z":
        _set_local_pauli(pauli_dict, q, "X")
    elif p == "Y":
        _set_local_pauli(pauli_dict, q, "Y")
        coeff *= -1
    else:
        raise ValueError(f"Unsupported Pauli {p}")

    return _pauli_dict_to_term(pauli_dict), coeff


def conjugate_term_by_Sdg(
    term: Tuple[Tuple[int, str], ...],
    coeff: complex,
    q: int,
) -> Tuple[Tuple[Tuple[int, str], ...], complex]:
    """
    Exact conjugation by S^\dagger(q):
        S^\dagger X S = -Y
        S^\dagger Y S = X
        S^\dagger Z S = Z
    """
    pauli_dict = _term_to_pauli_dict(term)
    p = _get_local_pauli(pauli_dict, q)

    if p == "I":
        pass
    elif p == "X":
        _set_local_pauli(pauli_dict, q, "Y")
        coeff *= -1
    elif p == "Y":
        _set_local_pauli(pauli_dict, q, "X")
    elif p == "Z":
        _set_local_pauli(pauli_dict, q, "Z")
    else:
        raise ValueError(f"Unsupported Pauli {p}")

    return _pauli_dict_to_term(pauli_dict), coeff


# Exact CNOT(c -> t) conjugation lookup on the local 2-qubit Pauli pair.
# Each entry: (phase, new_Pc, new_Pt)
_CNOT_CONJ_TABLE = {
    ("I", "I"): (1, "I", "I"),
    ("I", "X"): (1, "I", "X"),
    ("I", "Y"): (1, "Z", "Y"),
    ("I", "Z"): (1, "Z", "Z"),

    ("X", "I"): (1, "X", "X"),
    ("X", "X"): (1, "X", "I"),
    ("X", "Y"): (1, "Y", "Z"),
    ("X", "Z"): (-1, "Y", "Y"),

    ("Y", "I"): (1, "Y", "X"),
    ("Y", "X"): (1, "Y", "I"),
    ("Y", "Y"): (-1, "X", "Z"),
    ("Y", "Z"): (1, "X", "Y"),

    ("Z", "I"): (1, "Z", "I"),
    ("Z", "X"): (1, "Z", "X"),
    ("Z", "Y"): (1, "I", "Y"),
    ("Z", "Z"): (1, "I", "Z"),
}


def conjugate_term_by_CNOT(
    term: Tuple[Tuple[int, str], ...],
    coeff: complex,
    control: int,
    target: int,
) -> Tuple[Tuple[Tuple[int, str], ...], complex]:
    """
    Exact conjugation by CNOT(control -> target).
    """
    pauli_dict = _term_to_pauli_dict(term)
    pc = _get_local_pauli(pauli_dict, control)
    pt = _get_local_pauli(pauli_dict, target)

    phase, new_pc, new_pt = _CNOT_CONJ_TABLE[(pc, pt)]
    coeff *= phase

    _set_local_pauli(pauli_dict, control, new_pc)
    _set_local_pauli(pauli_dict, target, new_pt)

    return _pauli_dict_to_term(pauli_dict), coeff


# ============================================================
# Exact conjugation by the synthesized Clifford sequence
# ============================================================

def conjugate_single_term_by_factor_sequence_exact(
    term: Tuple[Tuple[int, str], ...],
    coeff: complex,
    factor_descriptions: Sequence[str],
) -> Tuple[Tuple[Tuple[int, str], ...], complex]:
    """
    Conjugate a single Pauli term exactly by the ordered factor sequence.
    """
    new_term = term
    new_coeff = coeff

    for desc in factor_descriptions:
        if desc.startswith("H("):
            q = int(desc[2:].strip("()"))
            new_term, new_coeff = conjugate_term_by_H(new_term, new_coeff, q)

        elif desc.startswith("Sdg("):
            q = int(desc[4:].strip("()"))
            new_term, new_coeff = conjugate_term_by_Sdg(new_term, new_coeff, q)

        elif desc.startswith("CNOT("):
            inside = desc[5:].strip("()")
            c, t = inside.split("->")
            new_term, new_coeff = conjugate_term_by_CNOT(
                new_term,
                new_coeff,
                int(c),
                int(t),
            )
        else:
            raise ValueError(f"Unknown factor description: {desc}")

    return new_term, new_coeff


def conjugate_qubit_operator_by_clifford_factors_exact(
    op: QubitOperator,
    factor_descriptions: Sequence[str],
    compress_abs_tol: float = 1e-12,
) -> QubitOperator:
    """
    Exact conjugation of an arbitrary QubitOperator by the synthesized Clifford.

    This preserves the spectrum.
    """
    transformed = QubitOperator()

    for term, coeff in op.terms.items():
        new_term, new_coeff = conjugate_single_term_by_factor_sequence_exact(
            term=term,
            coeff=coeff,
            factor_descriptions=factor_descriptions,
        )
        transformed += QubitOperator(new_term, new_coeff)

    transformed.compress(abs_tol=compress_abs_tol)
    return transformed


# ============================================================
# Synthesis result
# ============================================================

@dataclass
class CliffordSynthesisResult:
    mapped_qubits: List[int]
    elementary_factors: List[QubitOperator]
    factor_descriptions: List[str]
    full_clifford: Optional[QubitOperator]
    transformed_generators: List[QubitOperator]


# ============================================================
# Ordered symmetry Clifford synthesis
# ============================================================

def synthesize_ordered_symmetry_clifford(
    symmetries: Sequence[QubitOperator],
    n_qubits: Optional[int] = None,
    return_full_clifford: bool = True,
) -> CliffordSynthesisResult:
    """
    Synthesize a Clifford that maps an ordered set of commuting independent
    Pauli symmetries to single-qubit Z operators, hierarchically.

    Earlier symmetries are fixed first. Later symmetries may be multiplied
    by earlier ones to clear previously assigned symmetry qubits.
    """
    if len(symmetries) == 0:
        raise ValueError("Need at least one symmetry.")

    if n_qubits is None:
        n_qubits = 0
        for s in symmetries:
            n_qubits = max(n_qubits, qubit_operator_num_qubits(s))

    rows_x = []
    rows_z = []
    for s in symmetries:
        coeff, pauli_map = single_pauli_term(s)
        if not np.isclose(abs(coeff), 1.0):
            raise ValueError(
                "Each symmetry should be a Hermitian Pauli string with coefficient magnitude 1."
            )
        x, z = binary_from_pauli_map(pauli_map, n_qubits)
        rows_x.append(x)
        rows_z.append(z)

    xs = np.array(rows_x, dtype=np.uint8)
    zs = np.array(rows_z, dtype=np.uint8)

    m = len(symmetries)
    for i in range(m):
        for j in range(i + 1, m):
            if not binary_symplectic_commutes(xs[i], zs[i], xs[j], zs[j]):
                raise ValueError(f"Symmetries {i} and {j} do not commute.")

    mapped_qubits: List[int] = []
    elementary_factors: List[QubitOperator] = []
    factor_descriptions: List[str] = []

    for i in range(m):
        # Clear earlier pivot Z's from row i by multiplying with earlier rows.
        for j, q_prev in enumerate(mapped_qubits):
            if xs[i, q_prev] != 0:
                raise RuntimeError(
                    "Invariant violated: later row has X/Y on an earlier symmetry qubit."
                )
            if zs[i, q_prev] == 1:
                xs[i] ^= xs[j]
                zs[i] ^= zs[j]

        support = [q for q in range(n_qubits) if (xs[i, q] or zs[i, q]) and q not in mapped_qubits]
        if not support:
            raise ValueError(
                f"Symmetry {i} became trivial after hierarchical clearing. "
                f"Input set is not independent in this ordered sense."
            )

        pivot = support[0]
        mapped_qubits.append(pivot)

        # Rotate each active local Pauli to X.
        active_support = [q for q in range(n_qubits) if (xs[i, q] or zs[i, q])]
        for q in active_support:
            if q in mapped_qubits[:-1]:
                if xs[i, q] or zs[i, q]:
                    raise RuntimeError("Failed to clear previous pivot support.")
                continue

            if xs[i, q] == 1 and zs[i, q] == 0:
                pass  # X already
            elif xs[i, q] == 1 and zs[i, q] == 1:
                apply_Sdg_to_rows(xs, zs, q)
                elementary_factors.append(Sdg_factor(q))
                factor_descriptions.append(f"Sdg({q})")
            elif xs[i, q] == 0 and zs[i, q] == 1:
                apply_H_to_rows(xs, zs, q)
                elementary_factors.append(H_factor(q))
                factor_descriptions.append(f"H({q})")
            else:
                raise RuntimeError("Unexpected local Pauli state.")

        # Gather parity onto pivot.
        active_support = [q for q in range(n_qubits) if (xs[i, q] or zs[i, q])]
        for q in list(active_support):
            if q == pivot:
                continue
            apply_CNOT_to_rows(xs, zs, pivot, q)
            elementary_factors.append(CNOT_factor(pivot, q))
            factor_descriptions.append(f"CNOT({pivot}->{q})")

        # X(pivot) -> Z(pivot)
        apply_H_to_rows(xs, zs, pivot)
        elementary_factors.append(H_factor(pivot))
        factor_descriptions.append(f"H({pivot})")

    transformed_generators = []
    for i in range(m):
        pauli_map = pauli_map_from_binary(xs[i], zs[i])
        transformed_generators.append(pauli_dict_to_qubit_operator(pauli_map))

    full_clifford = None
    if return_full_clifford:
        full_clifford = I_op()
        for U in elementary_factors:
            full_clifford = U * full_clifford

    return CliffordSynthesisResult(
        mapped_qubits=mapped_qubits,
        elementary_factors=elementary_factors,
        factor_descriptions=factor_descriptions,
        full_clifford=full_clifford,
        transformed_generators=transformed_generators,
    )


# ============================================================
# Conjugating a QubitOperator by the synthesized Clifford
# ============================================================

def conjugate_single_pauli_by_factor_sequence(
    pauli_op: QubitOperator,
    factor_descriptions: Sequence[str],
    n_qubits: Optional[int] = None,
) -> QubitOperator:
    """
    Conjugate a single Pauli-string QubitOperator by the ordered factor sequence.

    Returns the transformed Pauli string, ignoring global phase/sign.

    This is done in binary symplectic form using the same gate sequence used
    during synthesis.
    """
    coeff, pauli_map = single_pauli_term(pauli_op)

    if n_qubits is None:
        n_qubits = qubit_operator_num_qubits(pauli_op)

    x, z = binary_from_pauli_map(pauli_map, n_qubits)

    for desc in factor_descriptions:
        if desc.startswith("H("):
            q = int(desc[2:].strip("()"))
            apply_H_to_pauli(x, z, q)
        elif desc.startswith("Sdg("):
            q = int(desc[4:].strip("()"))
            apply_Sdg_to_pauli(x, z, q)
        elif desc.startswith("S("):
            q = int(desc[2:].strip("()"))
            apply_S_to_pauli(x, z, q)
        elif desc.startswith("CNOT("):
            inside = desc[5:].strip("()")
            c, t = inside.split("->")
            apply_CNOT_to_pauli(x, z, int(c), int(t))
        else:
            raise ValueError(f"Unknown factor description: {desc}")

    new_map = pauli_map_from_binary(x, z)
    return pauli_dict_to_qubit_operator(new_map, coeff=coeff)


def conjugate_qubit_operator_by_clifford_factors(
    op: QubitOperator,
    factor_descriptions: Sequence[str],
    n_qubits: Optional[int] = None,
    compress_abs_tol: float = 1e-12,
) -> QubitOperator:
    """
    Conjugate an arbitrary QubitOperator by the synthesized Clifford sequence.

    Important:
    - This preserves coefficients but ignores any sign/global-phase changes
      from Pauli conjugation.
    - For symmetry block-structure visualization this is often enough.
    - If you need exact coefficients/signs for spectroscopy, add phase tracking.

    Parameters
    ----------
    op:
        Arbitrary QubitOperator Hamiltonian.
    factor_descriptions:
        The list from synthesize_ordered_symmetry_clifford(...).factor_descriptions
    n_qubits:
        Total qubit count.
    compress_abs_tol:
        Coefficient tolerance for compression.

    Returns
    -------
    transformed_op : QubitOperator
    """
    if n_qubits is None:
        n_qubits = qubit_operator_num_qubits(op)

    transformed = QubitOperator()

    for term, coeff in op.terms.items():
        term_op = QubitOperator(term, coeff)
        new_term_op = conjugate_single_pauli_by_factor_sequence(
            term_op,
            factor_descriptions=factor_descriptions,
            n_qubits=n_qubits,
        )
        transformed += new_term_op

    transformed.compress(abs_tol=compress_abs_tol)
    return transformed


# ============================================================
# Sector ordering utilities
# ============================================================

def int_to_bitstring(x: int, n_qubits: int) -> Tuple[int, ...]:
    """
    Convert basis-state integer to tuple of qubit bits.
    Qubit q corresponds to bit ((x >> q) & 1).
    """
    return tuple((x >> q) & 1 for q in range(n_qubits))


def symmetry_sector_label(
    basis_index: int,
    symmetry_qubits: Sequence[int],
) -> Tuple[int, ...]:
    """
    Return the computational-basis bitstring restricted to the symmetry qubits.
    After mapping symmetries to Z_q, this is the symmetry-sector label.

    Note:
    bit=0 corresponds to Z eigenvalue +1
    bit=1 corresponds to Z eigenvalue -1
    """
    return tuple((basis_index >> q) & 1 for q in symmetry_qubits)


def sector_ordering_from_symmetry_qubits(
    n_qubits: int,
    symmetry_qubits: Sequence[int],
    residual_qubit_order: Optional[Sequence[int]] = None,
) -> Tuple[List[int], Dict[Tuple[int, ...], List[int]], List[Tuple[int, ...]]]:
    """
    Build an ordering of computational basis states grouped first by symmetry sector.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    symmetry_qubits:
        List of qubits q_i onto which the chosen symmetries were mapped.
    residual_qubit_order:
        Optional order for tie-breaking within each sector.
        If None, uses ascending order of the non-symmetry qubits.

    Returns
    -------
    ordered_basis_indices:
        List of basis state indices in reordered order.
    sector_to_indices:
        Map sector bitstring -> list of original basis indices in that sector.
    ordered_sectors:
        Sector labels in the order used.
    """
    symmetry_qubits = list(symmetry_qubits)
    all_qubits = list(range(n_qubits))
    residual_qubits = [q for q in all_qubits if q not in symmetry_qubits]

    if residual_qubit_order is None:
        residual_qubit_order = residual_qubits
    else:
        residual_qubit_order = list(residual_qubit_order)

    sector_to_indices: Dict[Tuple[int, ...], List[int]] = {}

    for state in range(2 ** n_qubits):
        sec = symmetry_sector_label(state, symmetry_qubits)
        sector_to_indices.setdefault(sec, []).append(state)

    # Sort states inside each sector by residual qubit lexicographic bit pattern.
    def residual_key(state: int) -> Tuple[int, ...]:
        return tuple((state >> q) & 1 for q in residual_qubit_order)

    for sec in sector_to_indices:
        sector_to_indices[sec].sort(key=residual_key)

    ordered_sectors = sorted(sector_to_indices.keys())
    ordered_basis_indices: List[int] = []
    for sec in ordered_sectors:
        ordered_basis_indices.extend(sector_to_indices[sec])

    return ordered_basis_indices, sector_to_indices, ordered_sectors


def permutation_matrix_from_order(order: Sequence[int]) -> np.ndarray:
    """
    Build permutation matrix P such that:
        H_reordered = P H P^T
    where 'order[new_index] = old_index'.
    """
    dim = len(order)
    P = np.zeros((dim, dim), dtype=np.float64)
    for new_i, old_i in enumerate(order):
        P[new_i, old_i] = 1.0
    return P


# ============================================================
# Reordered Hamiltonian matrix and plotting
# ============================================================

@dataclass
class ReorderedHamiltonianResult:
    transformed_hamiltonian: QubitOperator
    transformed_matrix: np.ndarray
    reordered_matrix: np.ndarray
    basis_order: List[int]
    ordered_sectors: List[Tuple[int, ...]]
    sector_boundaries: List[int]


def reordered_matrix_by_sector(
    hamiltonian: QubitOperator,
    symmetry_qubits: Sequence[int],
    factor_descriptions: Sequence[str],
    n_qubits: Optional[int] = None,
) -> ReorderedHamiltonianResult:
    """
    If factor_descriptions is nonempty:
      1. Conjugate the Hamiltonian by the synthesized Clifford
      2. Convert to dense matrix
      3. Reorder basis by symmetry-sector labels

    If factor_descriptions is empty:
      - assume hamiltonian is already in the desired qubit basis.
    """
    if n_qubits is None:
        n_qubits = qubit_operator_num_qubits(hamiltonian)

    if len(factor_descriptions) > 0:
        transformed_h = conjugate_qubit_operator_by_clifford_factors_exact(
            hamiltonian,
            factor_descriptions=factor_descriptions,
        )
    else:
        transformed_h = hamiltonian

    H_mat = get_sparse_operator(transformed_h, n_qubits=n_qubits).toarray()

    basis_order, sector_to_indices, ordered_sectors = sector_ordering_from_symmetry_qubits(
        n_qubits=n_qubits,
        symmetry_qubits=symmetry_qubits,
    )

    P = permutation_matrix_from_order(basis_order)
    H_reordered = P @ H_mat @ P.T

    sector_boundaries = []
    running = 0
    for sec in ordered_sectors:
        running += len(sector_to_indices[sec])
        sector_boundaries.append(running)

    return ReorderedHamiltonianResult(
        transformed_hamiltonian=transformed_h,
        transformed_matrix=H_mat,
        reordered_matrix=H_reordered,
        basis_order=basis_order,
        ordered_sectors=ordered_sectors,
        sector_boundaries=sector_boundaries,
    )


def plot_reordered_hamiltonian(
    reordered_result: ReorderedHamiltonianResult,
    use_log10_abs: bool = True,
    eps: float = 1e-14,
    title: str = "Reordered Hamiltonian by symmetry sectors",
    figsize: Tuple[float, float] = (7, 7),
) -> None:
    """
    Plot the reordered Hamiltonian matrix with sector boundaries overlaid.
    """
    H = reordered_result.reordered_matrix
    if use_log10_abs:
        plot_data = np.log10(np.abs(H) + eps)
        cbar_label = r"$\log_{10}(|H_{ij}|+\epsilon)$"
    else:
        plot_data = np.abs(H)
        cbar_label = r"$|H_{ij}|$"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_data, origin="lower", interpolation="nearest", aspect="equal")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Draw sector boundaries.
    for b in reordered_result.sector_boundaries[:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Basis index (reordered)")
    ax.set_ylabel("Basis index (reordered)")
    plt.tight_layout()
    plt.show()


# ============================================================
# Convenience: one-shot pipeline
# ============================================================

@dataclass
class SymmetryBlockStructureResult:
    clifford_result: CliffordSynthesisResult
    reordered_result: ReorderedHamiltonianResult


def build_symmetry_block_structure(
    hamiltonian: QubitOperator,
    symmetries: Sequence[QubitOperator],
    n_qubits: Optional[int] = None,
    return_full_clifford: bool = False,
) -> SymmetryBlockStructureResult:
    """
    End-to-end helper:
      1. synthesize ordered Clifford from symmetries
      2. transform Hamiltonian
      3. reorder basis by resulting symmetry qubits
    """
    if n_qubits is None:
        n_qubits = max(
            qubit_operator_num_qubits(hamiltonian),
            max(qubit_operator_num_qubits(s) for s in symmetries),
        )

    clifford_result = synthesize_ordered_symmetry_clifford(
        symmetries=symmetries,
        n_qubits=n_qubits,
        return_full_clifford=return_full_clifford,
    )

    reordered_result = reordered_matrix_by_sector(
        hamiltonian=hamiltonian,
        symmetry_qubits=clifford_result.mapped_qubits,
        factor_descriptions=clifford_result.factor_descriptions,
        n_qubits=n_qubits,
    )

    return SymmetryBlockStructureResult(
        clifford_result=clifford_result,
        reordered_result=reordered_result,
    )



#### permutation helpers
from typing import Dict, List, Optional, Sequence, Tuple
from openfermion import QubitOperator


def invert_permutation(perm: Sequence[int]) -> List[int]:
    """
    Given perm[old_q] = new_q, return inv_perm[new_q] = old_q.
    """
    n = len(perm)
    inv = [None] * n
    for old_q, new_q in enumerate(perm):
        if not (0 <= new_q < n):
            raise ValueError("Invalid permutation entry.")
        if inv[new_q] is not None:
            raise ValueError("Permutation is not one-to-one.")
        inv[new_q] = old_q
    return inv


def permute_qubits_in_term(term: Tuple[Tuple[int, str], ...], perm: Sequence[int]) -> Tuple[Tuple[int, str], ...]:
    """
    Apply qubit permutation to a single Pauli term.

    perm[old_q] = new_q
    """
    new_term = tuple(sorted((perm[q], p) for q, p in term))
    return new_term


def permute_qubits_in_qubit_operator(
    op: QubitOperator,
    perm: Sequence[int],
    compress_abs_tol: float = 1e-12,
) -> QubitOperator:
    """
    Apply a qubit permutation to every term in a QubitOperator.

    Parameters
    ----------
    op:
        Input QubitOperator.
    perm:
        List with perm[old_q] = new_q.
    """
    out = QubitOperator()
    for term, coeff in op.terms.items():
        new_term = permute_qubits_in_term(term, perm)
        out += QubitOperator(new_term, coeff)
    out.compress(abs_tol=compress_abs_tol)
    return out


def permute_qubit_list(qubits: Sequence[int], perm: Sequence[int]) -> List[int]:
    """
    Apply perm[old_q] = new_q to a list of qubit indices.
    """
    return [perm[q] for q in qubits]


def make_symmetry_qubits_last_permutation(
    n_qubits: int,
    symmetry_qubits: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """
    Build a permutation that moves the given symmetry qubits to the end
    of the register, preserving their relative order.

    Returns
    -------
    perm:
        perm[old_q] = new_q
    new_symmetry_qubits:
        The symmetry qubit indices after permutation, typically
        [n-k, ..., n-1].
    """
    symmetry_qubits = list(symmetry_qubits)
    symmetry_set = set(symmetry_qubits)

    if len(symmetry_set) != len(symmetry_qubits):
        raise ValueError("symmetry_qubits contains duplicates.")

    for q in symmetry_qubits:
        if not (0 <= q < n_qubits):
            raise ValueError("symmetry qubit out of range.")

    nonsym = [q for q in range(n_qubits) if q not in symmetry_set]
    new_order_old_qubits = nonsym + symmetry_qubits
    # old qubits listed in the order they will appear as new indices 0..n-1

    perm = [None] * n_qubits
    for new_q, old_q in enumerate(new_order_old_qubits):
        perm[old_q] = new_q

    new_symmetry_qubits = [perm[q] for q in symmetry_qubits]
    return perm, new_symmetry_qubits

from dataclasses import dataclass
import numpy as np


@dataclass
class PermutedHamiltonianResult:
    permuted_hamiltonian: QubitOperator
    qubit_permutation: List[int]          # perm[old_q] = new_q
    permuted_symmetry_qubits: List[int]   # updated mapped qubits


def move_symmetry_qubits_to_end(
    transformed_hamiltonian: QubitOperator,
    mapped_qubits: Sequence[int],
    n_qubits: int,
) -> PermutedHamiltonianResult:
    """
    Permute qubits so that the mapped symmetry qubits are moved to the end
    of the register, preserving their relative order.
    """
    perm, new_mapped = make_symmetry_qubits_last_permutation(
        n_qubits=n_qubits,
        symmetry_qubits=mapped_qubits,
    )

    H_perm = permute_qubits_in_qubit_operator(
        transformed_hamiltonian,
        perm=perm,
    )

    return PermutedHamiltonianResult(
        permuted_hamiltonian=H_perm,
        qubit_permutation=perm,
        permuted_symmetry_qubits=new_mapped,
    )

def permute_hamiltonian_qubits(
    transformed_hamiltonian: QubitOperator,
    perm,
    sym_qubits,
    validate = False
) -> PermutedHamiltonianResult:
    """
    Permute qubits so that the mapped symmetry qubits are moved to positions specified by perm (list):

    perm[old] = new
    
    """    

    H_perm = permute_qubits_in_qubit_operator(
        transformed_hamiltonian,
        perm=perm,
    )

    # if validate:
    #     #isolate Hamiltonian blocks and check

    return PermutedHamiltonianResult(
        permuted_hamiltonian=H_perm,
        qubit_permutation=perm,
        permuted_symmetry_qubits=permute_qubit_list(sym_qubits, perm=perm),
    )

@dataclass
class SymmetryBlockStructurePackedResult:
    clifford_result: CliffordSynthesisResult
    transformed_hamiltonian: QubitOperator
    packed_hamiltonian: QubitOperator
    original_mapped_qubits: List[int]
    packed_symmetry_qubits: List[int]
    qubit_permutation: List[int]
    reordered_matrix: np.ndarray
    ordered_sectors: List[Tuple[int, ...]]
    sector_boundaries: List[int]


def build_symmetry_block_structure_with_packed_qubits(
    hamiltonian: QubitOperator,
    symmetries: Sequence[QubitOperator],
    n_qubits: int,
    return_full_clifford: bool = False,
    reorder_sector=False
) -> SymmetryBlockStructurePackedResult:
    """
    End-to-end:
      1. synthesize ordered Clifford
      2. transform Hamiltonian
      3. permute qubits so symmetry qubits are at the end
      4. reorder basis states by symmetry sectors
    """
    clifford_result = synthesize_ordered_symmetry_clifford(
        symmetries=symmetries,
        n_qubits=n_qubits,
        return_full_clifford=return_full_clifford,
    )

    transformed_h = conjugate_qubit_operator_by_clifford_factors_exact(
        hamiltonian,
        factor_descriptions=clifford_result.factor_descriptions,
    )

    packed = move_symmetry_qubits_to_end(
        transformed_hamiltonian=transformed_h,
        mapped_qubits=clifford_result.mapped_qubits,
        n_qubits=n_qubits,
    )

    if reorder_sector:
        reordered = reordered_matrix_by_sector(
            hamiltonian=packed.permuted_hamiltonian,
            symmetry_qubits=packed.permuted_symmetry_qubits,
            factor_descriptions=[],  # already transformed; no extra Clifford now
            n_qubits=n_qubits,
        )

        return SymmetryBlockStructurePackedResult(
            clifford_result=clifford_result,
            transformed_hamiltonian=transformed_h,
            packed_hamiltonian=packed.permuted_hamiltonian,
            original_mapped_qubits=clifford_result.mapped_qubits,
            packed_symmetry_qubits=packed.permuted_symmetry_qubits,
            qubit_permutation=packed.qubit_permutation,
            reordered_matrix=reordered.reordered_matrix,
            ordered_sectors=reordered.ordered_sectors,
            sector_boundaries=reordered.sector_boundaries,
        )
    else:

        return SymmetryBlockStructurePackedResult(
            clifford_result=clifford_result,
            transformed_hamiltonian=transformed_h,
            packed_hamiltonian=packed.permuted_hamiltonian,
            original_mapped_qubits=clifford_result.mapped_qubits,
            packed_symmetry_qubits=packed.permuted_symmetry_qubits,
            qubit_permutation=packed.qubit_permutation,
            reordered_matrix=None,
            ordered_sectors=None,
            sector_boundaries=None,
        )