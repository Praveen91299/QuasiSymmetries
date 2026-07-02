"""
Exact Clifford synthesis and Pauli/Hamiltonian conjugation utilities.

This version removes the previous sign-dropping conjugation path. All public
Hamiltonian conjugation uses exact Pauli phase/sign tracking. Exact conjugation
is implemented with integer binary symplectic masks instead of dict-per-gate
term rewrites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from openfermion import QubitOperator, get_sparse_operator


# ============================================================
# Basic Pauli / QubitOperator utilities
# ============================================================


def check_same_spectrum(
    h1: QubitOperator,
    h2: QubitOperator,
    n_qubits: int,
    atol: float = 1e-10,
) -> bool:
    """Compare spectra of two small Hermitian QubitOperators.

    This is a validation/debug helper and necessarily densifies the operators.
    Do not use it for large systems.
    """
    m1 = get_sparse_operator(h1, n_qubits=n_qubits).toarray()
    m2 = get_sparse_operator(h2, n_qubits=n_qubits).toarray()
    e1 = np.linalg.eigvalsh(m1)
    e2 = np.linalg.eigvalsh(m2)
    ok = np.allclose(e1, e2, atol=atol, rtol=0.0)
    if not ok:
        print("Max eigval diff:", np.max(np.abs(e1 - e2)))
    return bool(ok)


def qubit_operator_num_qubits(op: QubitOperator) -> int:
    """Infer the number of qubits touched by a QubitOperator."""
    max_q = -1
    for term in op.terms:
        for q, _ in term:
            max_q = max(max_q, q)
    return max_q + 1


def single_pauli_term(op: QubitOperator) -> Tuple[complex, Dict[int, str]]:
    """Parse a QubitOperator expected to contain exactly one Pauli string term."""
    if len(op.terms) != 1:
        raise ValueError("Expected a single Pauli string term.")
    (term, coeff), = op.terms.items()
    return coeff, {q: p for q, p in term}


def pauli_dict_to_qubit_operator(pauli_map: Dict[int, str], coeff: complex = 1.0) -> QubitOperator:
    """Convert {qubit: 'X'/'Y'/'Z'} to a QubitOperator."""
    if not pauli_map:
        return QubitOperator((), coeff)
    return QubitOperator(tuple(sorted(pauli_map.items())), coeff)


def binary_from_pauli_map(pauli_map: Dict[int, str], n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a Pauli string to binary symplectic form (x | z)."""
    x = np.zeros(n_qubits, dtype=np.uint8)
    z = np.zeros(n_qubits, dtype=np.uint8)
    for q, p in pauli_map.items():
        if q < 0 or q >= n_qubits:
            raise ValueError(f"Qubit index {q} outside n_qubits={n_qubits}.")
        if p == "X":
            x[q] = 1
        elif p == "Y":
            x[q] = 1
            z[q] = 1
        elif p == "Z":
            z[q] = 1
        else:
            raise ValueError(f"Unsupported Pauli {p!r}")
    return x, z


def pauli_map_from_binary(x: np.ndarray, z: np.ndarray) -> Dict[int, str]:
    """Convert binary symplectic form (x | z) to {qubit: Pauli}."""
    if len(x) != len(z):
        raise ValueError("x and z must have the same length.")
    pauli_map: Dict[int, str] = {}
    for q in range(len(x)):
        xb, zb = int(x[q]), int(z[q])
        if xb == 0 and zb == 0:
            continue
        if xb == 1 and zb == 0:
            pauli_map[q] = "X"
        elif xb == 0 and zb == 1:
            pauli_map[q] = "Z"
        elif xb == 1 and zb == 1:
            pauli_map[q] = "Y"
        else:
            raise RuntimeError("Invalid symplectic Pauli entry.")
    return pauli_map


def binary_symplectic_commutes(x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray) -> bool:
    """Check commutation via the symplectic inner product over GF(2)."""
    val = (np.dot(x1, z2) + np.dot(z1, x2)) % 2
    return bool(val == 0)


# ============================================================
# Optional elementary Clifford factors as QubitOperator sums
# ============================================================


def I_op() -> QubitOperator:
    return QubitOperator(())


def X_op(q: int) -> QubitOperator:
    return QubitOperator(((q, "X"),))


def Y_op(q: int) -> QubitOperator:
    return QubitOperator(((q, "Y"),))


def Z_op(q: int) -> QubitOperator:
    return QubitOperator(((q, "Z"),))


def H_factor(q: int) -> QubitOperator:
    return (X_op(q) + Z_op(q)) / np.sqrt(2.0)


def S_factor(q: int) -> QubitOperator:
    return ((1.0 + 1.0j) / 2.0) * I_op() + ((1.0 - 1.0j) / 2.0) * Z_op(q)


def Sdg_factor(q: int) -> QubitOperator:
    return ((1.0 - 1.0j) / 2.0) * I_op() + ((1.0 + 1.0j) / 2.0) * Z_op(q)


def CNOT_factor(control: int, target: int) -> QubitOperator:
    return 0.5 * (I_op() + Z_op(control) + X_op(target) - Z_op(control) * X_op(target))


def factor_from_parsed_gate(gate: Tuple[Union[str, int], ...]) -> QubitOperator:
    name = gate[0]
    if name == "H":
        return H_factor(int(gate[1]))
    if name == "S":
        return S_factor(int(gate[1]))
    if name == "Sdg":
        return Sdg_factor(int(gate[1]))
    if name == "CNOT":
        return CNOT_factor(int(gate[1]), int(gate[2]))
    raise ValueError(f"Unknown gate {gate!r}")


# ============================================================
# Binary conjugation of synthesis rows under elementary Cliffords
# ============================================================


def apply_H_to_rows(xs: np.ndarray, zs: np.ndarray, q: int) -> None:
    xs[:, q], zs[:, q] = zs[:, q].copy(), xs[:, q].copy()


def apply_Sdg_to_rows(xs: np.ndarray, zs: np.ndarray, q: int) -> None:
    zs[:, q] ^= xs[:, q]


def apply_S_to_rows(xs: np.ndarray, zs: np.ndarray, q: int) -> None:
    zs[:, q] ^= xs[:, q]


def apply_CNOT_to_rows(xs: np.ndarray, zs: np.ndarray, control: int, target: int) -> None:
    xs[:, target] ^= xs[:, control]
    zs[:, control] ^= zs[:, target]


# ============================================================
# Exact term conjugation with integer masks and phase/sign tracking
# ============================================================


ParsedGate = Tuple[Union[str, int], ...]
Term = Tuple[Tuple[int, str], ...]


_LOCAL_CODE = {
    (0, 0): 0,  # I
    (1, 0): 1,  # X
    (0, 1): 2,  # Z
    (1, 1): 3,  # Y
}

# Sign of CNOT(c -> t) conjugation by local Pauli pair codes:
# code order is I=0, X=1, Z=2, Y=3.
_CNOT_PHASE = np.ones((4, 4), dtype=np.int8)
_CNOT_PHASE[1, 2] = -1  # X_c Z_t -> -Y_c Y_t
_CNOT_PHASE[3, 3] = -1  # Y_c Y_t -> -X_c Z_t


def term_to_masks(term: Term) -> Tuple[int, int]:
    """Convert an OpenFermion Pauli term tuple to integer (x_mask, z_mask)."""
    x = 0
    z = 0
    for q, p in term:
        bit = 1 << q
        if p == "X":
            x |= bit
        elif p == "Y":
            x |= bit
            z |= bit
        elif p == "Z":
            z |= bit
        elif p == "I":
            continue
        else:
            raise ValueError(f"Unsupported Pauli {p!r}")
    return x, z


def masks_to_term(x: int, z: int, n_qubits: int) -> Term:
    """Convert integer (x_mask, z_mask) to an OpenFermion Pauli term tuple."""
    out: List[Tuple[int, str]] = []
    support = x | z
    while support:
        low = support & -support
        q = low.bit_length() - 1
        xb = 1 if (x & low) else 0
        zb = 1 if (z & low) else 0
        if xb and zb:
            out.append((q, "Y"))
        elif xb:
            out.append((q, "X"))
        elif zb:
            out.append((q, "Z"))
        support ^= low
    # n_qubits is included for API clarity and future validation.
    _ = n_qubits
    return tuple(out)


def _local_code(x: int, z: int, q: int) -> int:
    bit = 1 << q
    return _LOCAL_CODE[(1 if x & bit else 0, 1 if z & bit else 0)]


def apply_H_to_masks(x: int, z: int, sign: int, q: int) -> Tuple[int, int, int]:
    bit = 1 << q
    xb = bool(x & bit)
    zb = bool(z & bit)
    if xb and zb:
        sign = -sign  # H Y H = -Y
    if xb != zb:
        x ^= bit
        z ^= bit
    return x, z, sign


def apply_Sdg_to_masks(x: int, z: int, sign: int, q: int) -> Tuple[int, int, int]:
    bit = 1 << q
    xb = bool(x & bit)
    zb = bool(z & bit)
    if xb and not zb:
        sign = -sign  # Sdg X S = -Y
    if xb:
        z ^= bit
    return x, z, sign


def apply_S_to_masks(x: int, z: int, sign: int, q: int) -> Tuple[int, int, int]:
    bit = 1 << q
    xb = bool(x & bit)
    zb = bool(z & bit)
    if xb and zb:
        sign = -sign  # S Y Sdg = -X
    if xb:
        z ^= bit
    return x, z, sign


def apply_CNOT_to_masks(x: int, z: int, sign: int, control: int, target: int) -> Tuple[int, int, int]:
    pc = _local_code(x, z, control)
    pt = _local_code(x, z, target)
    sign *= int(_CNOT_PHASE[pc, pt])

    cbit = 1 << control
    tbit = 1 << target
    if x & cbit:
        x ^= tbit
    if z & tbit:
        z ^= cbit
    return x, z, sign


def parse_factor_description(desc: str) -> ParsedGate:
    """Parse a factor description like 'H(0)', 'Sdg(2)', or 'CNOT(0->3)'."""
    if desc.startswith("Sdg("):
        return ("Sdg", int(desc[4:].strip("()")))
    if desc.startswith("H("):
        return ("H", int(desc[2:].strip("()")))
    if desc.startswith("S("):
        return ("S", int(desc[2:].strip("()")))
    if desc.startswith("CNOT("):
        inside = desc[5:].strip("()")
        c, t = inside.split("->")
        return ("CNOT", int(c), int(t))
    raise ValueError(f"Unknown factor description: {desc}")


def parse_factor_descriptions(factor_descriptions: Sequence[Union[str, ParsedGate]]) -> List[ParsedGate]:
    parsed: List[ParsedGate] = []
    for desc in factor_descriptions:
        if isinstance(desc, str):
            parsed.append(parse_factor_description(desc))
        else:
            parsed.append(desc)
    return parsed


def invert_clifford_factor_sequence(
    factor_descriptions: Sequence[Union[str, ParsedGate]],
) -> List[ParsedGate]:
    """Return the gate sequence implementing the inverse Clifford."""
    inverse: List[ParsedGate] = []
    for gate in reversed(parse_factor_descriptions(factor_descriptions)):
        name = gate[0]
        if name == "Sdg":
            inverse.append(("S", int(gate[1])))
        elif name == "S":
            inverse.append(("Sdg", int(gate[1])))
        elif name == "H":
            inverse.append(("H", int(gate[1])))
        elif name == "CNOT":
            inverse.append(("CNOT", int(gate[1]), int(gate[2])))
        else:
            raise ValueError(f"Unknown parsed gate: {gate!r}")
    return inverse


def conjugate_single_term_by_parsed_gates_exact(
    term: Term,
    coeff: complex,
    parsed_gates: Sequence[ParsedGate],
    n_qubits: Optional[int] = None,
) -> Tuple[Term, complex]:
    """Exactly conjugate one Pauli term by parsed Clifford gates."""
    x, z = term_to_masks(term)
    sign = 1
    if n_qubits is None:
        n_qubits = max((q for q, _ in term), default=-1) + 1

    for gate in parsed_gates:
        name = gate[0]
        if name == "H":
            x, z, sign = apply_H_to_masks(x, z, sign, int(gate[1]))
        elif name == "Sdg":
            x, z, sign = apply_Sdg_to_masks(x, z, sign, int(gate[1]))
        elif name == "S":
            x, z, sign = apply_S_to_masks(x, z, sign, int(gate[1]))
        elif name == "CNOT":
            x, z, sign = apply_CNOT_to_masks(x, z, sign, int(gate[1]), int(gate[2]))
        else:
            raise ValueError(f"Unknown parsed gate: {gate!r}")

    return masks_to_term(x, z, n_qubits), coeff * sign


def conjugate_single_term_by_factor_sequence_exact(
    term: Term,
    coeff: complex,
    factor_descriptions: Sequence[Union[str, ParsedGate]],
    n_qubits: Optional[int] = None,
) -> Tuple[Term, complex]:
    """Exactly conjugate one Pauli term by a Clifford factor sequence."""
    return conjugate_single_term_by_parsed_gates_exact(
        term,
        coeff,
        parse_factor_descriptions(factor_descriptions),
        n_qubits=n_qubits,
    )


def conjugate_single_pauli_by_factor_sequence_exact(
    pauli_op: QubitOperator,
    factor_descriptions: Sequence[Union[str, ParsedGate]],
    n_qubits: Optional[int] = None,
) -> QubitOperator:
    """Exactly conjugate a single Pauli-string QubitOperator."""
    if n_qubits is None:
        n_qubits = qubit_operator_num_qubits(pauli_op)
    coeff, _ = single_pauli_term(pauli_op)
    (term, _), = pauli_op.terms.items()
    new_term, new_coeff = conjugate_single_term_by_factor_sequence_exact(
        term,
        coeff,
        factor_descriptions,
        n_qubits=n_qubits,
    )
    return QubitOperator(new_term, new_coeff)


def conjugate_qubit_operator_by_clifford_factors_exact(
    op: QubitOperator,
    factor_descriptions: Sequence[Union[str, ParsedGate]],
    n_qubits: Optional[int] = None,
    compress_abs_tol: float = 1e-12,
) -> QubitOperator:
    """Exactly conjugate an arbitrary QubitOperator by a Clifford sequence."""
    if n_qubits is None:
        n_qubits = qubit_operator_num_qubits(op)
    parsed = parse_factor_descriptions(factor_descriptions)
    transformed = QubitOperator()
    for term, coeff in op.terms.items():
        new_term, new_coeff = conjugate_single_term_by_parsed_gates_exact(
            term,
            coeff,
            parsed,
            n_qubits=n_qubits,
        )
        transformed += QubitOperator(new_term, new_coeff)
    transformed.compress(abs_tol=compress_abs_tol)
    return transformed


# Backward-compatible exact-only aliases. These intentionally no longer provide
# a phase-dropping path.
conjugate_single_pauli_by_factor_sequence = conjugate_single_pauli_by_factor_sequence_exact
conjugate_qubit_operator_by_clifford_factors = conjugate_qubit_operator_by_clifford_factors_exact


def inverse_conjugate_qubit_operator_by_clifford_factors_exact(
    op: QubitOperator,
    factor_descriptions: Sequence[Union[str, ParsedGate]],
    n_qubits: Optional[int] = None,
    compress_abs_tol: float = 1e-12,
) -> QubitOperator:
    """Return C† op C for the Clifford C described by the factor sequence."""
    return conjugate_qubit_operator_by_clifford_factors_exact(
        op,
        invert_clifford_factor_sequence(factor_descriptions),
        n_qubits=n_qubits,
        compress_abs_tol=compress_abs_tol,
    )


# ============================================================
# Synthesis result and ordered symmetry Clifford synthesis
# ============================================================


@dataclass
class CliffordSynthesisResult:
    mapped_qubits: List[int]
    factor_descriptions: List[str]
    parsed_gates: List[ParsedGate]
    transformed_generators: List[QubitOperator]
    elementary_factors: Optional[List[QubitOperator]] = None
    full_clifford: Optional[QubitOperator] = None


def _validate_symmetry_coeff(coeff: complex, *, atol: float = 1e-12) -> None:
    if not np.isclose(coeff.imag, 0.0, atol=atol) or not np.isclose(abs(coeff.real), 1.0, atol=atol):
        raise ValueError("Each symmetry coefficient must be real ±1 for a Hermitian Pauli symmetry.")


def synthesize_ordered_symmetry_clifford(
    symmetries: Sequence[QubitOperator],
    n_qubits: Optional[int] = None,
    return_full_clifford: bool = False,
    return_elementary_factors: bool = False,
) -> CliffordSynthesisResult:
    """Synthesize a Clifford mapping an ordered independent commuting set to Z pivots.

    Later rows may be multiplied by earlier rows during elimination. Therefore
    ``transformed_generators`` are the row-reduced generator basis, not
    necessarily the direct image of each original input generator.
    """
    if len(symmetries) == 0:
        raise ValueError("Need at least one symmetry.")

    if n_qubits is None:
        n_qubits = max(qubit_operator_num_qubits(s) for s in symmetries)

    rows_x: List[np.ndarray] = []
    rows_z: List[np.ndarray] = []
    for s in symmetries:
        coeff, pauli_map = single_pauli_term(s)
        _validate_symmetry_coeff(coeff)
        x, z = binary_from_pauli_map(pauli_map, n_qubits)
        if not np.any(x | z):
            raise ValueError("Identity is not an independent nontrivial symmetry generator.")
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
    mapped_set = set()
    factor_descriptions: List[str] = []
    parsed_gates: List[ParsedGate] = []

    def add_gate(desc: str) -> None:
        factor_descriptions.append(desc)
        parsed_gates.append(parse_factor_description(desc))

    for i in range(m):
        # Clear earlier pivot Zs from row i by multiplying with earlier rows.
        for j, q_prev in enumerate(mapped_qubits):
            if xs[i, q_prev] != 0:
                raise RuntimeError("Invariant violated: later row has X/Y on an earlier symmetry qubit.")
            if zs[i, q_prev] == 1:
                xs[i] ^= xs[j]
                zs[i] ^= zs[j]

        support = [q for q in range(n_qubits) if (xs[i, q] or zs[i, q]) and q not in mapped_set]
        if not support:
            raise ValueError(
                f"Symmetry {i} became trivial after hierarchical clearing; input set is dependent."
            )

        pivot = support[0]
        mapped_qubits.append(pivot)
        mapped_set.add(pivot)

        active_support = [q for q in range(n_qubits) if xs[i, q] or zs[i, q]]
        for q in active_support:
            if q in mapped_qubits[:-1]:
                if xs[i, q] or zs[i, q]:
                    raise RuntimeError("Failed to clear previous pivot support.")
                continue

            if xs[i, q] == 1 and zs[i, q] == 0:
                pass
            elif xs[i, q] == 1 and zs[i, q] == 1:
                apply_Sdg_to_rows(xs, zs, q)
                add_gate(f"Sdg({q})")
            elif xs[i, q] == 0 and zs[i, q] == 1:
                apply_H_to_rows(xs, zs, q)
                add_gate(f"H({q})")
            else:
                raise RuntimeError("Unexpected local Pauli state.")

        active_support = [q for q in range(n_qubits) if xs[i, q] or zs[i, q]]
        for q in list(active_support):
            if q == pivot:
                continue
            apply_CNOT_to_rows(xs, zs, pivot, q)
            add_gate(f"CNOT({pivot}->{q})")

        apply_H_to_rows(xs, zs, pivot)
        add_gate(f"H({pivot})")

    transformed_generators = [
        pauli_dict_to_qubit_operator(pauli_map_from_binary(xs[i], zs[i]))
        for i in range(m)
    ]

    elementary_factors: Optional[List[QubitOperator]] = None
    if return_elementary_factors or return_full_clifford:
        elementary_factors = [factor_from_parsed_gate(g) for g in parsed_gates]

    full_clifford = None
    if return_full_clifford:
        full_clifford = I_op()
        assert elementary_factors is not None
        for U in elementary_factors:
            full_clifford = U * full_clifford
        full_clifford.compress(abs_tol=1e-12)

    return CliffordSynthesisResult(
        mapped_qubits=mapped_qubits,
        factor_descriptions=factor_descriptions,
        parsed_gates=parsed_gates,
        transformed_generators=transformed_generators,
        elementary_factors=elementary_factors,
        full_clifford=full_clifford,
    )


# ============================================================
# Sector ordering utilities
# ============================================================


def int_to_bitstring(x: int, n_qubits: int) -> Tuple[int, ...]:
    """Return bits in qubit order q=0..n-1 for OpenFermion matrix basis index x.

    OpenFermion sparse matrices order computational basis states with qubit 0
    as the most-significant bit.
    """
    return tuple((x >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits))


def symmetry_sector_label(
    basis_index: int,
    symmetry_qubits: Sequence[int],
    n_qubits: Optional[int] = None,
) -> Tuple[int, ...]:
    """Return sector bits for an OpenFermion matrix basis index.

    If n_qubits is omitted, it is inferred from the largest requested qubit.
    Passing n_qubits explicitly is recommended.
    """
    if n_qubits is None:
        n_qubits = max(symmetry_qubits, default=-1) + 1
    return tuple((basis_index >> (n_qubits - 1 - q)) & 1 for q in symmetry_qubits)


def sector_ordering_from_symmetry_qubits(
    n_qubits: int,
    symmetry_qubits: Sequence[int],
    residual_qubit_order: Optional[Sequence[int]] = None,
) -> Tuple[List[int], Dict[Tuple[int, ...], List[int]], List[Tuple[int, ...]]]:
    """Build basis ordering grouped by symmetry-sector labels."""
    symmetry_qubits = list(symmetry_qubits)
    if len(set(symmetry_qubits)) != len(symmetry_qubits):
        raise ValueError("symmetry_qubits contains duplicates.")
    for q in symmetry_qubits:
        if q < 0 or q >= n_qubits:
            raise ValueError("symmetry qubit out of range.")

    symmetry_set = set(symmetry_qubits)
    residual_qubits = [q for q in range(n_qubits) if q not in symmetry_set]
    if residual_qubit_order is None:
        residual_qubit_order = residual_qubits
    else:
        residual_qubit_order = list(residual_qubit_order)
        if sorted(residual_qubit_order) != sorted(residual_qubits):
            raise ValueError("residual_qubit_order must contain exactly the non-symmetry qubits.")

    sector_to_indices: Dict[Tuple[int, ...], List[int]] = {}
    for state in range(1 << n_qubits):
        sec = symmetry_sector_label(state, symmetry_qubits, n_qubits=n_qubits)
        sector_to_indices.setdefault(sec, []).append(state)

    def residual_key(state: int) -> Tuple[int, ...]:
        return tuple((state >> (n_qubits - 1 - q)) & 1 for q in residual_qubit_order)  # type: ignore[arg-type]

    for sec in sector_to_indices:
        sector_to_indices[sec].sort(key=residual_key)

    ordered_sectors = sorted(sector_to_indices.keys())
    ordered_basis_indices: List[int] = []
    for sec in ordered_sectors:
        ordered_basis_indices.extend(sector_to_indices[sec])

    return ordered_basis_indices, sector_to_indices, ordered_sectors


def permutation_matrix_from_order(order: Sequence[int], sparse_output: bool = True):
    """Build P such that H_reordered = P H P.T. Prefer sparse output."""
    dim = len(order)
    rows = np.arange(dim)
    cols = np.asarray(order)
    data = np.ones(dim, dtype=np.float64)
    P = sparse.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    if sparse_output:
        return P
    return P.toarray()


# ============================================================
# Reordered Hamiltonian matrix and plotting
# ============================================================


@dataclass
class ReorderedHamiltonianResult:
    transformed_hamiltonian: QubitOperator
    transformed_matrix: sparse.csr_matrix
    reordered_matrix: sparse.csr_matrix
    basis_order: List[int]
    ordered_sectors: List[Tuple[int, ...]]
    sector_boundaries: List[int]


def reordered_matrix_by_sector(
    hamiltonian: QubitOperator,
    symmetry_qubits: Sequence[int],
    factor_descriptions: Sequence[Union[str, ParsedGate]],
    n_qubits: Optional[int] = None,
) -> ReorderedHamiltonianResult:
    """Transform Hamiltonian and reorder sparse matrix by symmetry sectors."""
    if n_qubits is None:
        n_qubits = qubit_operator_num_qubits(hamiltonian)

    if len(factor_descriptions) > 0:
        transformed_h = conjugate_qubit_operator_by_clifford_factors_exact(
            hamiltonian,
            factor_descriptions=factor_descriptions,
            n_qubits=n_qubits,
        )
    else:
        transformed_h = hamiltonian

    H_sparse = get_sparse_operator(transformed_h, n_qubits=n_qubits).tocsr()

    basis_order, sector_to_indices, ordered_sectors = sector_ordering_from_symmetry_qubits(
        n_qubits=n_qubits,
        symmetry_qubits=symmetry_qubits,
    )
    idx = np.asarray(basis_order, dtype=np.int64)
    H_reordered = H_sparse[idx, :][:, idx].tocsr()

    sector_boundaries: List[int] = []
    running = 0
    for sec in ordered_sectors:
        running += len(sector_to_indices[sec])
        sector_boundaries.append(running)

    return ReorderedHamiltonianResult(
        transformed_hamiltonian=transformed_h,
        transformed_matrix=H_sparse,
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
    """Plot the reordered Hamiltonian matrix with sector boundaries overlaid."""
    H = reordered_result.reordered_matrix
    H_dense = H.toarray() if sparse.issparse(H) else np.asarray(H)
    if use_log10_abs:
        plot_data = np.log10(np.abs(H_dense) + eps)
        cbar_label = r"$\log_{10}(|H_{ij}|+\epsilon)$"
    else:
        plot_data = np.abs(H_dense)
        cbar_label = r"$|H_{ij}|$"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_data, origin="lower", interpolation="nearest", aspect="equal")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    for b in reordered_result.sector_boundaries[:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Basis index (reordered)")
    ax.set_ylabel("Basis index (reordered)")
    plt.tight_layout()
    plt.show()


# ============================================================
# End-to-end pipeline
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
        factor_descriptions=clifford_result.parsed_gates,
        n_qubits=n_qubits,
    )
    return SymmetryBlockStructureResult(clifford_result, reordered_result)


# ============================================================
# Qubit permutation helpers
# ============================================================


def invert_permutation(perm: Sequence[int]) -> List[int]:
    n = len(perm)
    inv: List[Optional[int]] = [None] * n
    for old_q, new_q in enumerate(perm):
        if not (0 <= new_q < n):
            raise ValueError("Invalid permutation entry.")
        if inv[new_q] is not None:
            raise ValueError("Permutation is not one-to-one.")
        inv[new_q] = old_q
    return [int(q) for q in inv]


def permute_qubits_in_term(term: Term, perm: Sequence[int]) -> Term:
    return tuple(sorted((perm[q], p) for q, p in term))


def permute_qubits_in_qubit_operator(
    op: QubitOperator,
    perm: Sequence[int],
    compress_abs_tol: float = 1e-12,
) -> QubitOperator:
    out = QubitOperator()
    for term, coeff in op.terms.items():
        out += QubitOperator(permute_qubits_in_term(term, perm), coeff)
    out.compress(abs_tol=compress_abs_tol)
    return out


def permute_qubit_list(qubits: Sequence[int], perm: Sequence[int]) -> List[int]:
    return [perm[q] for q in qubits]


def make_symmetry_qubits_last_permutation(
    n_qubits: int,
    symmetry_qubits: Sequence[int],
) -> Tuple[List[int], List[int]]:
    symmetry_qubits = list(symmetry_qubits)
    symmetry_set = set(symmetry_qubits)
    if len(symmetry_set) != len(symmetry_qubits):
        raise ValueError("symmetry_qubits contains duplicates.")
    for q in symmetry_qubits:
        if not (0 <= q < n_qubits):
            raise ValueError("symmetry qubit out of range.")

    nonsym = [q for q in range(n_qubits) if q not in symmetry_set]
    new_order_old_qubits = nonsym + symmetry_qubits
    perm: List[Optional[int]] = [None] * n_qubits
    for new_q, old_q in enumerate(new_order_old_qubits):
        perm[old_q] = new_q
    perm_final = [int(q) for q in perm]
    return perm_final, [perm_final[q] for q in symmetry_qubits]


@dataclass
class PermutedHamiltonianResult:
    permuted_hamiltonian: QubitOperator
    qubit_permutation: List[int]
    permuted_symmetry_qubits: List[int]


def move_symmetry_qubits_to_end(
    transformed_hamiltonian: QubitOperator,
    mapped_qubits: Sequence[int],
    n_qubits: int,
) -> PermutedHamiltonianResult:
    perm, new_mapped = make_symmetry_qubits_last_permutation(n_qubits, mapped_qubits)
    H_perm = permute_qubits_in_qubit_operator(transformed_hamiltonian, perm=perm)
    return PermutedHamiltonianResult(H_perm, perm, new_mapped)


def permute_hamiltonian_qubits(
    transformed_hamiltonian: QubitOperator,
    perm: Sequence[int],
    sym_qubits: Sequence[int],
    validate: bool = False,
) -> PermutedHamiltonianResult:
    _ = validate
    H_perm = permute_qubits_in_qubit_operator(transformed_hamiltonian, perm=perm)
    return PermutedHamiltonianResult(H_perm, list(perm), permute_qubit_list(sym_qubits, perm=perm))


@dataclass
class SymmetryBlockStructurePackedResult:
    clifford_result: CliffordSynthesisResult
    transformed_hamiltonian: QubitOperator
    packed_hamiltonian: QubitOperator
    original_mapped_qubits: List[int]
    packed_symmetry_qubits: List[int]
    qubit_permutation: List[int]
    reordered_matrix: Optional[sparse.csr_matrix]
    ordered_sectors: Optional[List[Tuple[int, ...]]]
    sector_boundaries: Optional[List[int]]


def build_symmetry_block_structure_with_packed_qubits(
    hamiltonian: QubitOperator,
    symmetries: Sequence[QubitOperator],
    n_qubits: int,
    return_full_clifford: bool = False,
    reorder_sector: bool = False,
) -> SymmetryBlockStructurePackedResult:
    clifford_result = synthesize_ordered_symmetry_clifford(
        symmetries=symmetries,
        n_qubits=n_qubits,
        return_full_clifford=return_full_clifford,
    )
    transformed_h = conjugate_qubit_operator_by_clifford_factors_exact(
        hamiltonian,
        factor_descriptions=clifford_result.parsed_gates,
        n_qubits=n_qubits,
    )
    packed = move_symmetry_qubits_to_end(transformed_h, clifford_result.mapped_qubits, n_qubits)

    reordered_matrix = None
    ordered_sectors = None
    sector_boundaries = None
    if reorder_sector:
        reordered = reordered_matrix_by_sector(
            hamiltonian=packed.permuted_hamiltonian,
            symmetry_qubits=packed.permuted_symmetry_qubits,
            factor_descriptions=[],
            n_qubits=n_qubits,
        )
        reordered_matrix = reordered.reordered_matrix
        ordered_sectors = reordered.ordered_sectors
        sector_boundaries = reordered.sector_boundaries

    return SymmetryBlockStructurePackedResult(
        clifford_result=clifford_result,
        transformed_hamiltonian=transformed_h,
        packed_hamiltonian=packed.permuted_hamiltonian,
        original_mapped_qubits=clifford_result.mapped_qubits,
        packed_symmetry_qubits=packed.permuted_symmetry_qubits,
        qubit_permutation=packed.qubit_permutation,
        reordered_matrix=reordered_matrix,
        ordered_sectors=ordered_sectors,
        sector_boundaries=sector_boundaries,
    )

### sparse construction utilities
from scipy.sparse import csr_matrix

def sparse_qubit_permutation_unitary(
    perm: Sequence[int],
    msb_ordering: bool = True,
    dtype=complex,
) -> csr_matrix:
    """
    Build the sparse unitary matrix implementing a qubit permutation.

    Parameters
    ----------
    perm:
        Qubit permutation specified by

            perm[old_qubit] = new_qubit.

        For example, perm = [2, 0, 1] means

            old qubit 0 -> new qubit 2
            old qubit 1 -> new qubit 0
            old qubit 2 -> new qubit 1

    msb_ordering:
        If True, basis indices are interpreted with qubit 0 as the
        most-significant bit, consistent with OpenFermion matrix convention.

        If False, qubit 0 is interpreted as the least-significant bit.

    dtype:
        Matrix dtype.

    Returns
    -------
    U:
        Sparse permutation unitary such that

            |new_bits> = U |old_bits>,

        where

            new_bits[perm[q]] = old_bits[q].

        For an operator H, the permuted operator matrix is

            H_perm = U H U.conj().T.
    """
    perm = list(perm)
    n_qubits = len(perm)

    if sorted(perm) != list(range(n_qubits)):
        raise ValueError("perm must be a valid permutation of 0, ..., n_qubits-1.")

    dim = 1 << n_qubits

    rows = np.empty(dim, dtype=np.int64)
    cols = np.arange(dim, dtype=np.int64)
    data = np.ones(dim, dtype=dtype)

    for old_index in range(dim):
        new_index = 0

        for old_q, new_q in enumerate(perm):
            if msb_ordering:
                old_bit = (old_index >> (n_qubits - 1 - old_q)) & 1
                if old_bit:
                    new_index |= 1 << (n_qubits - 1 - new_q)
            else:
                old_bit = (old_index >> old_q) & 1
                if old_bit:
                    new_index |= 1 << new_q

        rows[old_index] = new_index

    U = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=dtype)
    return U

def sparse_clifford_unitary(clifford: CliffordSynthesisResult, n_qubits):
    """
    Return sparse matrix representing clifford
    """
    Ucliff = sparse.identity(1<<n_qubits)

    for gate in clifford.parsed_gates:
        factor = get_sparse_operator(factor_from_parsed_gate(gate), n_qubits)
        Ucliff = factor @ Ucliff
    return Ucliff
