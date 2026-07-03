
from openfermion import MolecularData, QubitOperator
#from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator
import numpy as np
from .clifford_symmetry_optimized import Clifford

BASIS_NAME = "sto-3g"
MULTIPLICITY = 1
CHARGE = 0


def permute_sym_to_start(
    hamiltonian,
    symmetries,
    n_qubits,
    verbose=False,
    return_clifford_perm=False,
):
    """Map symmetry generators to leading Z qubits and transform an operator.

    The returned :class:`Clifford` contains both the synthesized Clifford and
    final qubit permutation. When ``return_clifford_perm`` is true, this keeps
    the historical ``(operator, clifford, permutation)`` return shape.
    """
    clifford = Clifford.from_symmetries(
        symmetries,
        n_qubits=n_qubits,
        symmetry_qubits_first=True,
    )
    transformed = clifford.transform(hamiltonian)

    if verbose:
        print("Symmetries rotated to Z on qubits: ", clifford.mapped_qubits)
        print("Qubits permuted as:")
        for old_q, new_q in enumerate(clifford.permutation):
            print(old_q, "->", new_q)

    if return_clifford_perm:
        return transformed, clifford, list(clifford.permutation)
    return transformed

def linear_h4_geometry(R, n_H=4):
    """
    Build molecular geometry for linear H4 chain.
    
    Parameters
    ----------
    R : float
        Bond distance in Angstroms.
        
    Returns
    -------
    list of tuple
        List of (atom_symbol, (x, y, z)) tuples.
    """
    assert n_H % 2 == 0, "Odd number {} of Hydrogens specified.".format(n_H)
    
    return [("H", (0.0, 0.0, i*R)) for i in range(n_H)]

def h2o_geometry(bond_length, bond_angle_deg):
    theta = np.deg2rad(bond_angle_deg)
    half = theta / 2.0

    geometry = [
        ('O', (0.0, 0.0, 0.0)),
        ('H', ( bond_length * np.sin(half), 0.0, bond_length * np.cos(half))),
        ('H', (-bond_length * np.sin(half), 0.0, bond_length * np.cos(half))),
    ]
    return geometry

def build_H_chain_for_R(R, n_H=4):
    """
    Build Hamiltonian for H4 at distance R.
    
    Parameters
    ----------
    R : float
        Bond distance in Angstroms.
        
    Returns
    -------
    H : FermionOperator
        Hamiltonian.
    mol : MolecularData
        OpenFermion molecule object with computed properties.
    """
    geom = linear_h4_geometry(R, n_H)
    mol = MolecularData(geom, BASIS_NAME, MULTIPLICITY, CHARGE)
    mol = run_pyscf(mol, run_scf=1, run_fci=1)
    H_mol = mol.get_molecular_hamiltonian()
    H_ferm = get_fermion_operator(H_mol)
    return H_ferm, mol

def lih_geometry(bl):
    return [
    ('Li', (0.0, 0.0, -bl/2)),
    ('H', (0.0, 0.0, bl/2))
]

def h4_sq_geometry(bl):
    return [
        ('H', (-bl/2, -bl/2, 0.0)),
        ('H', (-bl/2, bl/2, 0.0)),
        ('H', (bl/2, -bl/2, 0.0)),
        ('H', (bl/2, bl/2, 0.0))
    ]

def h4_chain_geometry(bl):
    return [
        ('H', (-1.5*bl, 0.0, 0.0)),
        ('H', (-0.5*bl, 0.0, 0.0)),
        ('H', (0.5*bl, 0.0, 0.0)),
        ('H', (1.5*bl, 0.0, 0.0))
    ]

def truncate_qubitop(H, eps):
    """
    Truncate qubit operator by dropping terms with |coeff| < eps.
    
    Parameters
    ----------
    H : QubitOperator
        Input Hamiltonian.
    eps : float
        Truncation threshold.
        
    Returns
    -------
    QubitOperator
        Truncated Hamiltonian.
    """
    out = QubitOperator()
    for term, coeff in H.terms.items():
        if abs(coeff) >= eps:
            if abs(coeff.imag) < 1e-12:
                coeff = coeff.real
            out += QubitOperator(term, coeff)
    return out

def freeze_qubits(op: QubitOperator, frozen: dict[int, int]) -> QubitOperator:
    """
    Reduce an OpenFermion QubitOperator by freezing selected qubits to |0> or |1>.

    Args:
        op:
            OpenFermion QubitOperator.
        frozen:
            Dictionary {qubit_index: value}, where value is 0 or 1.

    Returns:
        A QubitOperator acting only on the unfrozen qubits. Qubit indices are
        compacted so that removed qubits disappear.

    Rule:
        Z_i -> +1 on |0>
        Z_i -> -1 on |1>
        X_i, Y_i -> 0 because they take |0>/<1> out of the frozen subspace.
    """
    frozen = dict(frozen)

    for q, val in frozen.items():
        if val not in (0, 1):
            raise ValueError(f"Frozen qubit {q} has value {val}; expected 0 or 1.")

    frozen_qubits = set(frozen)

    def remap_index(q: int) -> int:
        """Map old qubit index to new compacted index."""
        return q - sum(f < q for f in frozen_qubits)

    reduced = QubitOperator.zero()

    for term, coeff in op.terms.items():
        new_coeff = coeff
        new_term = []
        term_vanishes = False

        for q, pauli in term:
            if q in frozen_qubits:
                val = frozen[q]

                if pauli == "Z":
                    # Z|0> = +|0>, Z|1> = -|1>
                    if val == 1:
                        new_coeff *= -1

                elif pauli in ("X", "Y"):
                    # X and Y connect |0> <-> |1>, so projected expectation is zero
                    term_vanishes = True
                    break

                else:
                    raise ValueError(f"Unknown Pauli operator {pauli!r} on qubit {q}.")

            else:
                new_term.append((remap_index(q), pauli))

        if not term_vanishes:
            reduced += QubitOperator(tuple(new_term), new_coeff)

    return reduced


def split_pauli_operator_seniority(term, n_tapered: int):
    """Split a Pauli string into the first ``n_tapered`` qubits and the rest.

    This retains the historical function name used by the seniority code, but
    the operation itself is generic and can be used after
    ``permute_sym_to_start``.
    """
    if n_tapered < 0:
        raise ValueError("n_tapered must be nonnegative.")

    if isinstance(term, QubitOperator):
        if len(term.terms) != 1:
            raise ValueError("Expected a QubitOperator containing one Pauli term.")
        term = next(iter(term.terms))

    term = tuple(term)
    tapered_part = tuple(item for item in term if item[0] < n_tapered)
    remaining_part = tuple(item for item in term if item[0] >= n_tapered)
    return tapered_part, remaining_part


def _validate_basis_labels(labels, name: str):
    labels = tuple(int(label) for label in labels)
    if any(label not in (0, 1) for label in labels):
        raise ValueError(f"{name} must contain only computational-basis labels 0 or 1.")
    return labels


def pauli_matrix_element_with_basis_state(term, bra_labels, ket_labels):
    """Evaluate ``<bra_labels|term|ket_labels>`` for a Pauli string.

    Qubit labels follow OpenFermion order. In particular,
    ``<0|Y|1> = -1j`` and ``<1|Y|0> = +1j``.
    """
    if isinstance(term, QubitOperator):
        if len(term.terms) != 1:
            raise ValueError("Expected a QubitOperator containing one Pauli term.")
        term = next(iter(term.terms))

    bra_labels = _validate_basis_labels(bra_labels, "bra_labels")
    ket_labels = _validate_basis_labels(ket_labels, "ket_labels")
    if len(bra_labels) != len(ket_labels):
        raise ValueError("bra_labels and ket_labels must have equal length.")

    paulis = dict(term)
    if any(q < 0 or q >= len(bra_labels) for q in paulis):
        raise ValueError("The Pauli term acts outside the supplied basis labels.")

    value = 1.0 + 0.0j
    for q, (bra, ket) in enumerate(zip(bra_labels, ket_labels)):
        pauli = paulis.get(q)
        if pauli is None:
            if bra != ket:
                return 0.0
        elif pauli == "X":
            if bra == ket:
                return 0.0
        elif pauli == "Y":
            if bra == ket:
                return 0.0
            value *= -1.0j if (bra, ket) == (0, 1) else 1.0j
        elif pauli == "Z":
            if bra != ket:
                return 0.0
            if bra == 1:
                value *= -1.0
        else:
            raise ValueError(f"Unsupported Pauli operator {pauli!r}.")
    return np.real_if_close(value).item()


def taper_pauli_term(term, bra_labels, ket_labels):
    """Return ``<bra|term[:N]|ket> term[N:]`` without reindexing qubits."""
    bra_labels = _validate_basis_labels(bra_labels, "bra_labels")
    ket_labels = _validate_basis_labels(ket_labels, "ket_labels")
    if len(bra_labels) != len(ket_labels):
        raise ValueError("bra_labels and ket_labels must have equal length.")

    tapered_part, remaining_part = split_pauli_operator_seniority(
        term, len(bra_labels)
    )
    matrix_element = pauli_matrix_element_with_basis_state(
        tapered_part, bra_labels, ket_labels
    )
    return matrix_element * QubitOperator(remaining_part)


def shift_hamiltonian_qubits_uniformly(op: QubitOperator, shift: int) -> QubitOperator:
    """Remove ``shift`` leading identity qubits from a QubitOperator."""
    if shift < 0:
        raise ValueError("shift must be nonnegative.")

    shifted = QubitOperator.zero()
    for term, coefficient in op.terms.items():
        if any(q < shift for q, _pauli in term):
            raise ValueError(
                "Cannot shift: the operator still acts on a removed leading qubit."
            )
        shifted_term = tuple((q - shift, pauli) for q, pauli in term)
        shifted += QubitOperator(shifted_term, coefficient)
    shifted.compress()
    return shifted


def taper_hamiltonian(
    hamiltonian: QubitOperator,
    bra_labels,
    ket_labels,
    shift_to_zero: bool = True,
) -> QubitOperator:
    """Project leading qubits onto specified bra and ket basis labels.

    If ``N = len(bra_labels)``, this returns the effective operator

        ``<bra_labels| hamiltonian |ket_labels>``

    acting on the untapered qubits. This supports both diagonal sectors and
    off-diagonal blocks, including the complex phases generated by Pauli Y.
    It is intended for Hamiltonians returned by ``permute_sym_to_start``, where
    the symmetry qubits occupy indices ``0, ..., N - 1``.

    With ``shift_to_zero=True``, remaining indices ``N, N+1, ...`` are compacted
    to ``0, 1, ...``.
    """
    bra_labels = _validate_basis_labels(bra_labels, "bra_labels")
    ket_labels = _validate_basis_labels(ket_labels, "ket_labels")
    if len(bra_labels) != len(ket_labels):
        raise ValueError("bra_labels and ket_labels must have equal length.")

    tapered = QubitOperator.zero()
    for term, coefficient in hamiltonian.terms.items():
        tapered += coefficient * taper_pauli_term(
            term, bra_labels, ket_labels
        )
    tapered.compress()

    if shift_to_zero:
        return shift_hamiltonian_qubits_uniformly(
            tapered, len(bra_labels)
        )
    return tapered

def taper_symmetries(HQ, symmetries, bra_labels, ket_labels, n_qubits, verbose=False):
    """
    Rotate symmetries to the start and then taper for given labels

    HQ: QubitOperator
    symmetries: list[QubitOperator]
    bra_labels, ket_labels: list[int] bra and ket labels of 0, 1
    
    """
    n_sym = len(symmetries)
    assert n_sym == len(bra_labels) and n_sym == len(ket_labels), "invalid number of sector labels {}, {} passed for {} symmetries".format(len(bra_labels), len(ket_labels), n_sym)

    HQ_perm, clifford, perm = permute_sym_to_start(HQ, symmetries, n_qubits, verbose=verbose, return_clifford_perm=True)

    HQ_tapered = taper_hamiltonian(HQ_perm, bra_labels, ket_labels, True)

    return HQ_tapered

def has_complex_entries(HQ: QubitOperator, tol: float = 1e-12) -> bool:
    """
    Return True if QubitOperator H has complex-valued matrix entries
    in the computational basis.

    A Pauli term is imaginary-valued if it has an odd number of Y operators.
    Therefore:
      - odd # of Y with real coeff -> complex entries
      - even # of Y with complex coeff -> complex entries
      - odd # of Y with imaginary coeff -> real entries, up to phase
    """
    for term, coeff in HQ.terms.items():
        num_y = sum(pauli == "Y" for _, pauli in term)

        coeff_real = abs(coeff.real) > tol
        coeff_imag = abs(coeff.imag) > tol

        if num_y % 2 == 0:
            # Pauli string is real, so imaginary coeff gives complex entries
            if coeff_imag:
                return True
        else:
            # Pauli string is imaginary, so real coeff gives complex entries
            if coeff_real:
                return True

    return False

def split_diagonal_paulis(op: QubitOperator) -> tuple[QubitOperator, QubitOperator]:
    """
    Split a QubitOperator into computational-basis diagonal and non-diagonal parts.

    The diagonal part contains only identity and Pauli strings made entirely of Z
    operators. Any term containing X or Y is placed in the non-diagonal part.

    Args:
        op:
            OpenFermion QubitOperator to split.

    Returns:
        (diagonal, non_diagonal), both QubitOperator objects.
    """
    diagonal = QubitOperator.zero()
    non_diagonal = QubitOperator.zero()

    for term, coeff in op.terms.items():
        if all(pauli == "Z" for _, pauli in term):
            diagonal += QubitOperator(term, coeff)
        else:
            non_diagonal += QubitOperator(term, coeff)

    return diagonal, non_diagonal

class PauliStringAction:
    """
    Fast action of a single QubitOperator Pauli product on state vectors.

    OpenFermion's sparse convention maps qubit 0 to the most significant bit of
    the computational-basis index, so the masks use bit n_qubits - 1 - q.
    """
    def __init__(self, sym, n_qubits):
        if len(sym.terms) != 1:
            raise ValueError("PauliStringAction expects a single Pauli product.")

        (term, coeff), = sym.terms.items()
        self.n_qubits = n_qubits
        self.coeff = coeff
        self.term = term

        dim = 1 << n_qubits
        indices = np.arange(dim)
        targets = indices.copy()
        phases = np.full(dim, coeff, dtype=complex)

        for q, pauli in term:
            bit = n_qubits - 1 - q
            mask = 1 << bit
            bits = (indices & mask) != 0

            if pauli == "X":
                targets ^= mask
            elif pauli == "Y":
                targets ^= mask
                phases *= np.where(bits, -1.0j, 1.0j)
            elif pauli == "Z":
                phases *= np.where(bits, -1.0, 1.0)
            else:
                raise ValueError("Unknown Pauli operator {}".format(pauli))

        self.targets = targets
        self.phases = phases

    def apply(self, state, out=None):
        psi = np.asarray(state).reshape(-1)
        if out is None:
            out = np.empty_like(psi, dtype=complex)
        out[self.targets] = self.phases * psi
        return out

def prepare_pauli_actions(sym_ops, n_qubits):
    return [PauliStringAction(sym, n_qubits) for sym in sym_ops]

class PauliSumAction:
    """
    Apply a QubitOperator Pauli sum to state vectors without building a sparse matrix.

    If sparse_input=True, only nonzero input amplitudes are propagated.  This is
    useful for CI-like states with few determinants.
    """
    def __init__(self, op, n_qubits):
        self.n_qubits = n_qubits
        self.terms = []
        for term, coeff in op.terms.items():
            flip_mask = 0
            sign_mask = 0
            n_y = 0
            for q, pauli in term:
                bit_mask = 1 << (n_qubits - 1 - q)
                if pauli == "X":
                    flip_mask ^= bit_mask
                elif pauli == "Y":
                    flip_mask ^= bit_mask
                    sign_mask ^= bit_mask
                    n_y += 1
                elif pauli == "Z":
                    sign_mask ^= bit_mask
                else:
                    raise ValueError("Unknown Pauli operator {}".format(pauli))
            self.terms.append((coeff * (1.0j ** n_y), flip_mask, sign_mask))

    @staticmethod
    def _parity(values):
        return np.array([bin(int(v)).count("1") & 1 for v in values], dtype=bool)

    def apply(self, state, out=None, sparse_input=False, tol=1e-12):
        psi = np.asarray(state).reshape(-1)
        if out is None:
            out = np.zeros_like(psi, dtype=complex)
        else:
            out.fill(0.0)

        if sparse_input:
            nz = np.flatnonzero(np.abs(psi) > tol)
            for coeff, flip_mask, sign_mask in self.terms:
                targets = nz ^ flip_mask
                phases = np.full(len(nz), coeff, dtype=complex)
                phases[self._parity(nz & sign_mask)] *= -1.0
                out[targets] += phases * psi[nz]
        else:
            indices = np.arange(len(psi))
            for coeff, flip_mask, sign_mask in self.terms:
                phases = np.full(len(psi), coeff, dtype=complex)
                phases[self._parity(indices & sign_mask)] *= -1.0
                out[indices ^ flip_mask] += phases * psi
        return out

def prepare_pauli_sum_action(op, n_qubits):
    return PauliSumAction(op, n_qubits)
