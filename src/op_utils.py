
from openfermion import MolecularData, QubitOperator
#from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator
import numpy as np

BASIS_NAME = "sto-3g"
MULTIPLICITY = 1
CHARGE = 0

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