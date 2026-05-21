
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