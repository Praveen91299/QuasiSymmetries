
from openfermion import MolecularData, QubitOperator
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator

BASIS_NAME = "sto-3g"
MULTIPLICITY = 1
CHARGE = 0

def build_geometry(R, n_H=4):
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
    geom = build_geometry(R, n_H)
    mol = MolecularData(geom, BASIS_NAME, MULTIPLICITY, CHARGE)
    mol = run_pyscf(mol, run_scf=1, run_fci=1)
    H_mol = mol.get_molecular_hamiltonian()
    H_ferm = get_fermion_operator(H_mol)
    return H_ferm, mol

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

