from openfermion import FermionOperator, jordan_wigner, get_sparse_operator, normal_ordered, bravyi_kitaev, QubitOperator, get_fermion_operator, MolecularData, expectation
import numpy as np
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import expm as sparse_expm
import scipy
try:
    from opt_einsum import contract
except ModuleNotFoundError:
    contract = np.einsum


#excitation
Eij = lambda i, j: FermionOperator('{}^ {}'.format(i, j), 1.0)
ni = lambda i: Eij(i, i)

#spin symmetric excitations
#udud
Fij= lambda i, j: Eij(2*i, 2*j) + Eij(2*i+1, 2*j+1)
#uudd
Gij= lambda i, j, n_orb: Eij(i, j) + Eij(n_orb+i, n_orb+j)

def spatial_tbt_to_spin_tbt(tbt, spin_ord="udud"):
    n_orb = len(tbt)
    n_qubits = 2 * n_orb
    tbt_spin = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    if spin_ord == "udud":
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        tbt_spin[2*p, 2*q, 2*r, 2*s] = tbt[p, q, r, s]
                        tbt_spin[2*p+1, 2*q+1, 2*r, 2*s] = tbt[p, q, r, s]
                        tbt_spin[2*p, 2*q, 2*r+1, 2*s+1] = tbt[p, q, r, s]
                        tbt_spin[2*p+1, 2*q+1, 2*r+1, 2*s+1] = tbt[p, q, r, s]
    elif spin_ord == "uudd":
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        tbt_spin[p, q, r, s] = tbt[p, q, r, s]
                        tbt_spin[p+n_orb, q+n_orb, r, s] = tbt[p, q, r, s]
                        tbt_spin[p, q, r+n_orb, s+n_orb] = tbt[p, q, r, s]
                        tbt_spin[p+n_orb, q+n_orb, r+n_orb, s+n_orb] = tbt[p, q, r, s]
    else:
        assert False, "Spin order {} unrecognized.".format(spin_ord)
    return tbt_spin

def spatial_obt_to_spin_obt(obt, spin_ord="udud"):
    if spin_ord =="udud":
        return np.kron(obt, np.identity(2, complex))
    elif spin_ord== "uudd":
        return np.kron(np.identity(2, complex), obt)
    else:
        assert False, "Spin order {} unrecognized.".format(spin_ord)

def chem_ferm_to_chem_tbt(op: FermionOperator, n_qubits, tol = 1e-5):
    tbt = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits), complex)

    constant =  op.constant
    op = op - constant

    for key, coeff in zip(op.terms.keys(), op.terms.values()):

        if abs(coeff) < tol:
            continue
        
        assert len(key) == 4, "Not a two body operator"
        assert key[0][1] == 1 and key[1][1] == 0 and key[2][1] == 1 and key[3][1] == 0, "Operator not in chem two body ordering"

        tbt[key[0][0], key[1][0], key[2][0], key[3][0]] = coeff

    return tbt, constant

def chem_ferm_to_chem_obt(op: FermionOperator, n_qubits, tol=1e-5):
    obt = np.zeros((n_qubits, n_qubits), complex)

    constant = op.constant
    op = op - constant

    for key, coeff in zip(op.terms.keys(), op.terms.values()):

        if abs(coeff) < tol:
            continue
        
        assert len(key) == 2, "Not a two body operator"
        assert key[0][1] == 1 and key[1][1] == 0, "Operator not in chem one body ordering"

        obt[key[0][0], key[1][0]] = coeff
    
    return obt, constant

def rotate_chem_obt(obt, U):
    
    x, y = np.shape(U)
    U_conj = np.conjugate(U)

    a, b = np.shape(obt)
    obt_new = np.zeros(np.shape(obt), complex)

    assert y == a and y == b, "Incompatible rotation matrix."

    tbt_new = contract('pi,qj,ij->pq', U, U_conj, obt)

    return tbt_new

def rotate_chem_tbt(tbt, U):

    x, y = np.shape(U)
    U_conj = np.conjugate(U)

    a, b, c, d = np.shape(tbt)
    tbt_new = np.zeros(np.shape(tbt), complex)

    assert y == a and y == b and y == c and y == d, "Incompatible rotation matrix."

    tbt_new = contract('pi,qj,ijkl,rk,sl->pqrs', U, U_conj, tbt, U, U_conj)

    return tbt_new

def chem_tbt_to_chem_ferm(tbt):

    op = FermionOperator()
    a, b, c, d = np.shape(tbt)

    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    op += FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l), tbt[i, j, k, l])

    return op

def promote_cartan_twobody(op):
    """
    Check if cartan operator, and promote 1 body to 2 body terms

    """
    op_new = FermionOperator()
    op_new += op.constant
    for key, coeff in zip(op.terms.keys(), op.terms.values()):
        
        #check tbt
        if len(key) == 2:
            assert key[0][1] == 1 and key[1][1] == 0, "Operator not in Chem one body"
            assert key[0][0] == key[1][0], "Operator not diagonal"

            op_new += FermionOperator('{}^ {} {}^ {}'.format(key[0][0], key[0][0], key[0][0], key[0][0]), coeff)

        elif len(key) == 4:
            #check chem
            assert key[0][1] == 1 and key[1][1] == 0 and key[2][1] == 1 and key[3][1] == 0, "Operator not in chem two body ordering"
            #check cartan
            assert key[0][0] == key[1][0] and key[2][0] == key[3][0], "Operator not diagonal"

            op_new += FermionOperator('{}^ {} {}^ {}'.format(key[0][0], key[0][0], key[2][0], key[2][0]), coeff)
    
    return op_new

def build_sparse_basis(n_qubits, include_obt=False):
    """
    Builds dictionary of sparse versions of ferm op a_i^ a_j a_k^ a_l

    Uses Jordan-Wigner transform by default
    
    """
    basis_dict = {}

    for i in range(n_qubits):
        for j in range(n_qubits):
            if include_obt:
                basis_dict[(i, j)] = get_sparse_operator(jordan_wigner(FermionOperator('{}^ {}'.format(i, j), 1.0)), n_qubits)
            for k in range(n_qubits):
                for l in range(n_qubits):
                    basis_dict[(i, j, k, l)] = get_sparse_operator(jordan_wigner(FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l), 1.0)), n_qubits)
    
    print("...Built sparse basis for {} qubits".format(n_qubits))
    return basis_dict

def get_sparse_fermop(tbt, basis_dict):
    """
    Construct sparse operator represented by chemist ordered tensor, tbt using premade sparse operator tensor

    """

    n_qubits = len(tbt)
    op = csc_matrix((2**n_qubits, 2**n_qubits))

    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(n_qubits):
                    op += basis_dict[(i, j, k, l)]*tbt[i, j, k, l]
    
    return op

def return_sparse(op, n_qubits):
    """
    Returns sparse operator, does nothing if already sparse

    """
    if issparse(op):
        return op
    else:
        return get_sparse_operator(op, n_qubits)

def return_qubitop(op, n_qubits=None, transform='jw'):
    """
    Returns QubitOperator with mentioned transformation, does nothing if already qubit operator, raises assertion error if sparse and cannot be converted.
    
    """
    if type(op) is QubitOperator:
        return op
    assert not issparse(op), "Operator is sparse, cannot be converted to QubitOperator"

    if type(op) is FermionOperator:
        op = normal_ordered(op) # makes transforms faster

        if transform == 'jw':
            return jordan_wigner(op)
        elif transform == 'bk':
            return bravyi_kitaev(op, n_qubits)
    
    raise AssertionError("Operator type not recognized!")

Epq = lambda p, q: FermionOperator('{}^ {}'.format(p, q), 1.0)
g_pq_real = lambda p, q: Epq(p, q) - Epq(q, p)
g_pq_imag = lambda p, q: 1.j*(Epq(p, q) + Epq(q, p))

def get_U(mat, n_qubits):
    """
    Get the 2^N x 2^N unitary corresponding to the N x N matrix representation, mat of the U(N) algebra

    """

    assert np.shape(mat) == (n_qubits, n_qubits)

    coeff_mat = scipy.linalg.logm(mat)

    op = FermionOperator('', 0)
    for i in range(n_qubits):
        for j in range(n_qubits):
            op += coeff_mat[i, j]*Epq(i, j)
    
    s_op = get_sparse_operator(op, n_qubits)
    return sparse_expm(s_op)

def get_chem_tensors(molecule: MolecularData, verify=False):
    constant = molecule.nuclear_repulsion
    tbt = molecule.two_body_integrals
    tbt_chem = 0.5*tbt.transpose([0, 3, 2, 1]) # symmetric chemist ordering
    obt = molecule.one_body_integrals - np.einsum('piiq->pq',tbt_chem)

    if verify:
        diff_op = normal_ordered(construct_hamiltonian_from_chemtensors(constant, obt, tbt_chem) - get_fermion_operator(molecule.get_molecular_hamiltonian()))
        assert np.sum(np.abs(list(diff_op.terms.values()))) == 0, "Created operator does not match, has error term: {}".format(diff_op)
    return constant, obt, tbt_chem

def construct_ferm_from_chem_spatial_tbt(tbt, spin_ord = 'udud'):
    sh = np.shape(tbt)
    assert len(sh) == 4 and sh[0] == sh[1] and  sh[1] == sh[2] and sh[2] == sh[3], "Invalid two body tensor shape!"

    n_orb = len(tbt)
    if spin_ord == 'udud':
        hamiltonian_fermionic = np.sum([[[[tbt[i, j, k, l]*Fij(i, j)*Fij(k, l) for i in range(n_orb)] for j in range(n_orb)] for k in range(n_orb)] for l in range(n_orb)])
    else:
        hamiltonian_fermionic = np.sum([[[[tbt[i, j, k, l]*Gij(i, j, n_orb)*Gij(k, l, n_orb) for i in range(n_orb)] for j in range(n_orb)] for k in range(n_orb)] for l in range(n_orb)])
    return hamiltonian_fermionic

def construct_ferm_from_chem_spatial_obt(obt, spin_ord = 'udud'):
    sh = np.shape(obt)
    assert len(sh) == 2 and sh[0] == sh[1], "Invalid one body tensor!"

    n_orb = len(obt)
    if spin_ord == 'udud':
        hamiltonian_fermionic = np.sum([[obt[i, j]*Fij(i, j) for i in range(n_orb)] for j in range(n_orb)])
    else:
        hamiltonian_fermionic = np.sum([[obt[i, j]*Gij(i, j, n_orb) for i in range(n_orb)] for j in range(n_orb)])
    return hamiltonian_fermionic


def construct_hamiltonian_from_chemtensors(const, obt, tbt, spin_ord = 'udud'):
    Fij= lambda i, j: FermionOperator('{}^ {}'.format(2*i, 2*j), 1.0) + FermionOperator('{}^ {}'.format(2*i+1, 2*j+1), 1.0)
    Gij= lambda i, j, n_orb: FermionOperator('{}^ {}'.format(i, j), 1.0) + FermionOperator('{}^ {}'.format(n_orb+i, n_orb+j), 1.0)

    n_orb = len(obt)

    return const + construct_ferm_from_chem_spatial_obt(obt, spin_ord=spin_ord) + construct_ferm_from_chem_spatial_tbt(tbt, spin_ord=spin_ord)

# RDM utils


def get_1_2_rdms(state, n_orbs):
    """
    Construct 1, 2 rdms using sparse representation, #TODO currently not scalable

    rdm_1[p,q] = <p^ q>
    rdm_2[p,q,r,s] = <p^q^ sr>

    By default assumes jordan_wigner, modify if otherwise required

    """
    n_qubits = 2*n_orbs
    rdm_1 = np.zeros((n_qubits, n_qubits), complex)
    rdm_2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits), complex)

    for p in range(n_qubits):
        for q in range(n_qubits):

            s_op = get_sparse_operator(FermionOperator('{}^ {}'.format(p, q), 1.0), n_qubits)
            rdm_1[p, q] = expectation(s_op, state)

            for r in range(n_qubits):
                for s in range(n_qubits):
                    s_op = get_sparse_operator(FermionOperator('{}^ {}^ {} {}'.format(p, q, s, r), 1.0), n_qubits)
                    rdm_2[p, q, r, s] = expectation(s_op, state)
    return rdm_1, rdm_2

def get_chem_tbt_expectation_from_rdms(chem_tbt, rdm_1, rdm_2):
    """
    Returns expectation of the two body operator specified by chem_tbt
    
    """

    return np.einsum('pqrs,prqs', chem_tbt, rdm_2) + np.einsum('pqqs,ps', chem_tbt, rdm_1)

def get_obt_expectation_from_rdms(obt, rdm_1):
    return np.einsum('pq, pq', obt, rdm_1)

def rotate_npnq_U(p, q, U):
    """
    Returns chem two body tensor for Un_p n_q U^
    
    """
    U_conj = np.conjugate(U)

    return np.einsum('p,q,r,s->pqrs', U[:,p], U_conj[:,p], U[:,q], U_conj[:,q])

def rotate_npnq_U1U2(p, q, U1, U2):
    """
    Returns chem two body tensor for U1n_pU1^ U2n_q U2^
    
    """
    U1_conj = np.conjugate(U1)
    U2_conj = np.conjugate(U2)

    return np.einsum('p,q,r,s->pqrs', U1[:,p], U1_conj[:,p], U2[:,q], U2_conj[:,q])

def rotate_np_U(p, U):
    """
    Returns chem one body tensor for Un_p U^
    
    """
    U_conj = np.conjugate(U)

    return np.einsum('p,q->pq', U[:,p], U_conj[:,p])
