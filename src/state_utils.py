# HF, CISD, FCI stuff

import numpy as np
from scipy.sparse import csr_matrix
from pyscf.fci import cistring
import openfermion as of
from openfermion import MolecularData
from scipy.linalg import eigh
import scipy as sp
import math
from pyscf import fci

def to_str(occ_list):
    st = ''
    for occ in occ_list:
        st += str(occ)
    
    return st

def get_hf_occ(n_electrons, n_orbitals, spin_ord = 'udud', remove_qubit_loc = [], as_str=False):
    '''
    List slater determinant of HF
    '''
    hf = [1]*n_electrons + [0]*(2*n_orbitals - n_electrons)
    if spin_ord == 'uudd':
        hf = hf[::2] + hf[1::2]
    
    hf_f = []
    for i, a in enumerate(hf):
        if i not in remove_qubit_loc:
            hf_f.append(a)
    
    if as_str:
        return to_str(occ_list=hf_f)
    else:
        return hf_f

def get_hf_wfn(occ):
    wfn = [1.0]
    for i in occ:
        if i == 1:
            wfn = np.kron(wfn, [0, 1])
        else:
            wfn = np.kron(wfn, [1, 0])
    return wfn

### CISD
### Choi's code
#
# def get_gs(mol, op):
#     values, vectors = eigh(op.toarray())

#     order = np.argsort(values)
#     values = values[order]
#     vectors = vectors[:, order]
#     print(values)
#     if mol == 'ch2':
#         eigenvalue = values[3]
#         eigenstate = vectors[:, 3]
#     elif mol == 'h2ost2':
#         eigenvalue = values[5]
#         eigenstate = vectors[:, 5]
#     else:
#         eigenvalue = values[0]
#         eigenstate = vectors[:, 0]

#     return eigenvalue, eigenstate.T

def get_gs(op):
    """
    Returns gs energy and state of a given matrix

    """
    values, vectors = eigh(op.toarray())

    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    #print(values)
    
    eigenvalue = values[0]
    eigenstate = vectors[:, 0]

    return eigenvalue, eigenstate.T


def partial_order(x, y):
    """
    As described in arXiv:quant-ph/0003137 pg.10, computes the if x <= y where <= is a partial order and x and y are binary strings (but inputted as integers).
    Args:
        x, y (int): Integers that will be converted to binary to then check x <= y.

    Returns:
        partial_order(bool): Whether x <= y

    """
    if x > y:
        return False

    else:
        x_b, y_b = format(x, 'b'), format(y, 'b')

        if len(x_b) != len(y_b):
            while len(x_b) != len(y_b):
                x_b = '0' + x_b

        length = len(x_b)

        partial_order = False
        for l0 in range(length):
            if x_b[0:l0] == y_b[0:l0] and y_b[l0:length] == (length - l0)*'1':
                partial_order = True
                break

        return partial_order

def get_bk_tf_matrix(n_qubits):
    """
    Implementation from arXiv:quant-ph/0003137 and https://doi.org/10.1021/acs.jctc.8b00450. Given some reference occupation no's in the fermionic space, find the corresponding BK basis state in the qubit space.
    Args:
        n_qubits (int): No. of qubits
    Returns:
        tf_mat (np.array): Transformation matrix that converts fermionic occupation numbers to BK transformed basis vectors.
    """

    tf_mat = np.zeros((n_qubits, n_qubits))

    for i in range(n_qubits):
        if np.mod(i, 2) == 0:
            tf_mat[i, i] = 1
        elif np.mod(math.log(i+1, 2), 1) == 0:
            for j in range(i+1):
                tf_mat[i, j] = 1
        else:
            for j in range(n_qubits):
                if partial_order(j, i) == True:
                    tf_mat[i, j] = 1

    return tf_mat

def get_bk_basis_states(occ_no, n_qubits):
    """
    Implementation from arXiv:quant-ph/0003137 and https://doi.org/10.1021/acs.jctc.8b00450. Given some reference occupation no's in the fermionic space, find the corresponding BK basis state in the qubit space.
    Args:
        occ_no_list (List[str]): List of occupation number vectors. Occ no. vectors ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        basis_state (np.array): Basis vector in (BK transformed) qubit space corresponding to occ_no_state.
    """

    tf_mat = get_bk_tf_matrix(n_qubits)

    occ_no_vec = np.array(list(occ_no), dtype = int)
    qubit_state = np.mod(np.matmul(tf_mat, occ_no_vec), 2)

    return qubit_state

def get_jw_basis_states(occ_no_list, n_qubits):
    """
    Implementation from arXiv:quant-ph/0003137 and https://doi.org/10.1021/acs.jctc.8b00450. Given some reference occupation no's in the fermionic space, find the corresponding BK basis state in the qubit space.
    Args:
        occ_no_list (List[str]): List of occupation number vectors. Occ no. vectors ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        basis_state (np.array): Basis vector in (JW transformed) qubit space corresponding to occ_no_state.
    """

    jw_list = []
    for occ_no in occ_no_list:
        qubit_state = np.array(list(occ_no), dtype = int)
        jw_list.append(qubit_state)

    return jw_list


def find_index(basis_state):
    """
    Given some qubit/fermionic basis state, find the index of the a wavefunction that corresponds to that array.
    Args:
        basis_state (str or list/np.array): Occupation number vector/ Qubit basis state. If str, ordered from left to right going from 0 -> n-1 in terms of orbitals/qubits.
    Returns:
        index (int): Index of the basis in total Qubit space.
    """
    index = 0
    n_qubits = len(basis_state)
    for j in range(n_qubits):
        index += int(basis_state[j])*2**(n_qubits - j - 1)

    return index

def get_reference_state(occ_no_state, tf = 'bk', gs_format = 'dm'):
    """
    Given some occupation numebr vector, make the density matrix that corresponds to that state.
    Args:
        occ_no_state (str or list/np.array): Occupation number vector. If str, ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        dm (sp.sparse.coo_matrix): Density matrix (sparse for efficiency) of the reference state in qubit space.
        or wfs (np.array): wavefunction of the CISD state in qubit space.
    """
    n_qubits = len(occ_no_state)
    bk_basis_state = get_bk_basis_states(occ_no_state, n_qubits)
    index = find_index(bk_basis_state[0])

    if gs_format == 'wfs':
        wfs = np.zeros(2**n_qubits)
        wfs[index] = 1

        return wfs


    if gs_format == 'dm':

        dm = sp.sparse.coo_matrix(([1], ([index], [index])), shape = (2**n_qubits, 2**n_qubits))

        return dm

def get_occ_no(mol, n_qubits):
    """
    Given some molecule, find the reference occupation number state.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
    Returns:
        occ_no (str): Occupation no. vector.
    """
    n_electrons = {'h2': 2, 'lih': 4, 'beh2': 6, 'h2o': 10, 'nh3': 10, 'n2': 14, 'hf':10, 'ch4':10, 'co':14, 'h4':4, 'ch2':8, 'heh':2, 'h6':6, 'nh':8, 'h3':2, 'h4sq':4, 'h2ost':10, 'beh2st':6, 'h2ost2':10, 'beh2st2':6}
    occ_no = '1'*n_electrons[mol] + '0'*(n_qubits - n_electrons[mol])

    return occ_no

def get_jw_cisd_basis_states_wrap(ref_occ_nos, n_qubits):
    """
    Given some occupation number, find the all other occupation numbers that are achieved by single and double excitations.
    Args:
        ref_occ_nos (str): Reference (likely HF) occupation number ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        cisd_basis_states (List[str]): List of all occupation number achieved by singles and doubles from reference occupation number.
    """

    indices = [find_index(get_jw_basis_states(ref_occ_nos, n_qubits))]
    for occidx, occ_orbitals in enumerate(ref_occ_nos):
        if occ_orbitals == '1':
            annihilated_state = list(ref_occ_nos)
            annihilated_state[occidx] = '0'

            #Singles
            for virtidx, virtual_orbs in enumerate(ref_occ_nos):
                if virtual_orbs == '0':
                    new_state = annihilated_state[:]
                    new_state[virtidx] = '1'
                    indices.append(find_index(get_jw_basis_states(''.join(new_state), n_qubits)))

                    #Doubles
                    for occ2idx in range(occidx +1, n_qubits):
                        if ref_occ_nos[occ2idx] == '1':
                            annihilated_state_double = new_state[:]
                            annihilated_state_double[occ2idx] = '0'

                            for virt2idx in range(virtidx +1, n_qubits):
                                if ref_occ_nos[virt2idx] == '0':
                                    new_state_double = annihilated_state_double[:]
                                    new_state_double[virt2idx] = '1'
                                    indices.append(find_index(get_jw_basis_states(''.join(new_state_double), n_qubits)))
    return indices

def get_bk_cisd_basis_states_wrap(ref_occ_nos, n_qubits):
    """
    Given some occupation number, find the all other occupation numbers that are achieved by single and double excitations.
    Args:
        ref_occ_nos (str): Reference (likely HF) occupation number ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        cisd_basis_states (List[str]): List of all occupation number achieved by singles and doubles from reference occupation number.
    """

    indices = [find_index(get_bk_basis_states(ref_occ_nos, n_qubits))]
    for occidx, occ_orbitals in enumerate(ref_occ_nos):
        if occ_orbitals == '1':
            annihilated_state = list(ref_occ_nos)
            annihilated_state[occidx] = '0'

            #Singles
            for virtidx, virtual_orbs in enumerate(ref_occ_nos):
                if virtual_orbs == '0':
                    new_state = annihilated_state[:]
                    new_state[virtidx] = '1'
                    indices.append(find_index(get_bk_basis_states(''.join(new_state), n_qubits)))

                    #Doubles
                    for occ2idx in range(occidx +1, n_qubits):
                        if ref_occ_nos[occ2idx] == '1':
                            annihilated_state_double = new_state[:]
                            annihilated_state_double[occ2idx] = '0'

                            for virt2idx in range(virtidx +1, n_qubits):
                                if ref_occ_nos[virt2idx] == '0':
                                    new_state_double = annihilated_state_double[:]
                                    new_state_double[virt2idx] = '1'
                                    indices.append(find_index(get_bk_basis_states(''.join(new_state_double), n_qubits)))
    return indices

def get_bk_cisd_basis_states(mol, n_qubits):
    """
    Given some molecule, find the all BK basis vectors that correspond to occupation numbers that are achieved by single and double excitations.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
        n_qubits (int): No. of qubits
    Returns:
        bk_basis_states (List[array]): List of all BK basis states corresponding to occupation numbers achieved by singles and doubles from reference occupation number.
    """

    ref_occ_nos = get_occ_no(mol, n_qubits)
    indices = get_bk_cisd_basis_states_wrap(ref_occ_nos, n_qubits)
    return indices


def get_jw_cisd_basis_states(mol, n_qubits):
    """
    Given some molecule, find the all BK basis vectors that correspond to occupation numbers that are achieved by single and double excitations.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
        n_qubits (int): No. of qubits
    Returns:
        jw_basis_states (List[array]): List of all JW basis states corresponding to occupation numbers achieved by singles and doubles from reference occupation number.
    """

    ref_occ_nos = get_occ_no(mol, n_qubits)
    indices = get_jw_cisd_basis_states_wrap(ref_occ_nos, n_qubits)
    return indices

def create_hamiltonian_in_subspace(indices, Hq, n_qubits):
    """
    Given some basis states, create the Hamiltonian within the span of those basis states.
    Args:
        qubit_basis_states(List[array] or List[str]): List of basis vectors to create hamiltonian within
        Hq (QubitOperator): Qubit hamiltonian
        n_qubits (int): Number of qubits.
    Returns:
        H_mat_sub (sp.sparse.coo_matrix): Hamiltonian matrix defined in subspace.
        indices (List[int]): Gives the index in the 2**n dimensional space of the ith qubit_basis_state.
    """

    subspace_dim = len(indices)

    row_idx = []
    col_idx = []
    H_mat_elements = []

    #print(len(Hq.terms))
    elements_sum = np.zeros((len(indices),len(indices)))
    op_sum = of.QubitOperator.zero()
    for prog, op in enumerate(Hq):
        op_sum += op
        if (prog + 1)%350 == 0 or prog == len(Hq.terms) - 1:
            #print(prog)
            opspar = of.get_sparse_operator(op_sum, n_qubits)
            op_sum = of.QubitOperator.zero()
            for iidx, iindx in enumerate(indices):
                for jidx, jindx in enumerate(indices):
                    elements_sum[iidx, jidx] += opspar[iindx, jindx]
                 
    for iidx, iindx in enumerate(indices):
        for jidx, jindx in enumerate(indices):
            row_idx.append(iidx)
            col_idx.append(jidx)
            H_mat_elements.append(elements_sum[iidx, jidx])

    H_mat_sub = sp.sparse.coo_matrix((H_mat_elements, (row_idx, col_idx)), shape = (subspace_dim, subspace_dim))

    return H_mat_sub

def get_cisd_gs(occ_str, Hq, n_qubits, gs_format = 'dm', reduce_determinants = False, tf = 'bk'):
    """
    Finds the CISD wavefunction/density matrix in qubit space.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
        Hq (QubitOperator): Qubit hamiltonian
        n_qubits (int): No. of qubits
    Returns:
        dm (sp.sparse.coo_matrix): Density matrix (sparse for efficiency) of the CISD state in qubit space.
        or wfs (np.array): wavefunction of the CISD state in qubit space.
    """


    if tf == 'bk':
        indices = get_bk_cisd_basis_states_wrap(occ_str, n_qubits)
    elif tf == 'jw':
        indices = get_jw_cisd_basis_states_wrap(occ_str, n_qubits)
    else:
        return('Transformation Not Valid.')
    H_mat_cisd = create_hamiltonian_in_subspace(indices, Hq, n_qubits)

    #energy, gs = get_gs(mol, H_mat_cisd)
    energy, gs = get_gs(H_mat_cisd)

    if reduce_determinants == True:
        while np.linalg.norm(gs) > 0.99:
            min_index = np.argmin(np.abs(gs))
            gs[min_index] = 0

        gs = gs/np.linalg.norm(gs) #Renormalisation


    if gs_format == 'wfs': # TODO make to sparse or use SDState

        wfs = np.zeros(2**n_qubits)

        for iidx, iindx in enumerate(indices):
            wfs[iindx] = gs[iidx]

        wfs = wfs/np.linalg.norm(wfs)

        return energy, wfs

    if gs_format == 'dm':

        row_idx = []
        col_idx = []
        dm_vals = []

        for iidx, iindx in enumerate(indices):
            for jidx, jindx in enumerate(indices):
                row_idx.append(iindx)
                col_idx.append(jindx)
                dm_vals.append(gs[iidx]*np.conj(gs[jidx]))

        dm = sp.sparse.coo_matrix((dm_vals, (row_idx, col_idx)), shape = (2**n_qubits, 2**n_qubits))
        dm = dm / dm.diagonal().sum()

        return energy, dm

### FCI
# AI code, seems to work
def get_fci_state_openfermion(molecule: MolecularData, threshold=1e-12):
    """
    Return (energy, state, info) where state is a sparse column vector
    compatible with OpenFermion/get_ground_state conventions.

    Parameters
    ----------
    molecule
        MolecularData / PyscfMolecularData object already processed by run_pyscf.
    threshold : float
        Drop coefficients with |c| <= threshold.

    Returns
    -------
    energy : float
    state : scipy.sparse.csr_matrix
        Shape (2**n_qubits, 1)
    info : dict
    """
    if not hasattr(molecule, "_pyscf_data") or molecule._pyscf_data is None:
        raise ValueError("molecule._pyscf_data missing; use a molecule returned by run_pyscf.")

    pyscf_data = molecule._pyscf_data
    pyscf_mol = pyscf_data["mol"]
    pyscf_scf = pyscf_data["scf"]

    # Reuse stored FCI solver if present, else build one.
    solver = pyscf_data.get("fci", None)
    if solver is None:
        solver = fci.FCI(pyscf_mol, pyscf_scf.mo_coeff)
        solver.verbose = 0

    energy, ci = solver.kernel()
    ci = np.asarray(ci)

    norb = int(pyscf_scf.mo_coeff.shape[1])   # spatial orbitals
    n_qubits = 2 * norb
    dim = 1 << n_qubits

    nelec = int(pyscf_mol.nelectron)
    spin = int(pyscf_mol.spin)   # = n_alpha - n_beta
    n_alpha = (nelec + spin) // 2
    n_beta = nelec - n_alpha

    alpha_strings = np.asarray(cistring.make_strings(range(norb), n_alpha), dtype=np.int64)
    beta_strings = np.asarray(cistring.make_strings(range(norb), n_beta), dtype=np.int64)

    expected_shape = (len(alpha_strings), len(beta_strings))
    if ci.shape != expected_shape:
        raise ValueError(
            f"Unexpected CI tensor shape {ci.shape}; expected {expected_shape}."
        )

    def occupied_orbitals(det, norb):
        return [p for p in range(norb) if (det >> p) & 1]

    def pyscf_det_to_openfermion_index_and_phase(alpha_det, beta_det, norb):
        """
        PySCF CI basis is organized by separate alpha/beta strings.
        We embed into OpenFermion spin-orbital order:
            [a0, b0, a1, b1, ..., a_{norb-1}, b_{norb-1}]
        and build the basis index using qubit 0 as the leftmost tensor factor.

        Returns
        -------
        basis_index : int
        phase : +1 or -1
        """
        occ_alpha = occupied_orbitals(alpha_det, norb)
        occ_beta = occupied_orbitals(beta_det, norb)

        # Fermionic sign from reordering:
        # starting from all-alpha then all-beta ordering
        # into interleaved spin-orbital ordering.
        inversions = 0
        for p in occ_alpha:
            for q in occ_beta:
                if q < p:
                    inversions += 1
        phase = -1.0 if (inversions % 2) else 1.0

        # Build OpenFermion computational-basis index.
        # qubit 0 is the leftmost tensor factor, so it contributes to the
        # most-significant bit of the basis index.
        idx = 0
        for p in occ_alpha:
            q = 2 * p
            idx |= (1 << (n_qubits - 1 - q))
        for p in occ_beta:
            q = 2 * p + 1
            idx |= (1 << (n_qubits - 1 - q))

        return idx, phase

    rows = []
    data = []

    for ia, alpha_det in enumerate(alpha_strings):
        for ib, beta_det in enumerate(beta_strings):
            coeff = ci[ia, ib]
            if abs(coeff) > threshold:
                idx, phase = pyscf_det_to_openfermion_index_and_phase(
                    int(alpha_det), int(beta_det), norb
                )
                rows.append(idx)
                data.append(phase * coeff)

    if rows:
        cols = np.zeros(len(rows), dtype=np.int64)
        state = csr_matrix(
            (np.asarray(data, dtype=np.complex128), (np.asarray(rows), cols)),
            shape=(dim, 1),
            dtype=np.complex128,
        )
    else:
        state = csr_matrix((dim, 1), dtype=np.complex128)

    info = {
        "norb": norb,
        "n_qubits": n_qubits,
        "nelec": nelec,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "nnz": state.nnz,
    }
    return float(energy), state, info