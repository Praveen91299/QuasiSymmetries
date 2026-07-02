
from openfermion import commutator, get_sparse_operator, expectation, get_ground_state, hermitian_conjugated, QubitOperator, jordan_wigner, FermionOperator
import numpy as np
from scipy.sparse import identity as sparse_id
from copy import deepcopy
from src.op_utils import freeze_qubits
from src.tn import *
from src.clifford_symmetry_optimized import build_symmetry_block_structure_with_packed_qubits, permute_qubits_in_qubit_operator, sparse_qubit_permutation_unitary, sparse_clifford_unitary

def construct_projectors(sym_list: list[QubitOperator]):
    """
    Construct projectors to all subspaces defined by Pauli symmetries sym_list

    """
    if len(sym_list) == 0:
        return [QubitOperator('', coefficient=1.0)]
    
    projectors = []

    sym = sym_list[0]
    projectors_rec = construct_projectors(sym_list=sym_list[1:])
    for proj in projectors_rec:
        projectors.append((0.5 + 0.5 * sym)*proj)
        projectors.append((0.5 - 0.5 * sym)*proj)
    return projectors

def construct_projectors_sparse(sym_list_sparse: list, n_qubits):
    if len(sym_list_sparse) == 0:
        return [sparse_id(1<<n_qubits)]
    
    projectors = []

    sym_sparse = sym_list_sparse[0]
    projectors_rec = construct_projectors_sparse(sym_list_sparse=sym_list_sparse[1:], n_qubits=n_qubits)
    for proj in projectors_rec:
        projectors.append(0.5 * (sparse_id(1<<n_qubits) + sym_sparse)@proj)
        projectors.append(0.5 * (sparse_id(1<<n_qubits) - sym_sparse)@proj)
    return projectors

def get_sector_projectors(list_sym, sectors, n_qubits):
    """
    
    
    """

    n_sym = len(list_sym)
    list_sym_sparse = [get_sparse_operator(sym, n_qubits) for sym in list_sym]
    proj = []

    for sec in sectors:
        sec_proj = sparse_id(1<<n_qubits)
        assert len(sec) == n_sym

        for i, s in enumerate(sec):
            assert s == 1 or s == -1, "Invalid sector label {}".format(s)
            sec_proj = sec_proj @ (0.5*(sparse_id(1<<n_qubits) + s*list_sym_sparse[i]))

        proj.append(sec_proj)
    
    return proj

def find_overlaps(sym_ops, state, n_qubits):
    """
    Find coefficients of state in different symmetry subspaces

    <\psi Pi_s \psi> for all s vectors

    """
    projectors = construct_projectors(sym_ops)
    return [expectation(get_sparse_operator(proj, n_qubits), state) for proj in projectors]

def entropy(probs, tol=1e-5, log_base='2'):
    """
    Entropy (bits) of given probability distribution, truncates to entries >= tol

    """

    probs_trunc = []
    for p in probs:
        if abs(p) >= tol:
            probs_trunc.append(p)

    probs_trunc = np.array(probs_trunc)
    if log_base == '2' or log_base == 2:
        return np.sum(probs_trunc * np.log2(1/probs_trunc))
    elif log_base == 'e' or log_base == np.e:
        return np.sum(probs_trunc * np.log(1/probs_trunc))

def entropy_pauli_sym(projectors_sparse, state, n_qubits):
    return entropy([expectation(proj, state) for proj in projectors_sparse])

def entropy_pauli_syms(sym_ops, state, n_qubits, verbose=False):
    sym_sparse = [get_sparse_operator(sym, n_qubits) for sym in sym_ops]
    projs = construct_projectors_sparse(sym_sparse, n_qubits)
    ent = entropy_pauli_sym(projs, state, n_qubits)
    if verbose: print("Cut entropy: ", ent)
    return ent
    
def l1norm(op: QubitOperator, remove_const=False):
    """
    Returns Pauli L1

    """
    l1= np.sum(np.abs(list(op.terms.values())))
    if not remove_const:
        return l1
    return l1 - np.abs(op.constant)

def universal_grading(sym_ops, H, verbose=False):
    """
    Returns sum of Paulil1 of [S_i, H]

    """
    nc = sum([l1norm(commutator(sym, H)) for sym in sym_ops])
    if verbose: print("Non commutative l1: ", nc)
    return nc

def variance(sym_ops, state, n_qubits, verbose=False):
    v = np.sum([1 - expectation(get_sparse_operator(sym_op, n_qubits), state)**2 for sym_op in sym_ops])
    if verbose: print("Variance: ", v)
    return v

def find_commuting_paulis(H, sym_ops, verbose=False):
    """
    Finds Pauli products in H that commute with all sym_ops
    """
    def is_commuting(op1, op2, tol):
        comm = commutator(op1, op2)
        comm.compress()
        return np.isclose(np.sum(np.abs(list(comm.terms.values()))), 0, rtol=tol)
    
    HQ = deepcopy(H)
    c = HQ.constant
    HQ = HQ - c
    HQ.compress()
    n_total_pauli =  len(H.terms.keys())

    commuting_terms = []
    for term, coeff in H.terms.items():
        Pauli =  QubitOperator(term, coeff)

        if all([is_commuting(sym_op, Pauli, 1e-5) for sym_op in sym_ops]):
            commuting_terms.append(Pauli)
    
    if verbose: print("{}/{} Terms in H found to commute with all symmetries.".format(len(commuting_terms), n_total_pauli))

    return commuting_terms

def find_commuting_terms(H, sym_ops, verbose=False):
    """
    Finds Fermion strings in H that commute with all sym_ops
    """
    def is_commuting(op1, op2, tol):
        comm = commutator(op1, op2)
        comm.compress()
        return np.isclose(np.sum(np.abs(list(comm.terms.values()))), 0, rtol=tol)
    
    HQ = deepcopy(H)
    c = HQ.constant
    HQ = HQ - c
    HQ.compress()
    n_total =  len(H.terms.keys())

    commuting_terms = []
    for term, coeff in HQ.terms.items():
        t =  FermionOperator(term, coeff)

        if all([is_commuting(sym_op, jordan_wigner(t), 1e-5) for sym_op in sym_ops]):
            commuting_terms.append(t)
    
    if verbose: print("{}/{} Terms in H found to commuting with all symmetries.".format(len(commuting_terms), n_total))

    return commuting_terms

def comm_sq_exp_fast(sym_ops, H, state, n_qubits, verbose=False):
    """
    Compute sum_k <state| ( i[H, S_k] )^2 |state> efficiently.

    Parameters
    ----------
    sym_ops : list[QubitOperator]
        Symmetry operators (Pauli products).
    H : sparse operator
        Hamiltonian.
    state : np.ndarray
        State vector.
    n_qubits : int

    Returns
    -------
    float or complex
    """
    
    psi = np.asarray(state)

    # Reused for every symmetry operator
    Hpsi = H @ psi

    total = 0.0 + 0.0j
    for sym in sym_ops:
        S = get_sparse_operator(sym, n_qubits).tocsr()

        Spsi = S @ psi
        delta = 1j * ((H @ Spsi) - (S @ Hpsi))   # delta = i[H,S]|psi>

        # <psi| (i[H,S])^2 |psi> = || delta ||^2
        total += np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp

def comm_sq_exp_pauli_actions(pauli_actions, H, state, verbose=False, weights=None, Hpsi=None):
    """
    comm_sq_exp using direct Pauli-product permutation/phase actions.

    This avoids sparse S_k matvecs.  It is only valid when each symmetry is a
    single Pauli product.
    """
    psi = np.asarray(state).reshape(-1)
    H = H.tocsr()
    if Hpsi is None:
        Hpsi = H @ psi
    else:
        Hpsi = np.asarray(Hpsi).reshape(-1)

    if weights is None:
        weights = np.ones(len(pauli_actions))

    total = 0.0 + 0.0j
    for weight, action in zip(weights, pauli_actions):
        if weight == 0:
            continue
        Spsi = action.apply(psi)
        SHpsi = action.apply(Hpsi)
        delta = 1j * ((H @ Spsi) - SHpsi)
        total += weight * np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp

def prepare_sparse_symmetries(sym_ops, n_qubits):
    """
    Convert symmetry QubitOperators to CSR matrices once for repeated metrics.
    """
    return [get_sparse_operator(sym, n_qubits).tocsr() for sym in sym_ops]

def comm_sq_exp_sparse_syms(sym_ops_sparse, H, state, verbose=False, weights=None):
    """
    Repeated-evaluation version of comm_sq_exp_fast.

    sym_ops_sparse should be the output of prepare_sparse_symmetries.  Avoiding
    get_sparse_operator inside every objective call matters during orbital
    optimization.
    """
    psi = np.asarray(state).reshape(-1)
    H = H.tocsr()
    Hpsi = H @ psi

    if weights is None:
        weights = np.ones(len(sym_ops_sparse))

    total = 0.0 + 0.0j
    for weight, S in zip(weights, sym_ops_sparse):
        if weight == 0:
            continue

        Spsi = S @ psi
        delta = 1j * ((H @ Spsi) - (S @ Hpsi))
        total += weight * np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp


def get_entropies_at_cuts(state, n_qubits, log_base='2'):
    """
    Get bi-partite entanglement across all partitions of qubits

    state: np.array - state
    log_base: str - '2' or 'e'
    """
    entropies = []
    for k in range(1, n_qubits):
        u, d, v = np.linalg.svd(np.reshape(state, (1<<k, 1<<(n_qubits-k))))

        entropies.append(entropy(np.abs(d)**2, log_base=log_base))
    return entropies

def permute_sym_to_start(HQ, symmetries, n_qubits, verbose=False, return_clifford_perm=False):
    """
    Move qubits to the start
    
    """
    res = build_symmetry_block_structure_with_packed_qubits(
        hamiltonian=HQ,
        symmetries=symmetries,
        n_qubits=n_qubits,
    )

    #permute symmetries to the start
    H_trans = res.transformed_hamiltonian
    sym_mapped_qubits = res.original_mapped_qubits

    #syms to start + rest in order
    if verbose: print("Symmetries rotated to Z on qubits: ", sym_mapped_qubits)
    n_sym = len(sym_mapped_qubits) #locations of symmetry qubits - should go to the beginning
    perm = []
    ns=0
    nns =0
    for i in range(n_qubits): #qubit count
        if i in sym_mapped_qubits: 
            assert ns < n_sym, "Too many symmetry indices!"
            perm.append(sym_mapped_qubits.index(i))
            ns +=1
        else:
            perm.append(n_sym + nns)
            nns += 1

    if verbose:
        print("Qubits permuted as:")
        for i, p in enumerate(perm):
            print(i, "->", p)

    H_perm = permute_qubits_in_qubit_operator(H_trans, perm)
    if return_clifford_perm:
        return H_perm, res, perm
    else:
        return H_perm

def get_ent(symmetries, HQ, n_qubits, verbose=False, return_state=False, return_sparse_clifford=False, log_base='2'):
    """
    Get bi-partite entanglement across all partitions after diagonalizing symmetries and localizing them to qubits 0, 1, 2, ... in order

    """
    if len(symmetries) > 0:
        H_perm = permute_sym_to_start(HQ, symmetries, n_qubits, verbose=verbose)
    else:
        if verbose: print("No symmetries passed, returning original bond entanglements.")
        H_perm = HQ
    e_p, gs = get_ground_state(get_sparse_operator(H_perm, n_qubits))

    ents = get_entropies_at_cuts(gs, n_qubits, log_base=log_base)
    if verbose:
        print("Entropy of cuts (bits):")
        for i, e in enumerate(ents):
            print("{} | {} : {}".format(i+1, i+2, e))
    
    if return_state:
        return ents, H_perm, gs
    else:
        return ents, H_perm
    
def int_to_binary_list(x: int, n: int, MSB_first=True) -> list[int]:
    """
    Convert a nonnegative integer x to a length-n list of binary digits.

    The most significant bit comes first.

    Example:
        int_to_binary_list(6, 4) -> [0, 1, 1, 0]
    """
    if x < 0:
        raise ValueError("x must be nonnegative")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if x >= (1 << n):
        raise ValueError(f"x={x} cannot be represented with {n} bits")

    b = [(x >> i) & 1 for i in reversed(range(n))]

    if MSB_first:
        return b
    else:
        return list(reversed(b))

def get_single_sector_energies(HQ, list_sym, n_qubits, verbose=False):
    """
    Find ground state in symmetry sectors

    Rotates Hamiltonian and then freezes qubits, following with it solves for ground state energy

    """

    n_sym = len(list_sym)
    n_qubits_red = n_qubits - n_sym
    #all combinations

    H_perm = permute_sym_to_start(HQ, list_sym, n_qubits,False)
    frozen_qubits = list(range(n_sym))

    gs_e_list = []
    for i in range(1<<n_sym):
        sec_label = int_to_binary_list(i, n_sym, MSB_first=False)
        sec_dict = {s: v for s, v in zip(frozen_qubits, sec_label)}
        H_red_sec = freeze_qubits(H_perm, sec_dict)

        gs_e, gs = get_ground_state(get_sparse_operator(H_red_sec, n_qubits_red))
        gs_e_list.append(gs_e)
    
    if verbose: print("Minimum single sector energy: ", np.min(gs_e_list))
    return gs_e_list

def get_bipartite_mps(HQ, n_qubits, target_energy=None, bd=100, n_sweeps=100, tol=1.6e-3):
    """
    Uses pyblock2 to solve and calculate the bipartite entanglement

    """
    np.random.seed(0)
    
    mpo, driver = QO_to_block2_MPO(HQ, n_qubits)
    ket = driver.get_random_mps(tag="KET", bond_dim=bd, nroots=1)

    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=n_sweeps,
        bond_dims=None,
        noises=[1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6] + [0.0]*(n_sweeps - 6),
        thrds=[1e-10] * n_sweeps,
        dav_max_iter=50,
        iprint=0
    )

    if target_energy is not None:

        if np.abs(energy - target_energy) < tol:
            print("Bipartite entanglement: Warning dmrg not converged to reference energy...")

    return driver.get_bipartite_entanglement(ket), ket

def get_permuted_bipartite_entanglement(
    symmetries,
    HQ,
    n_qubits,
    fci_energy=None,
    fci_gs=None,
    verbose=False,
    return_state=False,
    return_U=False,
    log_base='e',
    use_dmrg=False,
    return_clifford_info=False,
):
    """
    Get bi-partite entanglement across all partitions after diagonalizing symmetries and localizing them to qubits 0, 1, 2, ... in order
    *Modified version of get_ent with dmrg calculations for speed.*

    """
    #permute
    if len(symmetries) > 0:
        H_perm, clifford, perm = permute_sym_to_start(HQ, symmetries, n_qubits, verbose=verbose, return_clifford_perm=True)
    else:
        if verbose: print("No symmetries passed, returning original bond entanglements.")
        H_perm = HQ
    
    #construct U
    if verbose: print("Constructing unitary from factors and permutations...")
    Ucliff_sparse = sparse_clifford_unitary(clifford.clifford_result, n_qubits)
    Uperm_sparse = sparse_qubit_permutation_unitary(perm, True)
    U = Uperm_sparse @ Ucliff_sparse
    clifford_info = {
        "factor_descriptions": list(clifford.clifford_result.factor_descriptions),
        "parsed_gates": list(clifford.clifford_result.parsed_gates),
        "permutation": list(perm),
    }

    #solve
    if use_dmrg:
        ents, gs = get_bipartite_mps(H_perm, n_qubits, target_energy=fci_energy)

        if log_base == '2':
            ents = ents / np.log(2)
    else:
        if fci_gs is not None:
            #transform state directly
            gs = U @ fci_gs
        else:
            e_p, gs = get_ground_state(get_sparse_operator(H_perm, n_qubits))
            if fci_energy is not None: assert np.isclose(fci_energy, e_p, atol=1e-5), "Permuted Hamiltonian ground state differs from fci by {}".format(fci_energy - e_p)
        ents = get_entropies_at_cuts(gs, n_qubits, log_base=log_base)
    
    if verbose:
        print("Entropy of cuts (log base = {}):".format(log_base))
        for i, e in enumerate(ents):
            print("{} | {} : {}".format(i+1, i+2, e))

    # Construct the requested return tuple without changing existing callers.
    if return_U:
        if return_state:
            result = (ents, H_perm, U, gs)
        else:
            result = (ents, H_perm, U)
    else:
        if return_state:
            result = (ents, H_perm, gs)
        else:
            result = (ents, H_perm)

    if return_clifford_info:
        return (*result, clifford_info)
    return result
