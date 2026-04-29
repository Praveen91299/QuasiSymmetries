
from openfermion import commutator, get_sparse_operator, expectation, get_ground_state, hermitian_conjugated, QubitOperator, jordan_wigner, FermionOperator
import numpy as np
from scipy.sparse import identity as sparse_id
from copy import deepcopy

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

def find_overlaps(sym_ops, state, n_qubits):
    """
    Find coefficients of state in different symmetry subspaces

    <\psi Pi_s \psi> for all s vectors

    """
    projectors = construct_projectors(sym_ops)
    return [expectation(get_sparse_operator(proj, n_qubits), state) for proj in projectors]

def entropy(probs, tol=1e-5):
    """
    Entropy (bits) of given probability distribution, truncates to entries >= tol

    """

    probs_trunc = []
    for p in probs:
        if abs(p) >= tol:
            probs_trunc.append(p)

    probs_trunc = np.array(probs_trunc)
    return np.sum(probs_trunc * np.log2(1/probs_trunc))

def entropy_pauli_sym(projectors_sparse, state, n_qubits):
    return entropy([expectation(proj, state) for proj in projectors_sparse])

def entropy_pauli_syms(sym_ops, state, n_qubits):
    sym_sparse = [get_sparse_operator(sym, n_qubits) for sym in sym_ops]
    projs = construct_projectors_sparse(sym_sparse, n_qubits)

    return entropy_pauli_sym(projs, state, n_qubits)
    
def l1norm(op: QubitOperator):
    """
    Returns Pauli L1

    """
    return np.sum(np.abs(list(op.terms.values())))

def universal_grading(sym_ops, H):
    """
    Returns sum of Paulil1 of [S_i, H]

    """
    return sum([l1norm(commutator(sym, H)) for sym in sym_ops])

def variance(sym_ops, state, n_qubits):
    return np.sum([1 - expectation(get_sparse_operator(sym_op, n_qubits), state)**2 for sym_op in sym_ops])

def find_commuting_paulis(H, sym_ops):
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
    
    print("{}/{} Terms in H found to commute with all symmetries.".format(len(commuting_terms), n_total_pauli))

    return commuting_terms

def find_commuting_terms(H, sym_ops):
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
    
    print("{}/{} Terms in H found to commuting with all symmetries.".format(len(commuting_terms), n_total))

    return commuting_terms

def comm_sq_exp_fast(sym_ops, H, state, n_qubits):
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

    return np.real_if_close(total)

from src.clifford import *
def get_entropies_at_cuts(state, n_qubits):
    entropies = []
    for k in range(1, n_qubits):
        u, d, v = np.linalg.svd(np.reshape(state, (1<<k, 1<<(n_qubits-k))))

        entropies.append(entropy(np.abs(d)**2))
    return entropies

def permute_sym_to_start(HQ, symmetries, n_qubits, verbose=False):
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
    return H_perm

def get_ent(symmetries, HQ, n_qubits, verbose=False, return_state=False):
    """
    Get bi-partite entanglement across all partitions after diagonalizing symmetries and localizing them to qubits 0, 1, 2, ... in order

    """
    H_perm = permute_sym_to_start(HQ, symmetries, n_qubits, verbose=verbose)
    e_p, gs = get_ground_state(get_sparse_operator(H_perm, n_qubits))

    ents = get_entropies_at_cuts(gs, n_qubits)
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

def get_BO_energies(HQ, list_sym, n_qubits):
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
    
    return gs_e_list