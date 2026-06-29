import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from openfermion import FermionOperator, hermitian_conjugated, normal_ordered
from openfermion.transforms import get_interaction_operator, jordan_wigner


from src.tn import find_dmrg_conv_bd_quimb

import pickle
import quimb.tensor as qtn
import numpy as np
from openfermion import count_qubits, jordan_wigner, MolecularData, get_sparse_operator
from src.state_utils import get_hf_wfn, get_hf_occ
from src.metrics import get_permuted_bipartite_entanglement, comm_sq_exp_fast, get_entropies_at_cuts
from src.sym import get_seniority_symmetries, hct_mod
from src.bliss import lp_bliss_paper_real_pauli_1norm
from benchmark_all import benchmark_syms, BenchmarkData
import pandas as pd
from src.fiedler import fiedler_order_from_state, reorder_statevector_axes
from src.bs.beam import find_commuting_symmetry_generators

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from src.tn import MPO_from_QubitOperator


def _pauli_dense_from_qubit_term(term, coeff, n_qubits):
    """
    Build a dense matrix for one OpenFermion QubitOperator term.

    term example:
        ((0, 'Z'), (3, 'X'))

    This returns:
        where, dense_operator_on_contiguous_support

    We use contiguous support because JW strings create long Z chains.
    """
    paulis = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    if len(term) == 0:
        return (), complex(coeff) * np.ones((1, 1), dtype=complex)

    sites = [i for i, _ in term]
    lo, hi = min(sites), max(sites)
    where = tuple(range(lo, hi + 1))

    op_map = {i: p for i, p in term}
    mat = np.array([[1.0 + 0.0j]])

    for site in where:
        mat = np.kron(mat, paulis[op_map.get(site, "I")])

    return where, complex(coeff) * mat
def local_expectation_compat(mps, mat, where):
    """
    Compute <mat_where> for a quimb MPS using the newer
    compute_local_expectation API.

    mat acts on the contiguous sites in `where`.
    """
    terms = {tuple(where): mat}

    try:
        return mps.compute_local_expectation(
            terms,
            normalized=True,
            return_all=False,
            method="canonical",
        )
    except TypeError:
        return mps.compute_local_expectation(
            terms,
            return_all=False,
            method="canonical",
        )


def quimb_expect_fermion_operator(
    mps,
    fop,
    n_qubits,
    cutoff=1e-12,
):
    """
    Compute <mps| fop |mps> by JW-transforming each FermionOperator term.
    """
    qop = jordan_wigner(normal_ordered(fop))
    total = 0.0 + 0.0j

    for qterm, coeff in qop.terms.items():
        if abs(coeff) < cutoff:
            continue

        where, mat = _pauli_dense_from_qubit_term(qterm, coeff, n_qubits)

        if where == ():
            total += mat[0, 0]
            continue

        total += local_expectation_compat(mps, mat, where)

    return total


def compute_rdms_from_quimb_mps(
    mps,
    n_orbs,
    cutoff=1e-12,
    verbose=False,
):
    gamma1 = np.zeros((n_orbs, n_orbs), dtype=complex)
    gamma2 = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs), dtype=complex)

    for p in range(n_orbs):
        for q in range(n_orbs):
            op = FermionOperator(((p, 1), (q, 0)), 1.0)
            gamma1[p, q] = quimb_expect_fermion_operator(
                mps,
                op,
                n_orbs,
                cutoff=cutoff,
            )

    for p in range(n_orbs):
        if verbose:
            print(f"2-RDM p={p + 1}/{n_orbs}")

        for q in range(n_orbs):
            for r in range(n_orbs):
                for s in range(n_orbs):
                    op = FermionOperator(
                        ((p, 1), (q, 1), (r, 0), (s, 0)),
                        1.0,
                    )
                    gamma2[p, q, r, s] = quimb_expect_fermion_operator(
                        mps,
                        op,
                        n_orbs,
                        cutoff=cutoff,
                    )

    return gamma1, gamma2

def pack_kappa(K):
    """
    Pack upper-triangular entries of a real antisymmetric matrix K.
    """
    n = K.shape[0]
    return np.array([K[i, j] for i in range(n) for j in range(i + 1, n)])


def unpack_kappa(x, n):
    """
    Build real antisymmetric K from packed upper-triangular parameters.
    """
    K = np.zeros((n, n), dtype=float)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = x[k]
            K[j, i] = -x[k]
            k += 1
    return K

from openfermion import InteractionRDM

def interaction_energy(io, gamma1, gamma2):
    """
    Energy using OpenFermion's own InteractionRDM convention.

    gamma1[p, q] = <a†_p a_q>
    gamma2[p, q, r, s] = <a†_p a†_q a_r a_s>
    """
    rdm = InteractionRDM(
        one_body_tensor=gamma1,
        two_body_tensor=gamma2,
    )
    return float(np.real_if_close(rdm.expectation(io)).real)

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from openfermion import normal_ordered
from openfermion.transforms import get_interaction_operator


def unpack_kappa(x, n):
    """
    Build real antisymmetric K from packed upper-triangular parameters.
    """
    K = np.zeros((n, n), dtype=float)

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = x[k]
            K[j, i] = -x[k]
            k += 1

    return K


def rotate_one_rdm(gamma1, U):
    """
    Rotate 1-RDM.

    gamma1[p, q] = <a†_p a_q>

    Returns:
        gamma1_rot[p, q]
    """
    return np.einsum(
        "pa,ab,qb->pq",
        U,
        gamma1,
        U,
        optimize=True,
    )


def rotate_two_rdm(gamma2, U):
    """
    Rotate 2-RDM.

    gamma2[p, q, r, s] = <a†_p a†_q a_r a_s>

    Returns:
        gamma2_rot[p, q, r, s]
    """
    x = np.einsum("pa,abcd->pbcd", U, gamma2, optimize=True)
    x = np.einsum("qb,pbcd->pqcd", U, x, optimize=True)
    x = np.einsum("rc,pqcd->pqrd", U, x, optimize=True)
    x = np.einsum("sd,pqrd->pqrs", U, x, optimize=True)

    return x


def energy_from_rotated_rdms(
    h1,
    h2,
    ecore,
    gamma1,
    gamma2,
    U,
    use_transpose=False,
):
    """
    Evaluate energy by rotating the RDMs and contracting with fixed integrals.

    If the result does not match your validated Hamiltonian-rotation result,
    try use_transpose=True.
    """
    Ur = U.T if use_transpose else U

    gamma1_rot = rotate_one_rdm(gamma1, Ur)
    gamma2_rot = rotate_two_rdm(gamma2, Ur)

    e = ecore
    e += np.einsum("pq,pq->", h1, gamma1_rot, optimize=True)
    e += np.einsum("pqrs,pqrs->", h2, gamma2_rot, optimize=True)

    return float(np.real_if_close(e).real)

def optimize_orbital_rotation_rdm_scheme(
    fermion_hamiltonian,
    gamma1,
    gamma2,
    maxiter=200,
    gtol=1e-8,
    method="BFGS",
    n_random_starts=0,
    random_scale=1e-2,
    seed=None,
    use_transpose=False,
    callback=None,
    verbose=False,
):
    """
    Optimize U = exp(K), with K real antisymmetric, using fixed Hamiltonian
    tensors and rotated RDMs.

    Parameters
    ----------
    fermion_hamiltonian:
        OpenFermion FermionOperator convertible to InteractionOperator.

    gamma1:
        1-RDM, gamma1[p, q] = <a†_p a_q>.

    gamma2:
        2-RDM, gamma2[p, q, r, s] = <a†_p a†_q a_r a_s>.

    n_random_starts:
        Number of random starts in addition to the zero start.

    random_scale:
        Standard deviation for random K parameters.

    use_transpose:
        Switches U -> U.T in the RDM rotation. Use whichever validates
        against your original K=0 / Hamiltonian-rotation convention.

    Returns
    -------
    best:
        scipy OptimizeResult with extra fields:
            best.K
            best.U
            best.energy
            best.all_results
            best.h1
            best.h2
            best.ecore
    """
    io = get_interaction_operator(normal_ordered(fermion_hamiltonian))

    ecore = io.constant
    h1 = np.asarray(io.one_body_tensor)
    h2 = np.asarray(io.two_body_tensor)

    n_orbs = h1.shape[0]
    n_params = n_orbs * (n_orbs - 1) // 2

    gamma1 = np.asarray(gamma1)
    gamma2 = np.asarray(gamma2)

    rng = np.random.default_rng(seed)

    def objective(x):
        K = unpack_kappa(x, n_orbs)
        U = expm(K)

        return energy_from_rotated_rdms(
            h1=h1,
            h2=h2,
            ecore=ecore,
            gamma1=gamma1,
            gamma2=gamma2,
            U=U,
            use_transpose=use_transpose,
        )

    starts = [np.zeros(n_params, dtype=float)]

    for _ in range(n_random_starts):
        starts.append(rng.normal(0.0, random_scale, size=n_params))

    all_results = []

    for istart, x0 in enumerate(starts):
        if verbose:
            print(f"Starting optimization {istart + 1}/{len(starts)}")

        res = minimize(
            objective,
            x0,
            method=method,
            options={
                "maxiter": maxiter,
                "gtol": gtol,
            },
            callback=callback,
        )

        K_opt = unpack_kappa(res.x, n_orbs)
        U_opt = expm(K_opt)

        res.K = K_opt
        res.U = U_opt
        res.energy = float(res.fun)
        res.start_index = istart
        res.x0 = x0

        all_results.append(res)

        if verbose:
            print(
                f"  start={istart}, "
                f"success={res.success}, "
                f"energy={res.energy:.12f}"
            )

    best = min(all_results, key=lambda r: r.energy)

    best.all_results = all_results
    best.h1 = h1
    best.h2 = h2
    best.ecore = ecore

    return best

def optimize_orbital_rotation_from_quimb_mps(
    fermion_hamiltonian,
    mps,
    n_orbs=None,
    rdm_cutoff=1e-12,
    maxiter=200,
    gtol=1e-8,
    method="BFGS",
    n_random_starts=0,
    random_scale=1e-2,
    seed=None,
    verbose=False,
):
    """
    End-to-end helper:

        FermionOperator + quimb MPS
            -> compute 1-/2-RDMs
            -> optimize U = exp(K)
            -> return OptimizeResult with U, K, energy, gamma1, gamma2

    Runs one zero start plus n_random_starts random starts.
    """
    io0 = get_interaction_operator(normal_ordered(fermion_hamiltonian))

    if n_orbs is None:
        n_orbs = io0.one_body_tensor.shape[0]

    gamma1, gamma2 = compute_rdms_from_quimb_mps(
        mps,
        n_orbs=n_orbs,
        cutoff=rdm_cutoff,
        verbose=verbose,
    )

    res = optimize_orbital_rotation_rdm_scheme(
        fermion_hamiltonian=fermion_hamiltonian,
        gamma1=gamma1,
        gamma2=gamma2,
        maxiter=maxiter,
        gtol=gtol,
        method=method,
        n_random_starts=n_random_starts,
        random_scale=random_scale,
        seed=seed,
        use_transpose=False,
        verbose=True,
    )

    res.gamma1 = gamma1
    res.gamma2 = gamma2

    return res

if __name__ == "__main__":


    directory = "./saved/hamiltonians/"
    system = 'H2O_corr'
        
    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    HQ = jordan_wigner(H)
    n_qubits = count_qubits(HQ)

    #solve for initial state
    compress_cutoff = 1e-20
    verbose=False

    mpo = MPO_from_QubitOperator(HQ, max_bond = None, mpo_cutoff = compress_cutoff, 
                                 verbose = verbose, compression_freq = 20)
    seed = 0
    bd = 5
    reps =5
    bsz = 2
    n_sweeps = 50
    sweep_tol = 1e-6
    tol=1e-3
    states = []
    energies = []
    best_mps = None
    min_diff = 100

    print(f'Starting max bd = {bd}, {reps} reps')
    for r in range(reps):
        guess_mps = qtn.MPS_rand_state(n_qubits, bd)
        
        dmrg = qtn.DMRG(mpo, bd, bsz = bsz, cutoffs = compress_cutoff, p0 = guess_mps)
        dmrg.opts['local_eig_tol'] = 1e-3
        dmrg.opts['pempsriodic_compress_ham_eps'] = compress_cutoff
        dmrg.opts['periodic_compress_norm_eps'] = compress_cutoff
        dmrg_conv = dmrg.solve(tol=sweep_tol, bond_dims=bd , max_sweeps = n_sweeps, 
                        sweep_sequence = 'RL', verbosity = 0, 
                        suppress_warnings = False, cutoffs = compress_cutoff)

        diff = abs(dmrg.energy - fci_e)
        
        states.append(dmrg.state)
        energies.append(dmrg.energy)

    pairs = sorted(zip(energies, states)) 

    lowest_energy, psi = pairs[0]
    if abs(lowest_energy - fci_e) < 1.6e-3:
        print("DMRG converged at bond dimension: {}.\nReduce reference bond dimension".format(bd))

    print("Starting reference energy: ", lowest_energy)
    print("FCI energy: ", fci_e)
    res = optimize_orbital_rotation_from_quimb_mps(
        fermion_hamiltonian=H,
        mps=psi,
        maxiter=200,
        gtol=1e-7,
        n_random_starts=1,
        random_scale=0.05,
        seed=0,
        verbose=True,
    )

    print("Best start index:", res.start_index)
    print("Best energy:", res.energy)
    print("All energies:", [r.energy for r in res.all_results])

    io = get_interaction_operator(normal_ordered(H))

    E_rdm = interaction_energy(io, res.gamma1, res.gamma2)

    E_direct = quimb_expect_fermion_operator(
        psi,
        H,
        n_qubits=io.one_body_tensor.shape[0],
    )

    print("RDM energy:   ", E_rdm)
    print("Direct energy:", E_direct)
    print("Difference:   ", E_rdm - E_direct.real)

    U = res.U
    K = res.K

    import copy
    from openfermion import InteractionRDM

    io0 = get_interaction_operator(normal_ordered(H))
    rdm0 = InteractionRDM(res.gamma1, res.gamma2)

    x_test = np.random.default_rng(123).normal(
        0.0,
        1e-2,
        size=io0.one_body_tensor.shape[0] * (io0.one_body_tensor.shape[0] - 1) // 2,
    )

    K_test = unpack_kappa(x_test, io0.one_body_tensor.shape[0])
    U_test = expm(K_test)

    io_rot = copy.deepcopy(io0)
    io_rot.rotate_basis(U_test)

    E_ham_rot = float(np.real_if_close(rdm0.expectation(io_rot)).real)

    E_rdm_rot = energy_from_rotated_rdms(
        h1=io0.one_body_tensor,
        h2=io0.two_body_tensor,
        ecore=io0.constant,
        gamma1=res.gamma1,
        gamma2=res.gamma2,
        U=U_test,
        use_transpose=False,
    )

    print("Hamiltonian rotation:", E_ham_rot)
    print("RDM rotation U:      ", E_rdm_rot, "diff =", E_rdm_rot - E_ham_rot)
    
    ### (Pauli) DMRG bond dimension after optimization
    from openfermion import get_fermion_operator, jordan_wigner, get_ground_state, get_sparse_operator
    import quimb.tensor as qtn
    Hrot = get_fermion_operator(io_rot)
    HQ_rot = jordan_wigner(Hrot)
    e, gs_rot = get_ground_state(get_sparse_operator(HQ_rot, n_qubits))

    print("Rotated:")
    gs_rot_mps = qtn.MatrixProductState.from_dense(gs_rot, cutoff = compress_cutoff)  
    result = find_dmrg_conv_bd_quimb(HQ_rot, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                            reps=1, verbose=False, compress_cutoff = compress_cutoff, sweep_tol = 1e-6,
                            noise = 1e0, bsz=2, guess_mps = gs_rot_mps, seed=0)

    print("Unrotated:")
    gs_mps = qtn.MatrixProductState.from_dense(fci_gs, cutoff = compress_cutoff)  
    og_result = find_dmrg_conv_bd_quimb(HQ, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                            reps=1, verbose=False, compress_cutoff = compress_cutoff, sweep_tol = 1e-6,
                            noise = 1e0, bsz=2, guess_mps = gs_mps, seed=0)

    