# %%
import pickle
import quimb.tensor as qtn
import numpy as np
from openfermion import count_qubits, jordan_wigner, MolecularData, get_sparse_operator
from quasisymmetries.state_utils import get_hf_wfn, get_hf_occ

def MPO_from_QubitOperator(H, max_bond = None, mpo_cutoff = 1e-10, verbose = True,
                           compression_freq = 20):
    """
    Make an MPO for operator H which is an Openfermion QubitOperator.
    """

    n = count_qubits(H)
    Zero2 = np.zeros((2, 2), dtype = float)

    #Initialize zero MPO
    mpo =  qtn.MPO_product_operator([Zero2] * n)

    coeffs, ops = get_coeffs_and_ops(H, n)

    for i, (coeff, op)  in enumerate(zip(coeffs, ops)):
        mpo += coeff * qtn.MPO_product_operator( op )
        
        if mpo_cutoff is not None and i % compression_freq == 0:
           mpo.compress(max_bond  = max_bond, cutoff = mpo_cutoff)

    if mpo_cutoff is not None:
           mpo.compress(max_bond  = max_bond, cutoff = mpo_cutoff)
           
    if verbose:
            print(f'Bond dimensions of MPO: {mpo.bond_sizes()}')
    
    return mpo

def get_coeffs_and_ops(of_op, n_qubits):
    """
    Returns:
        coeffs: list of coefficients
        ops: list of lists of 2x2 matrices (one list per term, length = n_qubits)
    """

    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    pauli_map = {'X': X, 'Y': Y, 'Z': Z}

    coeffs = []
    ops_list = []

    for term, coeff in of_op.terms.items():

        # Start with identity everywhere
        ops = [I.copy() for _ in range(n_qubits)]

        # Fill non-identity Paulis
        for qubit, pauli in term:
            ops[qubit] = pauli_map[pauli]

        coeffs.append(coeff)
        ops_list.append(ops)

    return coeffs, ops_list


def find_dmrg_conv_bd_quimb(Hq, n_qubits, exact_energy, bd_list, tol=1e-3, n_sweeps=10, 
                            reps=1, verbose=False, compress_cutoff = 1e-10, sweep_tol = 1e-6,
                            noise = 1e-3, bsz = 1, guess_mps = None):

    mpo = MPO_from_QubitOperator(Hq, max_bond = None, mpo_cutoff = compress_cutoff, 
                                 verbose = verbose, compression_freq = 20)

    if verbose:
        verbosity = 2
    else:
        verbosity = 0

    for bd in bd_list:
        if verbose: print(f'Starting bd = {bd}')
        if bd == 1:
            current_energy = 0.0
            for r in range(reps):
                if guess_mps is None:
                    guess_mps = qtn.MPS_rand_state(n_qubits, 1)
                dmrg = qtn.DMRG(mpo, bd, bsz = bsz, cutoffs = compress_cutoff, p0 = guess_mps)
                dmrg.opts['local_eig_tol'] = 1e-3
                dmrg.opts['pempsriodic_compress_ham_eps'] = compress_cutoff
                dmrg.opts['periodic_compress_norm_eps'] = compress_cutoff
                dmrg_conv = dmrg.solve(tol=sweep_tol, bond_dims=bd , max_sweeps = n_sweeps, 
                                sweep_sequence = 'RL', verbosity = verbosity, 
                                suppress_warnings = False, cutoffs = compress_cutoff)

                if dmrg.energy.real < current_energy.real:
                        current_energy = dmrg.energy.copy()
                        current_ket = dmrg.state.copy()

                if abs(dmrg.energy - exact_energy) <= tol:
                    print("DMRG converged at bond dimension: {}".format(bd))
                    
                    return bd, dmrg.energy
                
        else:
            #Use mps from previous bd as a guess
            if bsz ==1:
                guess_mps = current_ket.copy() 
            elif bsz==2:
                #bsz==2 doesn't add noise in quimb, so we add it
                guess_mps = current_ket.copy() + noise*qtn.MPS_rand_state(n_qubits, bond_dim=1)
                guess_mps.normalize()

            dmrg = qtn.DMRG(mpo, bd, bsz = bsz, cutoffs = compress_cutoff, p0 = guess_mps)
            dmrg.opts['pempsriodic_compress_ham_eps'] = compress_cutoff
            dmrg.opts['periodic_compress_norm_eps'] = compress_cutoff
            dmrg.opts['bond_expand_rand_strength'] = noise
            dmrg_conv = dmrg.solve(tol=sweep_tol, bond_dims=bd , max_sweeps = n_sweeps, 
                            sweep_sequence = 'RL', verbosity = verbosity, 
                            suppress_warnings = False, cutoffs = compress_cutoff)
        
            if dmrg.energy.real < current_energy.real:
                current_energy = dmrg.energy.copy()
                current_ket = dmrg.state.copy()

            if abs(dmrg.energy - exact_energy) <= tol:
                print("DMRG converged at bond dimension: {}".format(bd))
                return bd, dmrg.energy
            
    print(f'DMRG not converged at bd = {bd_list[-1]}')

    return bd_list[-1], dmrg.energy

def find_dmrg_conv_bd_quimb2(Hq, n_qubits, exact_energy, bd_list, tol=1e-3, n_sweeps=10, 
                            reps=1, verbose=False, compress_cutoff = 1e-10, sweep_tol = 1e-6,
                            noise = 1e-3, bsz = 1, guess_mps = None):

    mpo = MPO_from_QubitOperator(Hq, max_bond = None, mpo_cutoff = compress_cutoff, 
                                 verbose = verbose, compression_freq = 20)

    if verbose:
        verbosity = 2
    else:
        verbosity = 0

    for bd in bd_list:
        if verbose: print(f'Starting max bd = {bd}')
        for r in range(reps):
            if guess_mps is None:
                guess_mps = qtn.MPS_rand_state(n_qubits, 1)
            else:
                guess_mps += noise*qtn.MPS_rand_state(n_qubits, bond_dim=1)
                guess_mps.normalize() 
            dmrg = qtn.DMRG(mpo, bd, bsz = bsz, cutoffs = compress_cutoff, p0 = guess_mps)
            dmrg.opts['local_eig_tol'] = 1e-3
            dmrg.opts['pempsriodic_compress_ham_eps'] = compress_cutoff
            dmrg.opts['periodic_compress_norm_eps'] = compress_cutoff
            dmrg_conv = dmrg.solve(tol=sweep_tol, bond_dims=bd , max_sweeps = n_sweeps, 
                            sweep_sequence = 'RL', verbosity = verbosity, 
                            suppress_warnings = False, cutoffs = compress_cutoff)

            if abs(dmrg.energy - exact_energy) <= tol:
                print("DMRG converged at bond dimension: {}".format(bd))
                
                return bd, dmrg.energy
            
    print(f'DMRG not converged at bd = {bd_list[-1]}')

    return bd_list[-1], dmrg.energy

def get_hf_mps(hf_occ):
    
    arrays = []

    for i in hf_occ:
        if i == 1:
            arrays.append([0.0, 1.0])
        else:
            arrays.append([1.0, 0.0])

    return qtn.MPS_product_state(arrays) 

directory = "./saved/hamiltonians/"


systems = [
    'H4chain_eqm',
    'H4chain_corr',
    'H4chain_diss',
    'H4rect_corr',
    'H4rect_diss',
    'LiH_eqm',
    'LiH_corr',
    'H2O_eqm',
    'H2O_corr',
    'H2O_diss',
    'N2frozen_eqm',
    'N2frozen_corr',
    'N2frozen_diss'
]

outfile = "results_fci_guess.txt"

headers = ["System", "FCI", "CISD error", "HF error", "DMRG error", " FCI BD", "CISD BD", "DMRG BD"]
widths = [20] + [15] * 4 + [10] * 3


with open(outfile, "w") as f:

    # write header
    header_line = "".join(
        f"{h:<{w}}" for h, w in zip(headers, widths)
    )

    f.write(header_line + "\n")
    f.write("-" * sum(widths) + "\n")


for system in systems:
    filename= system
    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    HQ = jordan_wigner(H)
    molecule = MolecularData(filename=directory+system)
    n_qubits = count_qubits(HQ)
    Hs = get_sparse_operator(HQ, n_qubits)
    compress_cutoff =  1e-20
    hf_occ = get_hf_occ(molecule.n_electrons, molecule.n_orbitals)
    hf_gs = get_hf_wfn(hf_occ)
    fci_gs_mps = qtn.MatrixProductState.from_dense(fci_gs, cutoff = compress_cutoff)
    cisd_gs_mps = qtn.MatrixProductState.from_dense(cisd_gs, cutoff = compress_cutoff)
    hf_gs_mps =  get_hf_mps(hf_occ)

    #mpo = MPO_from_QubitOperator(HQ, max_bond = None, mpo_cutoff = compress_cutoff, 
    #                                verbose = False, compression_freq = 20)
    bd_list = [i for i in range(1,11,1)] + [i for i in range(12,21,2)] + [i for i in range(30,101,10)]
    noise = 1.0

    guess_mps = fci_gs_mps.copy()
    reps = 1
       
    dmrg_bd, dmrg_e = find_dmrg_conv_bd_quimb2(HQ, n_qubits, fci_e, bd_list, tol=1.6e-3, n_sweeps=100, 
                            reps=reps, verbose=True, compress_cutoff = compress_cutoff, 
                            sweep_tol = 1e-6, noise = noise, bsz=2, guess_mps = guess_mps)
    

    row = [system, fci_e, cisd_e - fci_e, molecule.hf_energy - fci_e, dmrg_e.real - fci_e, 
            max((fci_gs_mps.bond_sizes())), max(cisd_gs_mps.bond_sizes()), dmrg_bd]
    line = (
            f"{row[0]:<{widths[0]}s}"
            f"{row[1]:<{widths[1]}.10f}"
            f"{row[2]:<{widths[2]}.5e}"
            f"{row[3]:<{widths[3]}.5e}"
            f"{row[4]:<{widths[4]}.5e}"
            f"{row[5]:<{widths[4]}d}"
            f"{row[6]:<{widths[4]}d}"
            f"{row[7]:<{widths[4]}d}"
        )
    with open(outfile, "a") as f:
                f.write(line + "\n")
    
    #print(f'Max BD in fci_gs = {(fci_gs_mps.bond_sizes())}')
    #print(f'Max BD in cisd_gs = {(cisd_gs_mps.bond_sizes())}')
    #print(f'<CISD|H|CISD> in tn: {(cisd_gs_mps.H & mpo.apply(cisd_gs_mps)).contract()}')
    #print(f'<HF|H|HF)> in tn: {(hf_gs_mps.H & mpo.apply(hf_gs_mps)).contract()}')
    


# %%
