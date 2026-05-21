import pickle
import quimb.tensor as qtn
import numpy as np
from openfermion import count_qubits, jordan_wigner


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
                            noise = 1e-2, bsz = 1):

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
                guess_mps = qtn.MPS_rand_state(n_qubits, 1)
                dmrg = qtn.DMRG(mpo, bd, bsz = bsz, cutoffs = compress_cutoff, p0 = guess_mps)
                dmrg.opts['local_eig_tol'] = 1e-3
                dmrg.opts['pempsriodic_compress_ham_eps'] = compress_cutoff
                dmrg.opts['periodic_compress_norm_eps'] = compress_cutoff
                dmrg_conv = dmrg.solve(tol=sweep_tol, bond_dims=bd , max_sweeps = n_sweeps, 
                                sweep_sequence = 'RL', verbosity = verbosity, 
                                suppress_warnings = False, cutoffs = compress_cutoff)

                if dmrg.energy < current_energy:
                        current_energy = dmrg.energy.copy()
                        current_ket = dmrg.state.copy()

                if abs(dmrg.energy - exact_energy) <= tol:
                    print("DMRG converged at bond dimension: {}".format(bd))
                    
                    return bd
                
        else:
            #Use mps from previous bd as a guess
            guess_mps = current_ket 
            #can also add small random component
            #guess_mps = current_ket + 0.1*qtn.MPS_rand_state(n_qubits, bond_dim=bd)
            #guess_mps.compress(max_bond = bd)

            dmrg = qtn.DMRG(mpo, bd, bsz = 1, cutoffs = compress_cutoff, p0 = guess_mps)
            dmrg.opts['pempsriodic_compress_ham_eps'] = compress_cutoff
            dmrg.opts['periodic_compress_norm_eps'] = compress_cutoff
            dmrg.opts['bond_expand_rand_strength'] = noise
            dmrg_conv = dmrg.solve(tol=sweep_tol, bond_dims=bd , max_sweeps = n_sweeps, 
                            sweep_sequence = 'RL', verbosity = verbosity, 
                            suppress_warnings = False, cutoffs = compress_cutoff)

            if dmrg.energy < current_energy:
                    current_energy = dmrg.energy.copy()
                    current_ket = dmrg.state.copy()

            if abs(dmrg.energy - exact_energy) <= tol:
                print("DMRG converged at bond dimension: {}".format(bd))
                return bd
            
    print(f'DMRG not converged at bd = {bd_list[-1]}')

    return    


directory = "./saved/hamiltonians/"

system = 'H2O_diss'

filename= system
with open(directory+system+".pkl", "rb") as f:
    data = pickle.load(f)

H, fci_e, fci_gs, cisd_e, cisd_gs = data
HQ = jordan_wigner(H)
n_qubits = count_qubits(HQ)

bd_list = [1] + [2*i for i in range(1,6)] + [i for i in range(10,26,5)] + [i for i in range(50,101,25)]
find_dmrg_conv_bd_quimb(HQ, n_qubits, fci_e, bd_list, tol=1.6e-3, n_sweeps=50, 
                            reps=3, verbose=True, compress_cutoff = 1e-20, 
                            sweep_tol = 1e-6, noise = 1e-3, bsz=1)
print(fci_e, cisd_e)