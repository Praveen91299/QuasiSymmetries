from openfermion import QubitOperator
from openfermion.utils import count_qubits
from copy import deepcopy
from src.op_utils import has_complex_entries
import quimb.tensor as qtn
import numpy as np

try:
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except ImportError:
    DMRGDriver = None
    SymmetryTypes = None


def _require_pyblock2():
    if DMRGDriver is None or SymmetryTypes is None:
        raise ImportError(
            "pyblock2 is required for the block2 DMRG helpers in src.tn. "
            "Use find_dmrg_conv_bd_quimb for the quimb-only DMRG path."
        )

def QO_to_block2_Pauli(Operator: QubitOperator, n_qubits, tol=1e-5):
    """
    Returns Pauli term, constant for input to block2's mpo driver. Use the following code to initialize mpo

    driver = DMRGDriver(
        scratch="./tmp_block2_pauli",
        symm_type=SymmetryTypes.SGB,
        n_threads=4,
    )

    # In Pauli mode, only n_sites is required.
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)

    # Build MPO directly from the Pauli strings
    mpo = driver.get_mpo_any_pauli(paulis, ecore=const)


    """
    op = deepcopy(Operator)
    terms, constant = [], op.constant
    op -= constant
    op.compress()

    for term, coeff in op.terms.items():
        if abs(coeff) >= tol:
            ops = ["I"]*n_qubits

            for pauli in term:
                ops[pauli[0]] = pauli[1]
            
            st = "".join(ops)
            terms.append((st, coeff))
    
    return terms, constant

def get_mpo_any_pauli_complex(driver, op_list, ecore=None, **kwargs):
    """
    Complex-compatible replacement for driver.get_mpo_any_pauli.

    This removes the even-Y assertion and keeps the correct phase from
    physical Pauli Y operators.

    Requires:
        driver = DMRGDriver(symm_type=SymmetryTypes.SGB | SymmetryTypes.CPX)
        driver.initialize_system(n_sites=n_qubits, pauli_mode=True)
    """
    builder = driver.expr_builder()

    if ecore is not None and abs(ecore) > 0:
        builder.add_const(ecore)

    for ops, coeff in op_list:
        idxs = []
        op_chars = []

        for i, op in enumerate(ops):
            if op != "I":
                op_chars.append(op)
                idxs.append(i)

        if len(op_chars) == 0:
            builder.add_const(coeff)
            continue

        num_y = op_chars.count("Y")

        # pyblock2's Pauli-mode Y is effectively real -i*sigma_y,
        # so physical sigma_y contributes a factor of i.
        coeff_block2 = coeff * (1j ** num_y)

        builder.add_term("".join(op_chars), idxs, coeff_block2)

    expr = builder.finalize()
    return driver.get_mpo(expr, **kwargs)


def QO_to_block2_MPO_complex(HQ: QubitOperator, n_qubits: int):
    """
    Build a complex-compatible pyblock2 MPO from an OpenFermion QubitOperator.
    """
    _require_pyblock2()
    paulis, const = QO_to_block2_Pauli(HQ, n_qubits)

    driver = DMRGDriver(
        scratch="./tmp_block2_pauli",
        symm_type=SymmetryTypes.SGB | SymmetryTypes.CPX,
        n_threads=4,
        n_mkl_threads=1
    )

    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)
    mpo = get_mpo_any_pauli_complex(driver, paulis, ecore=const)

    return mpo, driver

def QO_to_block2_MPO(HQ, n_qubits):
    """
    
    """
    _require_pyblock2()
    paulis, const = QO_to_block2_Pauli(HQ, n_qubits)
    
    driver = DMRGDriver(
        scratch="./tmp_block2_pauli",
        symm_type=SymmetryTypes.SGB,
        n_threads=4,
        n_mkl_threads=1
    )

    # In Pauli mode, only n_sites is required.
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)
    mpo = driver.get_mpo_any_pauli(paulis, ecore=const)

    return mpo, driver

def find_dmrg_conv_bd(HQ, n_qubits, exact_energy, max_bd, tol=1e-3, n_sweeps=8, reps=1, verbose=False):
    """
    Repeats DMRG for upto max_bd, till convergence or reaches exact_energy within tol
    Uses pyblock2GM

    """
    #detect complex HQ
    is_cpx = has_complex_entries(HQ)

    if is_cpx:
        mpo, driver = QO_to_block2_MPO_complex(HQ, n_qubits)
    else:
        mpo, driver = QO_to_block2_MPO(HQ, n_qubits)

    print(driver.symm_type)
    # In Pauli mode, only n_sites is required.

    for bd in range(1, max_bd+1):
        if verbose: print("Bond dimension: ", bd)

        for r in range(reps):
            ket = driver.get_random_mps(tag="KET", bond_dim=bd, nroots=1) #nroots corresponds to number of MPS >1 for excited states

            # Run DMRG

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

            if verbose: print("Energy difference: {}".format(abs(energy - exact_energy)))

            if abs(energy - exact_energy) <= tol:
                if verbose: print("DMRG converged at bond dimension: {}".format(bd))

                return bd
    
    print("Not converged to exact energy with {} bond dimension.".format(max_bd))
    return False

def find_dmrg_conv_bd_mod(HQ, n_qubits, exact_energy, max_bd, tol=1e-3, n_sweeps=8, reps=1, verbose=False):
    """
    Repeats DMRG for upto max_bd, till convergence or reaches exact_energy within tol
    Uses pyblock2GM

    """
    #detect complex HQ
    is_cpx = has_complex_entries(HQ)

    if is_cpx:
        mpo, driver = QO_to_block2_MPO_complex(HQ, n_qubits)
    else:
        mpo, driver = QO_to_block2_MPO(HQ, n_qubits)

    print(driver.symm_type)
    # In Pauli mode, only n_sites is required.

    if verbose: 
        iprint = 1
    else:
        iprint = 0

    for bd in range(1, max_bd+1):
        if verbose: print("Bond dimension: ", bd)

        if bd == 1:
            current_energy = 0
            for r in range(reps):
                ket = driver.get_random_mps(tag="KET", bond_dim=bd, nroots=1) #nroots corresponds to number of MPS >1 for excited states
                # Run DMRG

                energy = driver.dmrg(
                    mpo,
                    ket,
                    n_sweeps=n_sweeps,
                    bond_dims=[bd],
                    noises=[1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6] + [0.0]*(n_sweeps - 6),
                    thrds=[1e-10] * n_sweeps,
                    dav_max_iter=50,
                    iprint=iprint
                )

                if verbose: print("Energy difference: {}".format(abs(energy - exact_energy)))

                if energy < current_energy:
                    current_energy = energy
                    current_ket = ket.deep_copy(f"best_bd_{bd}")

                if abs(energy - exact_energy) <= tol:
                    if verbose: print("DMRG converged at bond dimension: {}".format(bd))

                    return bd
                
        else:

            energy = driver.dmrg(
                mpo,
                current_ket,
                n_sweeps=n_sweeps,
                bond_dims=[bd],
                noises=[1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6] + [0.0]*(n_sweeps - 8),
                thrds=[1e-10] * n_sweeps,
                dav_max_iter=50,
                iprint=iprint
            )

            if verbose: print("Energy difference: {}".format(abs(energy - exact_energy)))

            if energy < current_energy:
                current_energy = energy
                #current_ket is already updated in place

            if abs(energy - exact_energy) <= tol:
                if verbose: print("DMRG converged at bond dimension: {}".format(bd))

                return bd

    print("Not converged to exact energy with {} bond dimension.".format(max_bd))
    return False

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


def find_dmrg_conv_bd_quimb(Hq, n_qubits, exact_energy, bd_list = None, tol=1.6e-3, n_sweeps=10, 
                            reps=1, verbose=False, compress_cutoff = 1e-10, sweep_tol = 1e-6,
                            noise = 1e-3, bsz = 2, guess_mps = None, seed=None, return_data=False):

    mpo = MPO_from_QubitOperator(Hq, max_bond = None, mpo_cutoff = compress_cutoff, 
                                 verbose = verbose, compression_freq = 20)

    if verbose:
        verbosity = 2
    else:
        verbosity = 0

    if seed is not None:
        np.random.seed(seed)

    if bd_list is None:
        bd_list = [i for i in range(1,11,1)] + [i for i in range(12,21,2)] + [i for i in range(30,101,10)]

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
                
                if return_data:
                    print("Returning MPO...")
                    data = {
                        "mpo": mpo
                    }
                    return bd, dmrg.energy, data
                else:
                    return bd, dmrg.energy
            
    print(f'DMRG not converged at bd = {bd_list[-1]}')
    
    if return_data:
        print("Returning MPO...")
        data = {
            "mpo": mpo
        }
        return bd_list[-1], dmrg.energy, data
    else:
        return bd_list[-1], dmrg.energy
