### tensor network benchmark related utils

from openfermion import QubitOperator
from copy import deepcopy
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

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

def QO_to_block2_MPO(HQ, n_qubits):
    """
    
    """
    paulis, const = QO_to_block2_Pauli(HQ, n_qubits)
    driver = DMRGDriver(
        scratch="./tmp_block2_pauli",
        symm_type=SymmetryTypes.SGB,
        n_threads=None,
    )

    # In Pauli mode, only n_sites is required.
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)

    # Build MPO directly from the Pauli strings
    mpo = driver.get_mpo_any_pauli(paulis, ecore=const)

    return mpo


def find_dmrg_conv_bd(HQ, n_qubits, exact_energy, max_bd, tol=1e-3, n_sweeps=8, reps=1):
    """
    Repeats DMRG for upto max_bd, till convergence or reaches exact_energy within tol

    """

    driver = DMRGDriver(
        scratch="./tmp_block2_pauli",
        symm_type=SymmetryTypes.SGB,
        n_threads=None,
    )

    # In Pauli mode, only n_sites is required.
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)
    mpo = QO_to_block2_MPO(HQ, n_qubits)

    for bd in range(1, max_bd+1):
        print("Bond dimension: ", bd)

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

            print("Energy difference: {}".format(abs(energy - exact_energy)))

            if abs(energy - exact_energy) <= tol:
                print("DMRG converged at bond dimension: {}".format(bd))

                return bd
    
    print("Not converged to exact energy with {} bond dimension.".format(max_bd))
    return False