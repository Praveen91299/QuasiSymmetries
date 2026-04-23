### tensor network benchmark related utils

from openfermion import QubitOperator
from copy import deepcopy

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