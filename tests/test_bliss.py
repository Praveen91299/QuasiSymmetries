import numpy as np
from openfermion import FermionOperator, get_sparse_operator

from quasisymmetries.bliss import lp_bliss_paper_real_pauli_1norm


def test_lp_bliss_uses_sparse_constraints_and_preserves_target_sector():
    hamiltonian = (
        FermionOperator("0^ 0", -1.0)
        + FermionOperator("1^ 1", -0.5)
        + FermionOperator("0^ 1", 0.2)
        + FermionOperator("1^ 0", 0.2)
    )
    shifted, info = lp_bliss_paper_real_pauli_1norm(
        hamiltonian,
        n_electrons=1,
        n_orb=2,
    )

    assert info["success"]
    assert info["lp_constraint_matrix_format"] == "sparse_csc"
    assert info["lp_killer_matrix_nnz"] > 0
    assert info["final_pauli_l1"] <= info["initial_pauli_l1"] + 1e-10

    original_matrix = get_sparse_operator(hamiltonian, n_qubits=2).toarray()
    shifted_matrix = get_sparse_operator(shifted, n_qubits=2).toarray()
    one_electron_indices = np.asarray([1, 2])
    assert np.allclose(
        original_matrix[np.ix_(one_electron_indices, one_electron_indices)],
        shifted_matrix[np.ix_(one_electron_indices, one_electron_indices)],
    )
