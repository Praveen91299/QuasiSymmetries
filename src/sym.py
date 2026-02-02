import numpy as np
from src.op_utils import *
from src.gf2_utils import *
from openfermion import count_qubits, jordan_wigner, QubitOperator

def find_approx_symm(H, n_sym=None, num_intervals=100, eps_max=None, verbose=True, print_new=True):
    """
    Perform additive symmetry sweep over epsilon truncation thresholds.
    
    For each epsilon in grid, truncates H, builds constraint matrix A,
    finds nullspace basis, and maintains an additive (nested) basis
    across all epsilon values.
    
    Parameters
    ----------
    H : QubitOperator
        Full Hamiltonian.
    num_intervals : int
        Number of epsilon discretization points.
    eps_max : float, optional
        Maximum epsilon. If None, uses 1.000001 * max|coeff|.
    verbose : bool
        Print progress messages.
    print_new : bool
        Print when new symmetries are discovered.
        
    Returns
    -------
    df : pd.DataFrame
        Per-epsilon summary statistics.
    basis_additive_list : list
        List of additive bases (one per epsilon).
    basis_rref_list : list
        List of RREF bases (fresh basis per epsilon).
    """
    # Fix n once from original H
    n_qubits = count_qubits(H)

    # Epsilon grid
    max_abs = max((abs(c) for c in H.terms.values()), default=0.0)
    if eps_max is None:
        eps_max = max_abs * 1.000001

    if n_sym is None:
        n_sym = n_qubits

    assert n_sym <= n_qubits, "Invalid number of symmetries {} requested for {} qubit Hamiltonian".format(n_sym, n_qubits)

    eps_grid = np.linspace(0.0, eps_max, num_intervals)

    basis_add = np.zeros((0, 2 * n_qubits), dtype=np.uint8)
    op_add = []
    add_epsilon = []

    for idx, eps in enumerate(eps_grid):
        # Truncate H
        Ht = truncate_qubitop(H, float(eps))

        # Build constraint matrix A from truncated H
        Gt, _, _, _ = qubitop_to_G_matrix(Ht, n=n_qubits)
        A = exchange_Gx_Gz(Gt, n_qubits)

        # Find symmetries = nullspace(A)
        basis = gf2_find_commuting_basis(Gt, n_qubits)# gf2_nullspace(A)
        basis_rref, piv = gf2_rref(basis)

        # Additive extension
        basis_add, added = gf2_extend_basis_additive(basis_add, basis_rref)

        # Sanity check: additive basis should lie in current nullspace
        ok_null = gf2_check_commuting(Gt, basis_add, n_qubits)# gf2_check_in_nullspace(A, basis_add)
        assert ok_null, "Symmetry set not in null space of truncated Hamiltonian!"

        # Convert to strings
        add_strs = [symplectic_to_pauli_string(v, n_qubits) for v in added] if added.size else []
        basis_add_strs = [symplectic_to_pauli_string(v, n_qubits) for v in basis_add] if basis_add.size else []

        for i, basis in enumerate(added): #for every added symmetry
            add_epsilon.append(eps)
            op_add.append(QubitOperator(add_strs[i], 1.0))
            if verbose: print("Added Pauli string: {} at threshold: {}".format(add_strs[i], eps))

        if len(basis_add) == n_sym:
            return op_add, add_epsilon
    
    print("Error: Did not find {} symmetries, only {} found!!".format(n_sym, len(op_add)))
    return op_add, add_epsilon