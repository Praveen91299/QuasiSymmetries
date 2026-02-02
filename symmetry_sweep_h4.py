"""
Symmetry Sweep Analysis for Molecular Hamiltonians under Jordan-Wigner Transformation.

This script performs additive symmetry analysis of truncated Jordan-Wigner 
Hamiltonians for H4 linear chain at various bond distances. It identifies 
emergent symmetries as epsilon-truncation thresholds are varied.

Dependencies:
    pip install numpy pandas openfermion openfermionpyscf pyscf

Usage:
    python symmetry_sweep_h4.py
"""

import numpy as np
import pandas as pd
from src.op_utils import *
from src.gf2_utils import *
from openfermion import jordan_wigner

# Epsilon discretization
NUM_EPS_DIVISIONS = 1000

# Radius sweep [1.0..3.0 step 0.2] in Angstroms
R_VALUES = list(np.round(np.arange(1.0, 3.0, 0.2), 10)) + [3.0]

# Molecular parameters
BASIS_NAME = "sto-3g"
MULTIPLICITY = 1
CHARGE = 0

# Epsilon max policy:
#   "per_R_max": eps_max = 1.000001 * max |coeff| of that R's Hamiltonian
#   "global_max": use the maximum over all R (comparable epsilon scale across R)
EPS_POLICY = "per_R_max"

# Output verbosity
VERBOSE_PER_R = True
PRINT_HEAD_ROWS = 3

from openfermion import count_qubits

def build_commutation_constraints_A(G, n):
    """
    Build constraint matrix A for commutation conditions.
    
    For each row g_i in G, the constraint row is (g_z,i | g_x,i) so that:
        (g_z,i | g_x,i) · (s_x | s_z) = 0 (mod 2)
    
    Parameters
    ----------
    G : ndarray
        Symplectic matrix (Gx | Gz) of shape (m, 2n).
    n : int
        Number of qubits.
        
    Returns
    -------
    A : ndarray
        Constraint matrix of shape (m, 2n).
    """
    Gx = G[:, :n]
    Gz = G[:, n:]
    A = np.concatenate([Gz, Gx], axis=1).astype(np.uint8)
    return A

def symmetry_sweep_additive(H, num_intervals=100, eps_max=None, verbose=True, print_new=True):
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
    n_fixed = count_qubits(H)

    # Epsilon grid
    max_abs = max((abs(c) for c in H.terms.values()), default=0.0)
    if eps_max is None:
        eps_max = max_abs * 1.000001

    eps_grid = np.linspace(0.0, eps_max, num_intervals)

    basis_add = np.zeros((0, 2 * n_fixed), dtype=np.uint8)
    basis_additive_list = []
    basis_rref_list = []
    rows = []
    prev_dim = 0

    for idx, eps in enumerate(eps_grid):
        # Truncate H
        Ht = truncate_qubitop(H, float(eps))

        # Build constraint matrix A from truncated H
        Gt, _, _, _ = qubitop_to_G_matrix(Ht, n=n_fixed)
        A = build_commutation_constraints_A(Gt, n_fixed)

        # Find symmetries = nullspace(A)
        basis = gf2_nullspace(A)
        basis_rref, piv = gf2_rref(basis)

        # Additive extension
        basis_add, added = gf2_extend_basis_additive(basis_add, basis_rref)

        # Sanity check: additive basis should lie in current nullspace
        ok_null = gf2_check_in_nullspace(A, basis_add)

        # Convert to strings
        add_strs = [symplectic_to_pauli_string(v, n_fixed) for v in added] if added.size else []
        basis_add_strs = [symplectic_to_pauli_string(v, n_fixed) for v in basis_add] if basis_add.size else []

        # Record results
        rows.append({
            "eps_idx": int(idx),
            "epsilon": float(eps),
            "terms_left": int(len(Ht.terms)),
            "k_nullspace": int(basis.shape[0]),
            "k_additive": int(basis_add.shape[0]),
            "new_added": int(added.shape[0]) if added.size else 0,
            "new_added_syms": " | ".join(add_strs) if add_strs else "",
            "additive_basis_syms": " || ".join(basis_add_strs) if basis_add_strs else "",
            "additive_in_nullspace?": bool(ok_null),
        })

        basis_additive_list.append(basis_add.copy())
        basis_rref_list.append(basis_rref.copy())

        # Print on changes
        if verbose and print_new and basis_add.shape[0] != prev_dim:
            print(f"[eps_idx={idx:4d}] ε={eps:.6g}  terms_left={len(Ht.terms):4d}  "
                  f"k_null={basis.shape[0]:2d}  k_additive={basis_add.shape[0]:2d}  "
                  f"(+{len(add_strs)})  ok_null={ok_null}")
            for s in add_strs:
                print("   +", s)
            prev_dim = basis_add.shape[0]

    df = pd.DataFrame(rows)
    return df, basis_additive_list, basis_rref_list


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run symmetry sweep over all R values."""
    all_rows = []
    basis_additive_by_R = {}

    for R in R_VALUES:
        print(f"\n{'='*20} R = {R} Å {'='*20}")
        
        H, mol = build_H_chain_for_R(R)
        H_R = jordan_wigner(H)
        
        if EPS_POLICY == "per_R_max":
            max_abs = max((abs(c) for c in H_R.terms.values()), default=0.0)
            eps_max = 1.000001 * max_abs
        else:
            eps_max = None

        print(f"JW terms = {len(H_R.terms)}, eps_max = {eps_max:.6g}")

        df_sym, basis_add_list, _basis_rref_list = symmetry_sweep_additive(
            H_R,
            num_intervals=NUM_EPS_DIVISIONS,
            eps_max=eps_max,
            verbose=VERBOSE_PER_R,
            print_new=True,
        )

        # Attach metadata
        df_sym.insert(0, "R_Ang", float(R))
        df_sym["jw_terms_full"] = int(len(H_R.terms))
        df_sym["basis"] = BASIS_NAME
        df_sym["geometry"] = "H4 linear z: [0,R,2R,3R]"
        df_sym["fci_energy"] = float(getattr(mol, "fci_energy", np.nan))

        all_rows.append(df_sym)
        basis_additive_by_R[float(R)] = basis_add_list

        # Quick sanity print
        if PRINT_HEAD_ROWS > 0:
            print("\nHead (sanity):")
            print(df_sym.head(PRINT_HEAD_ROWS)[
                ["R_Ang", "eps_idx", "epsilon", "terms_left", 
                 "k_nullspace", "k_additive", "new_added", "additive_in_nullspace?"]
            ].to_string(index=False))

    # Combined dataframe
    df_R_eps_additive = pd.concat(all_rows, ignore_index=True)

    print(f"\nDONE. Combined df shape: {df_R_eps_additive.shape}")
    
    return df_R_eps_additive, basis_additive_by_R


if __name__ == "__main__":
    df_results, basis_data = main()
