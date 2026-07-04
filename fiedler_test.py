### Fiedler ordering for 

### June 8, 2026

from quasisymmetries.tn import find_dmrg_conv_bd_quimb

import pickle
import quimb.tensor as qtn
import numpy as np
from openfermion import count_qubits, jordan_wigner, MolecularData, get_sparse_operator
from quasisymmetries.state_utils import get_hf_wfn, get_hf_occ
from quasisymmetries.metrics import get_permuted_bipartite_entanglement, comm_sq_exp_fast, get_entropies_at_cuts
from quasisymmetries.sym import get_seniority_symmetries, hct_mod
from quasisymmetries.bliss import lp_bliss_paper_real_pauli_1norm
from quasisymmetries.benchmark import benchmark_syms, BenchmarkData
import pandas as pd
from quasisymmetries.fiedler import fiedler_order_from_state, reorder_statevector_axes
from quasisymmetries.bs.beam import find_commuting_symmetry_generators

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from quasisymmetries.benchmark import BenchmarkData
from quasisymmetries.fiedler import do_fiedler_reordering

import quimb.tensor as qtn
from quasisymmetries.tn import find_dmrg_conv_bd_quimb

directory = "./saved/hamiltonians/"

systems = [
    'LIH_corr'
]
#     'H4chain_eqm',
#     'H4chain_corr',
#     'H4chain_diss',
#     'H4rect_corr',
#     'H4rect_diss',
#     'LiH_eqm',
#     'LiH_corr',
#     'H2O_eqm',
#     'H2O_corr',
#     'H2O_diss',
#     'N2frozen_eqm',
#     'N2frozen_corr',
#     'N2frozen_diss'
# ]

#options
verbose=True
bd_rows = []

date="_JUNE14" #to keep track of outputs
cost_func_tag = '_nc_exp_cisd'
output_filename = "./saved/" + cost_func_tag + date + "_fiedler"

for system in systems:

    with open(output_filename, 'a') as f:
        print('\n\n' + system, file=f)
    
    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    HQ = jordan_wigner(H)
    n_qubits = count_qubits(H)
    Hs = get_sparse_operator(HQ, n_qubits)
    print(system, fci_e)

    log_base=np.e
    ents = get_entropies_at_cuts(fci_gs, n_qubits, log_base=log_base)
    for i, e in enumerate(ents):
        print(i+1, i+2, e)

    
    

    data = BenchmarkData.load_datasets('./saved/results/MAY27/_nc_exp_cisd_MAY27{}_datasets'.format(system))
    print("Importing {} symmetries:".format(data[1].tag))
    syms = data[1].symmetries
    print(syms)

    with open(output_filename, 'a') as f:
        print("\nUsing saved {}:\n".format(data[1].tag), file=f)
        for sym in syms:
            print(sym, file=f)  
    
    ents_rotated, H_perm, gs_rot = get_permuted_bipartite_entanglement(syms, HQ, n_qubits, fci_e, fci_gs, return_state=True, log_base=log_base, verbose=True)
    with open(output_filename, 'a') as f:
        print("\nEntanglement after packing Clifford:\n", file=f)
        for i, e in enumerate(ents_rotated):
            print("{}|{}: {}".format(i+1, i+2, e), file=f)

    
    ent_reord, H_reord, psi_reord, fiedler_info = do_fiedler_reordering(H_perm, gs_rot, n_qubits=n_qubits, verbose=True, log_base=log_base)

    syms_reordered = [syms[i] for i in fiedler_info["ordering"]]
    with open(output_filename, 'a') as f:
        print("\nReordered symmetries:\n", file=f)
        for i, sym in enumerate(syms_reordered):
            print("Site {}: {}".format(i+1, sym), file=f)  

    with open(output_filename, 'a') as f:
        print("\nEntanglement after Fiedler reordering:\n", file=f)
        for i, e in enumerate(ent_reord):
            print("{}|{}: {}".format(i+1, i+2, e), file=f)
    #reorder bond dimension
    

    gs_rot_mps = qtn.MatrixProductState.from_dense(psi_reord, cutoff = 1e-20)     
    dmrg_bd, _ = find_dmrg_conv_bd_quimb(H_reord, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                        reps=1, verbose=False, compress_cutoff = 1e-20, sweep_tol = 1e-6,
                        noise = 1e0, bsz=2, guess_mps = gs_rot_mps, seed=0)
    print("Rotated, permuted reordered BD:", dmrg_bd)

    with open(output_filename, 'a') as f:
        print("DMRG bond dimension for convergence: {} (before reordering: {})".format(dmrg_bd, data[1].dmrg_bd), file=f)