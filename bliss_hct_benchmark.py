### May 26, testing new DMRG calculation stuff

from src.tn import find_dmrg_conv_bd_quimb

import pickle
import quimb.tensor as qtn
import numpy as np
from openfermion import count_qubits, jordan_wigner, MolecularData, get_sparse_operator
from src.state_utils import get_hf_wfn, get_hf_occ
from src.metrics import get_permuted_bipartite_entanglement, comm_sq_exp_fast
from src.sym import get_seniority_symmetries, hct_mod
from src.bliss import lp_bliss_paper_real_pauli_1norm
from benchmark_all import benchmark_syms, BenchmarkData
import pandas as pd

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

#options
verbose=True
bd_rows = []

date="_MAY30" #to keep track of outputs
cost_func_tag = '_nc_exp_cisd'
output_filename = "./saved/" + cost_func_tag + date + "_BLISS"

for system in systems:

    with open(output_filename, 'a') as f:
        print('\n\n' + system, file=f)
    
    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    HQ = jordan_wigner(H)
    molecule = MolecularData(filename=directory+system)
    n_qubits = count_qubits(HQ)
    Hs = get_sparse_operator(HQ, n_qubits)

    n_elec = molecule.n_electrons if system[:2] != 'N2' else molecule.n_electrons - 4 # 4 frozen for N2
    print(n_elec)

    H_bliss, info = lp_bliss_paper_real_pauli_1norm(
        H,
        n_electrons=n_elec,
        n_orb=n_qubits,
    )
    print("Pauli BLISS completed, Relative Pauli L1 reduction: {}".format(info["relative_pauli_l1_reduction"]))
    HQ_bliss = jordan_wigner(H_bliss)

    comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_gs, n_qubits)
    # comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, fci_gs, n_qubits)
    # var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)
    # var_fci = lambda s_list: variance(s_list, fci_gs, n_qubits)

    sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized
    sym_metric_func = lambda s: (-1)*sym_group_score_func([s]) # HCT minimized

    # #sym and rotation
    sym_hct, eps = hct_mod(HQ, n_qubits, sym_metric_func= sym_metric_func, use_coeffs_eps=True)
    sym_hct_bliss, eps_bliss = hct_mod(HQ_bliss, n_qubits, sym_metric_func= sym_metric_func, use_coeffs_eps=True)

    data_hct = benchmark_syms(sym_hct, HQ, fci_gs, fci_e, n_qubits, False, True, print_to_file=output_filename, tag=r"HCT($n_q$)")
    data_bliss_hct = benchmark_syms(sym_hct_bliss, HQ, fci_gs, fci_e, n_qubits, False, True, print_to_file=output_filename, tag=r"Pauli BLISS+HCT($n_q$)")
    
    datasets = [data_hct, data_bliss_hct]
    save_filename = output_filename + system + "_datasets"
    BenchmarkData.save_datasets(datasets, save_filename)
    
    #analysis
    _ = BenchmarkData.plot_cut_entropies(datasets, fci_gs, output_filename + system + "_cutentropy.png")

    cols = ["system"] + [data.tag for data in datasets]
    bd_rows.append(dict(zip(cols, [system] + [data.dmrg_bd for data in datasets])))
    df = pd.DataFrame(bd_rows)
    df.to_csv(output_filename + "_dmrg_bd.csv", index=False)