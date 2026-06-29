#run and benchmark H8cube_3A


import os
import pickle

import pandas as pd
from openfermion import count_qubits, jordan_wigner, get_sparse_operator, MolecularData
from src.bs.beam import find_commuting_symmetry_generators
from src.metrics import comm_sq_exp_fast, variance
from src.sym import get_quartic_symmetries, get_seniority_symmetries, hct_mod, bs_hct
from benchmark_all import BenchmarkData, benchmark_syms
import numpy as np
from src.fiedler import do_fiedler_reordering
from src.bliss import lp_bliss_paper_real_pauli_1norm
from src.sym import hct_mod
from src.bs.beam import BeamSearch_Symmetries

def do_fiedler_analysis(syms, HQ, fci_gs, fci_e, n_qubits, log_base=np.e, verbose=True, write_to_file=False, filename=None):
    ent_reord, H_reord, psi_reord, fiedler_info = do_fiedler_reordering(HQ, fci_gs, n_qubits=n_qubits, verbose=verbose, log_base=log_base)

    syms_reordered = [syms[i] for i in fiedler_info["ordering"]]
    #reorder bond dimension

    if verbose:
        print("\nReordered symmetries:\n")
        for i, sym in enumerate(syms_reordered):
            print("Site {}: {}".format(i+1, sym))  
        
        print("\nEntanglement after Fiedler reordering:\n")
        for i, e in enumerate(ent_reord):
            print("{}|{}: {}".format(i+1, i+2, e))


    gs_rot_mps = qtn.MatrixProductState.from_dense(psi_reord, cutoff = 1e-20)     
    dmrg_bd, _, mpo_data = find_dmrg_conv_bd_quimb(H_reord, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                        reps=1, verbose=False, compress_cutoff = 1e-20, sweep_tol = 1e-6,
                        noise = 1e0, bsz=2, guess_mps = gs_rot_mps, seed=0, return_data=True)
    if verbose: print("Rotated, permuted reordered BD:", dmrg_bd)
    mpo = mpo_data["mpo"]
    mpo_bd = max(mpo.bond_sizes())
    if write_to_file:
        with open(output_filename, 'a') as f:
            print("\n\nFiedler reordering:\nReordered symmetries:\n", file=f)
            for i, sym in enumerate(syms_reordered):
                print("Site {}: {}".format(i+1, sym), file=f)  
            
            print("\nEntanglement after Fiedler reordering:\n", file=f)
            for i, e in enumerate(ent_reord):
                print("{}|{}: {}".format(i+1, i+2, e), file=f)
            
            
            print("DMRG MPO bond dimension: {}".format(mpo_bd), file=f)
            print("DMRG bond dimension for convergence: {}".format(dmrg_bd), file=f) 

    info = {
        "fielder_info": fiedler_info,
        "H_reord": H_reord,
        "psi_reord": psi_reord,
        "sym_reord": syms_reordered,
        "ent_reord": ent_reord,
        "dmrg_bd": dmrg_bd,
        "mpo_bd": mpo_bd,
        "mpo": mpo
    }
    return info

bd_rows = []
mpo_bd_rows = []

directory = "./saved/hamiltonians/"
log_base =np.e
date="_JUNE29" #to keep track of outputs
cost_func_tag = '_nc_exp_cisd'
output_filename = "./saved/results/JUNE29/" + cost_func_tag + date
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

systems = [
    "H4chain_corr",
    "H4chain_diss"
]

for system in systems:

    with open(output_filename, 'a') as f:
        print('\n\n' + system, file=f)

    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    molecule = MolecularData(filename=directory + system)

    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    HQ = jordan_wigner(H)
    n_qubits = count_qubits(H)
    Hs = get_sparse_operator(HQ, n_qubits)

    #define cost function
    comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_gs, n_qubits)
    comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, fci_gs, n_qubits)
    var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)
    var_fci = lambda s_list: variance(s_list, fci_gs, n_qubits)

    sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized
    sym_group_var_func = lambda s_list: (-1)*var_cisd(s_list) # BS score maximized
    sym_metric_func = lambda s: (-1)*sym_group_score_func([s]) # HCT minimized

    cf_dict = {'Comm': sym_group_score_func, 'Var' : sym_group_var_func, '1-norm': None}

    #make symmetries
    bw=16 # beam width for bs-hct and bs
    sym_hct_N_2, eps = hct_mod(HQ, n_qubits//2, use_coeffs_eps=True, sym_metric_func=sym_metric_func)
    sym_hct_N, eps = hct_mod(HQ, n_qubits, use_coeffs_eps=True, sym_metric_func=sym_metric_func)

    #bliss cost function
    n_elec = molecule.n_electrons if system[:2] != 'N2' else molecule.n_electrons - 4 # 4 frozen for N2
    print(n_elec)
    print("Starting spin-restricted BLISS routine...")
    H_bliss, bliss_info = lp_bliss_paper_real_pauli_1norm(
        H,
        n_electrons=n_elec,
        n_orb=n_qubits,
    )
    print("Pauli BLISS completed, Relative Pauli L1 reduction: {}".format(bliss_info["relative_pauli_l1_reduction"]))
    HQ_bliss = jordan_wigner(H_bliss)
    sym_hct_bliss, eps_bliss = hct_mod(HQ_bliss, n_qubits, sym_metric_func= sym_metric_func, use_coeffs_eps=True)

    # seniority
    print("Seniority symmetries:")
    sym_sen = get_seniority_symmetries(n_qubits)
    print(sym_sen)

    #beam search
    print("\nBeam Search ({}) with exact-symmetry seeding:".format(bw))
    cost_function =  'Comm'
    print(f'Starting cost function: {cost_function}')
    sym_bs_N_2 = BeamSearch_Symmetries(
        HQ,
        target_rank=n_qubits//2,
        beam_width=bw,
        heavy_core_fraction=0.95,
        include_pairwise_products=True,
        pairwise_seed_terms=12,
        seed_with_exact_symmetries=True,
        score_func= cf_dict[cost_function], # this function maximizes the cost function TODO invert this
        include_hct_symmetries = True,
        hct_n_sym = n_qubits//2,
        hct_use_coeffs_eps = True,
    )

    sym_bs_N = BeamSearch_Symmetries(
        HQ,
        target_rank=n_qubits,
        beam_width=bw,
        heavy_core_fraction=0.95,
        include_pairwise_products=True,
        pairwise_seed_terms=12,
        seed_with_exact_symmetries=True,
        score_func= cf_dict[cost_function], # this function maximizes the cost function TODO invert this
        include_hct_symmetries = True,
        hct_n_sym = n_qubits,
        hct_use_coeffs_eps = True,
    )

    #benchmarks
    datasets = []
    verbose = True
    datasets.append(benchmark_syms(sym_bs_N_2, HQ, fci_gs, fci_e, 
                                    n_qubits, N_2_sym=True, print_to_file=output_filename, 
                                    tag="BS N/2" + f" {cost_function}", verbose = verbose, log_base=log_base))

    bs_N_data, bs_N_processed_data = benchmark_syms(sym_bs_N, HQ, fci_gs, fci_e, 
                                    n_qubits, N_2_sym=False, print_to_file=output_filename, 
                                    tag="BS N" + f" {cost_function}",verbose = verbose, return_processed_data=True, log_base=log_base)

    datasets.append(bs_N_data)

    #mpo bd:
    bs_N_mpo = bs_N_processed_data["mpo"]

    bs_N_mpo_bd = max(bs_N_mpo.bond_sizes())
    with open(output_filename, 'a') as f:
        print("\n Beam search N transformed MPO bd : {}".format(bs_N_mpo_bd), file=f)
    

    datasets.append(benchmark_syms(sym_hct_N_2, HQ, fci_gs, fci_e,
                                            n_qubits, N_2_sym=True, print_to_file=output_filename,
                                            tag="HCT N/2" + f" {cost_function}", verbose=verbose, log_base=log_base))

    hct_N_data, hct_N_processed_data = benchmark_syms(sym_hct_N, HQ, fci_gs, fci_e,
                                    n_qubits, N_2_sym=False, print_to_file=output_filename,
                                    tag="HCT N" + f" {cost_function}", verbose=verbose, return_processed_data=True, log_base=log_base)

    datasets.append(hct_N_data)
    
    hct_N_mpo = hct_N_processed_data["mpo"]
    hct_N_mpo_bd = max(hct_N_mpo.bond_sizes())
    with open(output_filename, 'a') as f:
        print("\n HCT N transformed MPO bd : {}".format(hct_N_mpo_bd), file=f)

    datasets.append(benchmark_syms(sym_sen, HQ, fci_gs, fci_e,
                                    n_qubits, N_2_sym=True, print_to_file=output_filename,
                                    tag="SEN N/2" + f" {cost_function}", verbose=verbose, log_base=log_base))

    datasets.append(benchmark_syms(sym_hct_bliss, HQ, fci_gs, fci_e,
                                    n_qubits, N_2_sym=False, print_to_file=output_filename,
                                    tag=r"Pauli BLISS+HCT($n_q$)", verbose=verbose, log_base=log_base))


    save_filename = output_filename + system + "_datasets"
    BenchmarkData.save_datasets(datasets, save_filename)

    #analysis
    #entropy graphs
    _ = BenchmarkData.plot_cut_entropies(datasets, fci_gs, output_filename + system + "_cutentropy.png")

    #dmrg bds
    # unrotated DMRG (as paulis) and entanglement
    from src.metrics import get_entropies_at_cuts
    ents_og = get_entropies_at_cuts(fci_gs, n_qubits, log_base=log_base)

    with open(output_filename, 'a') as f:
        
        print("\nOriginal fci_gs Entanglement :\n", file=f)
        for i, e in enumerate(ents_og):
            print("{}|{}: {}".format(i+1, i+2, e), file=f)

    # Fermionic MPO/MPS benchmarks are omitted because they require pyblock2.

    #qubit
    compress_cutoff = 1e-20
    import quimb.tensor as qtn
    from src.tn import find_dmrg_conv_bd_quimb
    gs_mps = qtn.MatrixProductState.from_dense(fci_gs, cutoff = compress_cutoff)     
    dmrg_bd, _, og_mpo_data = find_dmrg_conv_bd_quimb(HQ, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                        reps=1, verbose=False, compress_cutoff = compress_cutoff, sweep_tol = 1e-6,
                        noise = 1e0, bsz=2, guess_mps = gs_mps, seed=0, return_data=True)
    og_qubit_mpo = og_mpo_data["mpo"]
    og_mpo_bd = max(og_qubit_mpo.bond_sizes())
    with open(output_filename, 'a') as f:
        print("Original Qubit Hamiltonian MPO bd: ", og_mpo_bd, file=f)
        print("Original Qubit Hamiltonian mps bd for convergence: ", dmrg_bd, file=f)
    #fiedler reordering for symmetries
    #hct

    with open(output_filename, 'a') as f:
        print("Fiedler reordering for HCT N syms: ", file=f)
    hct_N_fiedler_info = do_fiedler_analysis(sym_hct_N, hct_N_processed_data["H_perm"], hct_N_processed_data["gs_rot"], fci_e, n_qubits, log_base=log_base, verbose=True, write_to_file=True, filename=output_filename)

    #bs
    with open(output_filename, 'a') as f:
        print("Fiedler reordering for Beam Search N syms: ", file=f)
    bs_N_fiedler_info = do_fiedler_analysis(sym_bs_N, bs_N_processed_data["H_perm"], bs_N_processed_data["gs_rot"], fci_e, n_qubits, log_base=log_base, verbose=True, write_to_file=True, filename=output_filename)

    #write fiedler info to file

    #add to csv
    cols = ["system", "Original Qubit"] + [data.tag for data in datasets] + ["HCT fiedler", "BS fiedler"]
    bd_rows.append(dict(zip(cols, [system, dmrg_bd] + [data.dmrg_bd for data in datasets] + [hct_N_fiedler_info["dmrg_bd"], bs_N_fiedler_info["dmrg_bd"]])))
    df = pd.DataFrame(bd_rows)
    df.to_csv(output_filename + "_dmrg_bd.csv", index=False)

    mpo_cols = ["system", "Original Qubit", "HCT N", "BS N", "HCT fiedler", "BS fiedler"]
    mpo_bd_rows.append(dict(zip(mpo_cols, [system, og_mpo_bd,
                                          hct_N_mpo_bd, bs_N_mpo_bd,
                                          hct_N_fiedler_info["mpo_bd"], bs_N_fiedler_info["mpo_bd"]])))
    pd.DataFrame(mpo_bd_rows).to_csv(output_filename + "_mpo_bd.csv", index=False)
