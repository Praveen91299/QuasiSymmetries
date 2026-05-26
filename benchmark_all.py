# %%
# %%
from __future__ import annotations
# benchmark scripts
from openfermion import count_qubits, jordan_wigner, QubitOperator, get_ground_state, get_sparse_operator, MolecularData, get_fermion_operator
import pickle
import numpy as np
from openfermion import FermionOperator
from copy import deepcopy
from openfermion import commutator, QubitOperator
#from src.state_utils import get_cisd_gs, get_fci_state_openfermion
#from src.op_utils import build_H_chain_for_R, h2o_geometry
from src.sym import get_quartic_symmetries, get_seniority_symmetries, hct_mod, bs_hct

from src.bs.utils import *
from src.bs.beam import *

# hct
#from src.state_utils import get_hf_occ, get_hf_wfn

from src.metrics import *
from src.tn import find_dmrg_conv_bd_quimb
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class BenchmarkData:
    tag: str = ''
    symmetries: list[QubitOperator] = field(default_factory=list)
    non_commuting_l1: float = 0
    num_commuting_terms: int = 0
    sym_entropy: float = 0
    cut_entropies: list[float] = field(default_factory=list)
    dmrg_bd: int = 0
    single_sector_e: float = 0

    def write_to_file(self, filename):
        ent_str = "\n".join([val.__str__() for val in self.cut_entropies])
        with open(filename, 'a') as f:
            print(self.tag, file=f)
            print("Symmetries:", file=f)
            print("\n".join([sym.__str__() for sym in self.symmetries]), file=f)
            print("Non-commutator L1: ", self.non_commuting_l1, file=f)
            print("Entropy: ", self.sym_entropy, file=f)
            print("Commuting terms: ", self.num_commuting_terms, file=f)
            print("Cut entropies:\n", ent_str, file=f)
            print("DMRG conv BD: ", self.dmrg_bd, file=f)
            print("Single sector energy: ", self.single_sector_e, file=f)
    
    def save(self, filename):
        """
        Save as pickle object

        """
        with open(filename, 'wb') as f:
            pickle.dump(self, file=f)
    
    @classmethod
    def save_datasets(cls, datasets: list[BenchmarkData], filename):
        """
        Save datasets into single file as a dictionary
        """
        data_dict = {}

        for data in datasets:
            data_dict[data.tag] = data

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(data_dict, f)


    @classmethod
    def plot_cut_entropies(cls, datasets: list[BenchmarkData], gs=None, filename: str = None):
        """
        Save and return plot for entropies.
        """
        fig, ax = plt.subplots()

        for data in datasets:
            n_qubits = len(data.cut_entropies) + 1
            x = range(1, n_qubits)
            ax.plot(x, data.cut_entropies, label=data.tag)
        
        if gs is not None:
            #reference state
            n_qubits = int(np.log2(len(gs)))
            gs_ent = get_entropies_at_cuts(gs, n_qubits)
            x = range(1, n_qubits)
            ax.plot(x, gs_ent, label="Reference")

        ax.legend()
        ax.set_xlabel("Bond index")
        ax.set_ylabel("Entropy (bits)")

        if filename is not None: fig.savefig(filename, dpi=300, bbox_inches="tight")

        return fig

def benchmark_syms(list_syms, HQ, fci, fci_e, n_qubits, N_2_sym=False, verbose=True, print_to_file=None, tag=""):
    """
    Run all benchmarks for symmetries

    """
    print(tag)
    nc_l1 = universal_grading(list_syms, HQ, verbose=verbose)
    c = len(find_commuting_paulis(HQ, list_syms, verbose=verbose))
    ent, H_perm, gs_rot = get_ent(list_syms, HQ, n_qubits, verbose=verbose, return_state=True)
    dmrg_bd = find_dmrg_conv_bd(H_perm, n_qubits, fci_e, max_bd=30, tol=1.6e-3, n_sweeps=50, reps=20, verbose=verbose)    

    #ent and dmrg
    if N_2_sym:
        ent_N_2 = entropy_pauli_syms(list_syms, fci, n_qubits, verbose=verbose)
        ss_energies = get_single_sector_energies(HQ, list_syms, n_qubits, verbose=verbose)
        ss_e = np.min(ss_energies)
        #N/2 syms, single sector, BO energies TODO K and BO energies, but they are not really relevant here

        data = BenchmarkData(tag=tag, symmetries=list_syms, non_commuting_l1 = nc_l1, num_commuting_terms=c,  sym_entropy=ent_N_2, cut_entropies=ent, dmrg_bd=dmrg_bd, single_sector_e=ss_e)
    else:
        data = BenchmarkData(tag=tag, symmetries=list_syms, non_commuting_l1 = nc_l1, num_commuting_terms=c, cut_entropies=ent, dmrg_bd=dmrg_bd)
    
    if print_to_file is not None:
        data.write_to_file(print_to_file)

    return data

directory = "./saved/hamiltonians/"

systems = [
    'H4chain_eqm']
'''
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
'''
bd_rows = []

for system in systems:
    print(f'Starting system: {system}')
    filename= system

    date="_MAY18" #to keep track of outputs
    cost_func_tag = '_nc_exp_cisd'
    output_filename = "./saved/" + cost_func_tag + date
    with open(output_filename, 'a') as f:
        print(system, file=f)
    
    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    molecule = MolecularData(filename=directory+system)
    HQ = jordan_wigner(H)
    n_qubits = count_qubits(HQ)
    Hs = get_sparse_operator(HQ, n_qubits)

    #state specific cost functions
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
    n_sym = n_qubits//2
    #sym_hct_N_2, eps = hct_mod(HQ, n_sym, use_coeffs_eps=True, sym_metric_func=sym_metric_func)
    #sym_hct_N, eps = hct_mod(HQ, n_qubits, use_coeffs_eps=True, sym_metric_func=sym_metric_func)
    #sym_sen = get_seniority_symmetries(n_qubits)
    #sym_quar = get_quartic_symmetries(n_qubits)[:n_sym-1]
    #bs_hct_N_2 = bs_hct(HQ, n_sym, beam_width=bw, sym_metric_func=sym_metric_func, use_coeffs_eps=True)
    #sym_bs_hct_N_2 = bs_hct_N_2[0].syms
    #bs_hct_N = bs_hct(HQ, n_qubits, beam_width=bw, sym_metric_func=sym_metric_func, use_coeffs_eps=True)
    #sym_bs_hct_N = bs_hct_N[0].syms


    print("\nBeam Search ({}) with exact-symmetry seeding:".format(bw))

    sym_bs_N_2_list = []
    sym_bs_N_list = []
    datasets = []

    for cost_function in ['Comm', 'Var', '1-norm']:

        print(f'Starting cosf function: {cost_function}')

        sym_bs_N_2 = find_commuting_symmetry_generators(
            HQ,
            target_rank=n_sym,
            beam_width=bw,
            heavy_core_fraction=0.95,
            include_pairwise_products=True,
            pairwise_seed_terms=12,
            seed_with_exact_symmetries=True,
            score_func= cf_dict[cost_function] # this function maximizes the cost function TODO invert this
        )

        sym_bs_N = find_commuting_symmetry_generators(
            HQ,
            target_rank=n_qubits,
            beam_width=bw,
            heavy_core_fraction=0.95,
            include_pairwise_products=True,
            pairwise_seed_terms=12,
            seed_with_exact_symmetries=True,
            score_func= cf_dict[cost_function] # this function maximizes the cost function TODO invert this
        )

        sym_bs_N_2_list.append(sym_bs_N_2)
        sym_bs_N_list.append(sym_bs_N)

        datasets.append(benchmark_syms(sym_bs_N_2, HQ, fci_gs, fci_e, 
                                       n_qubits, N_2_sym=True, print_to_file=output_filename, 
                                       tag="BS N/2" + f" {cost_function}"))
        
        datasets.append(benchmark_syms(sym_bs_N_2, HQ, fci_gs, fci_e, 
                                       n_qubits, N_2_sym=True, print_to_file=output_filename, 
                                       tag="BS N" + f" {cost_function}"))

    #diagonostics
    #to do k for BO energy within chemical accuracy
    #data_hct_N_2 = benchmark_syms(sym_hct_N_2, HQ, fci_gs, fci_e, n_qubits, N_2_sym=True, print_to_file=output_filename, tag="HCT N/2")
    #data_hct_N = benchmark_syms(sym_hct_N, HQ, fci_gs, fci_e, n_qubits, N_2_sym=False, print_to_file=output_filename, tag="HCT N")
    #data_sen = benchmark_syms(sym_sen, HQ, fci_gs, fci_e, n_qubits, N_2_sym=True, print_to_file=output_filename, tag="SEN")
    #data_quar = benchmark_syms(sym_quar, HQ, fci_gs, fci_e, n_qubits, N_2_sym=True, print_to_file=output_filename, tag="QUAR")
    #data_bs_N_2 = benchmark_syms(sym_bs_N_2, HQ, fci_gs, fci_e, n_qubits, N_2_sym=True, print_to_file=output_filename, tag="BS({}) N/2".format(bw))
    #data_bs_N = benchmark_syms(sym_bs_N, HQ, fci_gs, fci_e, n_qubits, N_2_sym=False, print_to_file=output_filename, tag="BS({}) N".format(bw))
    #data_bs_hct_N_2 = benchmark_syms(sym_bs_hct_N_2, HQ, fci_gs, fci_e, n_qubits, N_2_sym=True, print_to_file=output_filename, tag="BS-HCT({}) N/2".format(bw))
    #data_bs_hct_N = benchmark_syms(sym_bs_hct_N, HQ, fci_gs, fci_e, n_qubits, N_2_sym=False, print_to_file=output_filename, tag="BS-HCT({}) N".format(bw))
    
    #datasets = [data_hct_N_2, data_hct_N, data_sen, data_quar, data_bs_N_2, data_bs_N, data_bs_hct_N_2, data_bs_hct_N]

    #save data objects
    save_filename = output_filename + system + "_datasets"
    BenchmarkData.save_datasets(datasets, save_filename)
    
    #analysis
    #entropy graphs
    _ = BenchmarkData.plot_cut_entropies(datasets, fci_gs, output_filename + system + "_cutentropy.png")

    #dmrg bds
    cols = ["system"] + [data.tag for data in datasets]
    bd_rows.append(dict(zip(cols, [system] + [data.dmrg_bd for data in datasets])))
    df = pd.DataFrame(bd_rows)
    df.to_csv(output_filename + "_dmrg_bd.csv", index=False)
# %%
