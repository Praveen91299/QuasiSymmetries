### diagonal vs non-diagonal symmetry test

from quasisymmetries.bs.beam import build_candidate_pool_hct
from quasisymmetries.bs.utils import exact_pauli_symmetry_basis, qubit_operator_terms, mask_to_qubit_operator

### May 26, testing new DMRG calculation stuff

from quasisymmetries.tn import find_dmrg_conv_bd_quimb

import pickle
import quimb.tensor as qtn
import numpy as np
from openfermion import count_qubits, jordan_wigner, MolecularData, get_sparse_operator
from quasisymmetries.state_utils import get_hf_wfn, get_hf_occ
from quasisymmetries.metrics import get_permuted_bipartite_entanglement, comm_sq_exp_fast
from quasisymmetries.sym import get_seniority_symmetries, hct_mod
from quasisymmetries.bliss import lp_bliss_paper_real_pauli_1norm

directory = "./saved/hamiltonians/"

#options
verbose=True

outfile = "./saved/n2_pool_test.txt"
system = "N2frozen_corr"

filename= system
with open(directory+system+".pkl", "rb") as f:
    data = pickle.load(f)
H, fci_e, fci_gs, cisd_e, cisd_gs = data
HQ = jordan_wigner(H)
molecule = MolecularData(filename=directory+system)
n_qubits = count_qubits(HQ)
Hs = get_sparse_operator(HQ, n_qubits)


# H_bliss, info = lp_bliss_paper_real_pauli_1norm(
#     H,
#     n_electrons=(molecule.n_electrons),
#     n_orb=(molecule.n_orbitals * 2),
# )
# print("Pauli BLISS completed, Relative Pauli L1 reduction: {}".format(info["relative_pauli_l1_reduction"]))
# HQ_bliss = jordan_wigner(H_bliss)

comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_gs, n_qubits)
# comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, fci_gs, n_qubits)
# var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)
# var_fci = lambda s_list: variance(s_list, fci_gs, n_qubits)

sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized

hamiltonian=HQ
beam_width = 16
max_candidates_from_terms = 256
hct_n_sym = n_qubits
hct_use_coeffs_eps: bool = True
include_pairwise_products = False
pairwise_seed_terms = 12
max_pauli_weight=None
max_exact_symmetry_seeds = None
score_func = sym_group_score_func

only_diagonal_terms = False

seed_generators = None

exact_syms = exact_pauli_symmetry_basis(hamiltonian, n_qubits=n_qubits)
if max_exact_symmetry_seeds is not None:
    exact_syms = exact_syms[:max_exact_symmetry_seeds]
seed_generators = exact_syms

#build candidate pool
n_qubits, terms = qubit_operator_terms(hamiltonian, n_qubits)

def filter_diagonal_terms(terms, verbose=True):
    #filter diagonal terms only
    
    terms_filtered = []
    for term in terms:
        x, z = term.mask
        if x == 0:
            terms_filtered.append(term)

    if verbose: print("{}/{} terms retained".format(len(terms_filtered), len(terms)))
    return terms_filtered

if only_diagonal_terms:
    terms = filter_diagonal_terms(terms, True)

candidate_pool = build_candidate_pool_hct(
    terms,
    n_qubits,
    max_candidates_from_terms=max_candidates_from_terms,
    include_pairwise_products=include_pairwise_products,
    pairwise_seed_terms=pairwise_seed_terms,
    max_pauli_weight=max_pauli_weight,
    include_hct_symmetries = False,
    hct_n_sym = hct_n_sym,
    hct_use_coeffs_eps = hct_use_coeffs_eps,
)

from quasisymmetries.bs.beam import beam_search_symmetries, local_swap_refine

target_rank=n_qubits
heavy_core_fraction=0.95
do_local_refine=True
local_refine_passes=10

syms = beam_search_symmetries(
    hamiltonian,
    candidate_pool,
    target_rank=target_rank,
    n_qubits=n_qubits,
    beam_width=beam_width,
    heavy_core_fraction=heavy_core_fraction,
    initial_generators=seed_generators,
    score_func=score_func
)
print(syms)

if do_local_refine:
    syms = local_swap_refine(
        hamiltonian,
        syms,
        candidate_pool,
        n_qubits=n_qubits,
        max_passes=local_refine_passes,
        score_func=score_func
    )

print(syms)

from quasisymmetries.benchmark import benchmark_syms

data_n2_og =  benchmark_syms(syms, HQ, fci_gs, fci_e, n_qubits, False, True, tag="N2 BS unmodified", print_to_file=outfile)

### diagonal only pool
terms_filtered = filter_diagonal_terms(terms, True)

candidate_pool = build_candidate_pool_hct(
    terms_filtered,
    n_qubits,
    max_candidates_from_terms=max_candidates_from_terms,
    include_pairwise_products=include_pairwise_products,
    pairwise_seed_terms=pairwise_seed_terms,
    max_pauli_weight=max_pauli_weight,
    include_hct_symmetries = False,
    hct_n_sym = hct_n_sym,
    hct_use_coeffs_eps = hct_use_coeffs_eps,
)

from quasisymmetries.bs.utils import mask_to_qubit_operator
for term in candidate_pool:
    print(mask_to_qubit_operator(term, n_qubits))

from quasisymmetries.bs.beam import beam_search_symmetries, local_swap_refine

target_rank=n_qubits
heavy_core_fraction=0.95
do_local_refine=True
local_refine_passes=10

syms = beam_search_symmetries(
    hamiltonian,
    candidate_pool,
    target_rank=target_rank,
    n_qubits=n_qubits,
    beam_width=beam_width,
    heavy_core_fraction=heavy_core_fraction,
    initial_generators=seed_generators,
    score_func=score_func
)

print(syms)

do_local_refine=True
if do_local_refine:
    syms = local_swap_refine(
        hamiltonian,
        syms,
        candidate_pool,
        n_qubits=n_qubits,
        max_passes=local_refine_passes,
        score_func=score_func
    )

print(syms)

from quasisymmetries.benchmark import benchmark_syms, BenchmarkData

data_n2_filtered =  benchmark_syms(syms, HQ, fci_gs, fci_e, n_qubits, False, True, tag="N2 BS diagonal", print_to_file=outfile)

datasets = [data_n2_og, data_n2_filtered]
BenchmarkData.save_datasets(datasets=datasets, filename='./saved/n2_og_fil_datasets')
