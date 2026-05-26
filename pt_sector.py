from __future__ import annotations
# benchmark scripts

from openfermion import count_qubits, jordan_wigner, get_ground_state, get_sparse_operator, MolecularData
import pickle

import numpy as np
from src.sym import get_seniority_symmetries, hct_mod

from src.bs.utils import *
from src.bs.beam import *
import pprint

# hct
from src.state_utils import get_hf_occ

from src.metrics import *
import matplotlib.pyplot as plt
import pickle
from src.metrics import *
from src.pt import computational_basis_matrix_element, coupled_computational_basis_states

directory = './saved/hamiltonians/'
system = 'H4chain_corr'

with open(directory+system+".pkl", "rb") as f:
    data = pickle.load(f)
H, fci_e, fci_gs, cisd_e, cisd_gs = data
molecule = MolecularData(filename=directory+system)
HQ = jordan_wigner(H)
n_qubits = count_qubits(HQ)
Hs = get_sparse_operator(HQ, n_qubits)

from src.op_utils import split_diagonal_paulis

def separate_H(HQ, list_syms, verbose=False, verify=True):
    const = HQ.constant
    HQ_mod = HQ - const
    HQ_mod.compress()

    H0 = sum(find_commuting_paulis(HQ_mod, list_syms, verbose)) + const
    V = HQ - H0

    #diagonal and non-diagonal commuting

    Z0, V0 = split_diagonal_paulis(H0)

    return Z0, V0, V

def get_computational_basis_symmetry_sector(basis_state, list_sym):
    """
    Returns tuple of +-1 for symmetry sector information
    """
    return tuple([round(computational_basis_matrix_element(basis_state, sym, basis_state).real) for sym in list_sym])

comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_gs, n_qubits)
comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, fci_gs, n_qubits)
var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)
var_fci = lambda s_list: variance(s_list, fci_gs, n_qubits)

sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized
sym_metric_func = lambda s: (-1)*sym_group_score_func([s]) # HCT minimized

sen_sym= get_seniority_symmetries(n_qubits)

hct_sym, _ = hct_mod(HQ, n_qubits//2, sym_metric_func=sym_metric_func, use_coeffs_eps=True)
list_sym = hct_sym

Z0, V0, V = separate_H(HQ, list_sym)
ref_hf = get_hf_occ(molecule.n_electrons, molecule.n_orbitals)
#ref_hf = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

ref_sec = get_computational_basis_symmetry_sector(ref_hf, list_sym)
print("Reference symmetry sector: ", ref_sec)



cb = coupled_computational_basis_states(V, ref_hf)

E0 = computational_basis_matrix_element(ref_hf, Z0, ref_hf)
print("Reference energy: ", E0)
Ei_dict = {}
eps_dict = {}
sec_dict = {}

eps_thresh = 0
lower_Ei_basis = {}

for b in cb:
    Ei = computational_basis_matrix_element(b, Z0, b)
    Ei_dict[b] = Ei
    if np.isclose(Ei, E0):
        print("Warning degeneracy betweem ref and ", b)
    Vi0 = computational_basis_matrix_element(b, V, ref_hf)
    eps = np.abs(Vi0)**2 / (E0 - Ei)
    sec = get_computational_basis_symmetry_sector(b, list_sym)

    #print(b, sec, Ei, Ei - E0, Vi0, eps)
    eps_dict[b] = eps
    #symmetry sector information
    #if sec == (1,-1, 1, 1, 1, -1) or sec == (-1, -1, 1, 1, 1, 1):#(1, 1, 1, -1, -1, -1, -1) or sec == (1, 1, 1, 1, 1, 1, 1):
    #    print(b, Ei, Vi0, eps, sec)
    if Ei < E0 or eps >0:
        lower_Ei_basis[b] = Ei
        print("Warning: \nSector {}\nState: {}\nEi: {}\nVi0: {}\neps: {}".format(sec, b, Ei, Vi0, eps))
    if sec not in sec_dict:
        sec_dict[sec] = np.abs(eps)
    else:
        sec_dict[sec] += np.abs(eps)

print(sec_dict)
#filter:
sec_dict_filtered = {sec: val for sec, val in zip(sec_dict.keys(), sec_dict.values()) if np.abs(val) > eps_thresh}
print("Filtered: ")
pprint.pprint(sec_dict_filtered)

Hs = get_sparse_operator(HQ, n_qubits)
sec_filtered = list(sec_dict_filtered.keys())

ref_proj = get_sector_projectors(list_sym, [ref_sec], n_qubits)[0]
filtered_projs = get_sector_projectors(list_sym, sec_filtered, n_qubits)

e, wf = get_ground_state(ref_proj@ Hs @ref_proj)
print(e)

diffs = []
pred_diff = []
for sec, proj in zip(sec_filtered, filtered_projs):
    #find energy of extended space
    print("\n", sec)
    ext_proj = proj + ref_proj
    e_ext, wfn = get_ground_state(ext_proj@ Hs @ext_proj)
    print(e_ext, e)
    diff = e_ext - e
    diffs.append(diff)
    pred_diff.append(sec_dict_filtered[sec])
    print(pred_diff[-1], diff)

#all together
proj_joint = sum(filtered_projs) + ref_proj

e_ext, wfn = get_ground_state(proj_joint@ Hs @proj_joint)
print(e_ext)
print(fci_e)
print(e_ext - fci_e)

import matplotlib.pyplot as plt

plt.plot(np.abs(pred_diff), np.abs(diffs), '*')
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-12, 1])
plt.xlim([1e-12, 1])
plt.xlabel(r"Predicted sector contribution $I_{\vec s}$ (Ha)")
plt.ylabel(r"Energy reduction on sector inclusion (Ha)")

#plt.savefig("./saved/sec_selection_PT_h2o_corr_sen.png", dpi=300)


### todo
# n2 figure out missing sector when implementing upto pt2
# implement multi-reference search
# implement pt3 and perhaps a recursive search method

# def int_to_binary_tuple(n, length):
#     return tuple(int(bit) for bit in format(n, f"0{length}b"))

# ps = construct_projectors_sparse([get_sparse_operator(sym, n_qubits) for sym in sen_sym], n_qubits)
# exps = {int_to_binary_tuple(i, len(sen_sym)): expectation(proj, fci_gs) for i, proj in enumerate(ps)}

# for sec, exp in exps.items():
#     if np.abs(exp) >= 1e-5:
#         print(sec, np.abs(exp), sum(sec))
