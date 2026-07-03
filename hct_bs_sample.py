### sample code to test HCT and beam search

from openfermion import count_qubits, jordan_wigner, QubitOperator, get_ground_state, get_sparse_operator, MolecularData, get_fermion_operator
from openfermionpyscf import run_pyscf
import numpy as np
from openfermion import FermionOperator
from copy import deepcopy
from openfermion import commutator
from src.state_utils import get_cisd_gs, get_fci_state_openfermion
from src.op_utils import build_H_chain_for_R
from src.sym import get_quartic_symmetries, get_seniority_symmetries, hct_mod

from src.bs.utils import *
from src.bs.beam import *

# hct
from src.metrics import variance, comm_sq_exp_fast, find_commuting_paulis, universal_grading
from src.state_utils import get_hf_occ, get_hf_wfn

def h2o_geometry(bond_length, bond_angle_deg):
    theta = np.deg2rad(bond_angle_deg)
    half = theta / 2.0

    geometry = [
        ('O', (0.0, 0.0, 0.0)),
        ('H', ( bond_length * np.sin(half), 0.0, bond_length * np.cos(half))),
        ('H', (-bond_length * np.sin(half), 0.0, bond_length * np.cos(half))),
    ]
    return geometry

bl = 1.6
geometry = [
    ('Li', (0.0, 0.0, -bl/2)),
    ('H', (0.0, 0.0, bl/2))
]

# bl = 2.0
# geometry = [
#     ('N', (0.0, 0.0, -bl/2)),
#     ('N', (0.0, 0.0, bl/2))
# ]

#geometry = h2o_geometry(2.1, 104.5) 

basis = 'sto-3g'
multiplicity = 1  # singlet
charge = 0

# Create molecule object
molecule = MolecularData(
    geometry=geometry,
    basis=basis,
    multiplicity=multiplicity,
    charge=charge
)

# Run PySCF to compute integrals (no need for correlated methods)
molecule = run_pyscf(
    molecule,
    run_scf=True,
    run_mp2=False,
    run_cisd=False,
    run_ccsd=False,
    run_fci=True
)

# n_H = 4
# H, molecule =  build_H_chain_for_R(1.0, n_H)

# Get second-quantized electronic Hamiltonian and wavefunctions
H = get_fermion_operator(molecule.get_molecular_hamiltonian())
n_qubits = count_qubits(H)
HQ = jordan_wigner(H)
Hs = get_sparse_operator(HQ, n_qubits)

e, gs, gs_info = get_fci_state_openfermion(molecule)
gs = gs.toarray()
hf_occ = get_hf_occ(molecule.n_electrons, molecule.n_orbitals, as_str=True)
cisd_e, cisd_wfn= get_cisd_gs(hf_occ, HQ, n_qubits, 'wfs', tf='jw')

#pick your favorite metric functions (to lower):
comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_wfn, n_qubits)
comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, gs, n_qubits)
var_cisd = lambda s_list: variance(s_list, cisd_wfn, n_qubits)
var_fci = lambda s_list: variance(s_list, gs, n_qubits)

sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized
sym_metric_func = lambda s: (-1)*sym_group_score_func([s]) # HCT minimized

n_sym = n_qubits//2
sym_hct, eps = hct_mod(HQ, n_sym, use_coeffs_eps=True, num_intervals=5000, sym_metric_func=sym_metric_func)
sym_sen = get_seniority_symmetries(n_qubits)
sym_quar = get_quartic_symmetries(n_qubits)

print("\nBeam Search with exact-symmetry seeding:")
sym_bs = find_commuting_symmetry_generators(
    HQ,
    target_rank=n_sym,
    beam_width=16,
    heavy_core_fraction=0.95,
    include_pairwise_products=True,
    pairwise_seed_terms=12,
    seed_with_exact_symmetries=True,
    score_func=sym_group_score_func,
    score_is_separable=True,
)
for s in sym_bs:
    print("  ", s)

# evaluate metrics
from src.metrics import universal_grading, entropy_pauli_syms, find_commuting_paulis, get_ent

print("Non-commutativity")
print("HCT:{}".format(universal_grading(sym_hct, HQ)))
print("Seniority:{}".format(universal_grading(sym_sen, HQ)))
print("Quartic:{}".format(universal_grading(sym_quar, HQ)))
print("BS:{}".format(universal_grading(sym_bs, HQ)))

print("Entropies:")
print("HCT: ", entropy_pauli_syms(sym_hct, gs, n_qubits))
print("Seniority: ", entropy_pauli_syms(sym_sen, gs, n_qubits))
print("Quartic: ", entropy_pauli_syms(sym_quar, gs, n_qubits))
print("BS: ", entropy_pauli_syms(sym_bs, gs, n_qubits))

print("HCT, SEN, QUARTIC, BS (non commuting terms)")
_ = find_commuting_paulis(HQ, sym_hct)
_ = find_commuting_paulis(HQ, sym_sen)
_ = find_commuting_paulis(HQ, sym_quar)
_ = find_commuting_paulis(HQ, sym_bs)

from src.tn import find_dmrg_conv_bd

max_bd=20
og_bd = find_dmrg_conv_bd(HQ, n_qubits, e, max_bd=max_bd, n_sweeps=50, reps=10)
print("DMRG bd for convergence for original Hamiltonian: ", og_bd)

# bi-partite entanglement
print("Sen")
sen_ent, H_perm_sen, gs_sen = get_ent(sym_sen, HQ, n_qubits, verbose=True, return_state=True)
sen_bd = find_dmrg_conv_bd(H_perm_sen, n_qubits, e, max_bd=max_bd, n_sweeps=50, reps=10)
print("DMRG bd for convergence: ", sen_bd)


print("HCT N/2 syms")
hct_ent, H_perm_hct, gs_hct = get_ent(sym_hct, HQ, n_qubits, verbose=True, return_state=True)
hct_bd = find_dmrg_conv_bd(H_perm_hct, n_qubits, e, max_bd=max_bd, n_sweeps=50, reps=10)
print("DMRG bd for convergence: ", hct_bd)

print("BS N/2 syms")
bs_ent, H_perm_bs, gs_bs = get_ent(sym_bs, HQ, n_qubits, verbose=True, return_state=True)
bs_bd = find_dmrg_conv_bd(H_perm_bs, n_qubits, e, max_bd=max_bd, n_sweeps=50, reps=10)
print("DMRG bd for convergence: ", bs_bd)
