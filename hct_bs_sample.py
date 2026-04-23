### sample code to test HCT and beam search

from openfermion import count_qubits, jordan_wigner, QubitOperator, get_ground_state, get_sparse_operator, MolecularData, get_fermion_operator
from openfermionpyscf import run_pyscf
import numpy as np
from openfermion import FermionOperator
from copy import deepcopy
from openfermion import commutator
from src.state_utils import get_cisd_gs, get_fci_state_openfermion
from src.sym import get_quartic_symmetries, get_seniority_symmetries, find_approx_symm

from src.bs.utils import *
from src.bs.beam import *

# hct
from src.metrics import variance
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

# bl = 1.6
# geometry = [
#     ('Li', (0.0, 0.0, -bl/2)),
#     ('H', (0.0, 0.0, bl/2))
# ]

# bl = 2.0
# geometry = [
#     ('N', (0.0, 0.0, -bl/2)),
#     ('N', (0.0, 0.0, bl/2))
# ]

# n_H = 4
# H, molecule =  build_H_chain_for_R(1.0, n_H)

geometry = h2o_geometry(2.1, 104.5) 

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

# Get second-quantized electronic Hamiltonian and wavefunctions
H = get_fermion_operator(molecule.get_molecular_hamiltonian())
n_qubits = count_qubits(H)
HQ = jordan_wigner(H)
Hs = get_sparse_operator(HQ, n_qubits)

e, gs, gs_info = get_fci_state_openfermion(molecule)
gs = gs.toarray()
hf_occ = get_hf_occ(molecule.n_electrons, molecule.n_orbitals, as_str=True)
cisd_e, cisd_wfn= get_cisd_gs(hf_occ, HQ, n_qubits, 'wfs', tf='jw')


n_sym = n_qubits//2
sym_hct, eps = find_approx_symm(jordan_wigner(H), n_sym)#, sym_metric_func=lambda s: variance([s], gs, n_qubits))
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
    score_func=lambda s: -variance(s, cisd_wfn, n_qubits) # comm_sq_exp_fast(s, Hs, gs, n_qubits)# Change or remove as needed
)
for s in sym_bs:
    print("  ", s)

print("\nValidation:")
print(validate_symmetry_generators(HQ, sym_bs))

#evaluate non-commutativity for each symmetry
from src.metrics import universal_grading, entropy_pauli_syms, find_commuting_paulis

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