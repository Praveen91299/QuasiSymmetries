##script that was used to generate all molecular geometries and integrals

from openfermion import count_qubits, jordan_wigner, get_ground_state, get_sparse_operator, MolecularData, get_fermion_operator

from openfermionpyscf import run_pyscf
from quasisymmetries.state_utils import get_cisd_gs
from quasisymmetries.state_utils import get_hf_wfn, get_hf_occ
import pickle

# bl = 2.2
# geometry = [
#     ('N', (0.0, 0.0, -bl/2)),
#     ('N', (0.0, 0.0, bl/2))
# ]

def h4_rect_geometry(bla, blb):
    return [
        ('H', (-bla/2, -blb/2, 0.0)),
        ('H', (-bla/2, blb/2, 0.0)),
        ('H', (bla/2, -blb/2, 0.0)),
        ('H', (bla/2, blb/2, 0.0))
    ]

def h4_chain_geometry(bl):
    return [
        ('H', (-1.5*bl, 0.0, 0.0)),
        ('H', (-0.5*bl, 0.0, 0.0)),
        ('H', (0.5*bl, 0.0, 0.0)),
        ('H', (1.5*bl, 0.0, 0.0))
    ]

def lih_geometry(bl):
    return [
    ('Li', (0.0, 0.0, -bl/2)),
    ('H', (0.0, 0.0, bl/2))
]

basis = 'sto-3g'
multiplicity = 1  # singlet
charge = 0

# fci_es = []
# cisd_es = []
geometry = lih_geometry(4.0)
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
    run_fci=False
)

H = get_fermion_operator(molecule.get_molecular_hamiltonian())
n_qubits = count_qubits(H)
HQ = jordan_wigner(H)
Hs = get_sparse_operator(HQ, n_qubits)
fci_e, fci_gs = get_ground_state(Hs)
hf_occ = get_hf_occ(molecule.n_electrons, molecule.n_orbitals, as_str=True)
cisd_e, cisd_wfn= get_cisd_gs(hf_occ, HQ, n_qubits, 'wfs', tf='jw')

print(bl, fci_e, abs(cisd_e - fci_e))

#H, (fci_e, fci_wfn), (cisd_e, cisd_wfn)
data = (H, fci_e, fci_gs, cisd_e, cisd_wfn)
directory = "./saved/hamiltonians/"
filename = "LIH_corr"
with open(directory+filename+".pkl", "wb") as f:
    pickle.dump(data, f)



molecule.filename = directory + filename
molecule.save()
print(fci_e, cisd_e)

### to load:
"""
import pickle
from openfermion import MolecularData, jordan_wigner, count_qubits

directory = "./saved/hamiltonians/"

systems = [
    'H4chain_eqm',
    'H4chain_corr',
    'H4chain_diss',
    'H4rect_corr',
    'H4rect_diss',
    'LIH_eqm',
    'LIH_corr',
    'H2O_eqm',
    'H2O_corr',
    'H2O_diss',
    'N2frozen_eqm',
    'N2frozen_corr',
    'N2frozen_diss'
]

for system in systems:
    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    mol = MolecularData(filename="./saved/hamiltonians/H2O_corr")

    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    HQ = jordan_wigner(H)
    n_qubits = count_qubits(H)

"""