import csv
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh
from openfermion import (
    MolecularData,
    get_fermion_operator,
    get_number_preserving_sparse_operator,
)
from pyblock2.algebra.io import MPSTools
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from quasisymmetries.mpo import build_qc_mpo_from_openfermion_molecule


def statevector_to_sz_determinants(
    state,
    n_spatial_orbitals,
    active_orbitals,
    core_orbitals=(),
    n_electrons=None,
    spin=None,
    cutoff=1e-12,
):
    """
    Convert an OpenFermion/Jordan-Wigner statevector to Block2 SZ determinant
    strings. Frozen-core orbitals are projected to doubly occupied, and any
    omitted non-core orbitals are projected to empty.
    """
    state = np.asarray(state).reshape(-1)
    n_qubits = 2 * n_spatial_orbitals
    if state.size != 2**n_qubits:
        raise ValueError(
            f"State has length {state.size}; expected {2**n_qubits}."
        )

    active_orbitals = list(active_orbitals)
    core_orbitals = set(core_orbitals)
    omitted_orbitals = (
        set(range(n_spatial_orbitals))
        - core_orbitals
        - set(active_orbitals)
    )
    occ_chars = {(0, 0): "0", (1, 0): "a", (0, 1): "b", (1, 1): "2"}
    dets, coeffs = [], []

    for basis_index in np.flatnonzero(np.abs(state) > cutoff):
        # OpenFermion's sparse basis places mode 0 at the most-significant bit.
        bits = [
            (int(basis_index) >> (n_qubits - 1 - mode)) & 1
            for mode in range(n_qubits)
        ]
        if any(
            (bits[2 * p], bits[2 * p + 1]) != (1, 1)
            for p in core_orbitals
        ):
            continue
        if any(
            (bits[2 * p], bits[2 * p + 1]) != (0, 0)
            for p in omitted_orbitals
        ):
            continue

        active_bits = [
            bits[2 * p + sigma]
            for p in active_orbitals
            for sigma in (0, 1)
        ]
        if n_electrons is not None and sum(active_bits) != n_electrons:
            continue
        active_spin = sum(active_bits[::2]) - sum(active_bits[1::2])
        if spin is not None and active_spin != spin:
            continue

        dets.append(
            "".join(
                occ_chars[(bits[2 * p], bits[2 * p + 1])]
                for p in active_orbitals
            )
        )
        coeffs.append(state[basis_index])

    if not coeffs:
        raise ValueError("The state has no weight in the requested active space.")

    coeffs = np.asarray(coeffs, dtype=complex)
    projected_norm = float(np.linalg.norm(coeffs))
    coeffs /= projected_norm

    # Remove the arbitrary eigensolver phase. These molecular states are real.
    phase_index = int(np.argmax(np.abs(coeffs)))
    coeffs *= np.exp(-1j * np.angle(coeffs[phase_index]))
    if np.linalg.norm(coeffs.imag) > 1e-8:
        # A real Hamiltonian can return a complex linear combination of
        # degenerate real eigenvectors. Its real and imaginary components are
        # independently valid eigenvectors; retain the component with more
        # weight so it can be imported into the real Block2 driver.
        real_norm = np.linalg.norm(coeffs.real)
        imag_norm = np.linalg.norm(coeffs.imag)
        coeffs = coeffs.real if real_norm >= imag_norm else coeffs.imag
        coeffs /= np.linalg.norm(coeffs)
    else:
        coeffs = coeffs.real

    return dets, coeffs, projected_norm


def determinants_to_su2_pyblock_mps(
    dets,
    coeffs,
    n_sites,
    n_electrons,
    spin,
    orb_sym,
    stack_mem=int(2 * 1024**3),
):
    """
    Import determinant coefficients in Block2 SZ mode, then use Block2's
    Clebsch-Gordan transformation to obtain a spin-adapted SU2 pyblock MPS.

    A separate temporary driver is used because changing Block2 symmetry types
    changes its global frame. The returned Python tensor representation remains
    valid after the temporary SZ driver is destroyed.
    """
    with tempfile.TemporaryDirectory(prefix="pyblock2_fci_sz_") as scratch:
        sz_driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SZ,
            n_threads=1,
            stack_mem=stack_mem,
        )
        sz_driver.initialize_system(
            n_sites=n_sites,
            n_elec=n_electrons,
            spin=spin,
            orb_sym=orb_sym,
        )
        sz_mps = sz_driver.get_mps_from_csf_coefficients(
            dets,
            coeffs,
            tag="FCI-SZ",
            dot=1,
            full_fci=False,
            iprint=0,
        )
        sz_driver.align_mps_center(sz_mps, 0)
        py_sz_mps = MPSTools.from_block2(sz_mps)
        sz_basis = [
            Counter(
                {
                    basis.quanta[j]: basis.n_states[j]
                    for j in range(basis.n)
                }
            )
            for basis in sz_driver.ghamil.basis
        ]

        return MPSTools.trans_sz_to_su2(
            py_sz_mps,
            sz_basis,
            sz_driver.target,
            target_twos=spin,
            cutoff=1e-13,
        )


def pyblock_su2_to_block2_mps(py_su2_mps, driver, tag):
    """Convert a Python SU2 tensor MPS into the active SU2 Block2 driver."""
    su2_basis = [
        Counter(
            {
                basis.quanta[j]: basis.n_states[j]
                for j in range(basis.n)
            }
        )
        for basis in driver.ghamil.basis
    ]
    mps = MPSTools.to_block2(
        py_su2_mps,
        su2_basis,
        center=0,
        tag=tag,
        left_vacuum=driver.left_vacuum,
    )
    mps.info.save_data(f"{driver.scratch}/{tag}-mps_info.bin")
    return driver.adjust_mps(mps, dot=2)[0]


SYSTEMS = [
    "H4chain_eqm",
    "H4chain_corr",
    "H4chain_diss",
    "H4rect_corr",
    "H4rect_diss",
    "LIH_eqm",
    "LIH_corr",
    "H2O_eqm",
    "H2O_corr",
    "H2O_diss",
    "N2frozen_eqm",
    "N2frozen_corr",
    "N2frozen_diss",
]

HAMILTONIAN_DIRECTORY = Path("./saved/hamiltonians")
OUTPUT_CSV = Path("./saved/results/fermionic_mpo_mps_bond_dimensions_test.csv")
BOND_DIMS = (
    list(range(1, 11))
    + list(range(12, 21, 2))
    + list(range(30, 101, 10))
)
N_SWEEPS = 50
ENERGY_TOLERANCE = 1.6e-3


def benchmark_system(system):
    """Find the lowest tested converged MPS bond dimension for one system."""
    print("\n" + "=" * 72)
    print("System:", system)
    print("=" * 72)

    filename = HAMILTONIAN_DIRECTORY / system
    with open(f"{filename}.pkl", "rb") as f:
        _, saved_fci_energy, fci_gs, _, _ = pickle.load(f)
    molecule = MolecularData(filename=str(filename))

    ncore = 2 if system.startswith("N2") else 0
    active_orbitals = list(range(ncore, molecule.n_orbitals))
    n_elec_active = molecule.n_electrons - 2 * ncore
    spin_target = molecule.multiplicity - 1

    if fci_gs.size == 2 ** (2 * len(active_orbitals)):
        fci_dets, fci_coeffs, projection_norm = (
            statevector_to_sz_determinants(
                fci_gs,
                n_spatial_orbitals=len(active_orbitals),
                active_orbitals=range(len(active_orbitals)),
                n_electrons=n_elec_active,
                spin=spin_target,
            )
        )
    elif fci_gs.size == 2 ** (2 * molecule.n_orbitals):
        fci_dets, fci_coeffs, projection_norm = (
            statevector_to_sz_determinants(
                fci_gs,
                n_spatial_orbitals=molecule.n_orbitals,
                active_orbitals=active_orbitals,
                core_orbitals=range(ncore),
                n_electrons=n_elec_active,
                spin=spin_target,
            )
        )
    else:
        raise ValueError(
            "FCI vector size matches neither the full nor active orbital space."
        )

    print("Saved FCI energy:", saved_fci_energy)
    print("FCI determinants retained:", len(fci_dets))
    print("FCI active-space projection norm:", projection_norm)

    py_su2_fci_mps = determinants_to_su2_pyblock_mps(
        fci_dets,
        fci_coeffs,
        n_sites=len(active_orbitals),
        n_electrons=n_elec_active,
        spin=spin_target,
        orb_sym=[1] * len(active_orbitals),
    )

    out = build_qc_mpo_from_openfermion_molecule(
        molecule,
        ncore=ncore,
        symm_type=SymmetryTypes.SU2,
        iprint=0,
    )
    driver = out["driver"]
    mpo = out["mpo"]

    active_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=out["core_orbitals"],
        active_indices=out["active_orbitals"],
    )
    active_sparse = get_number_preserving_sparse_operator(
        get_fermion_operator(active_hamiltonian),
        num_qubits=2 * out["ncas"],
        num_electrons=out["n_elec"],
    )
    reference_energy = float(eigsh(active_sparse, k=1, which="SA")[0][0])
    print("Matching active-space FCI energy:", reference_energy)

    fci_mps = pyblock_su2_to_block2_mps(
        py_su2_fci_mps,
        driver,
        tag=f"FCI-SU2-{system}",
    )
    identity_mpo = driver.get_identity_mpo()
    exact_norm = driver.expectation(fci_mps, identity_mpo, fci_mps)
    exact_energy = driver.expectation(fci_mps, mpo, fci_mps) / exact_norm
    if abs(exact_energy - reference_energy) > 1e-8:
        raise RuntimeError(
            f"SU2 FCI MPS energy check failed: "
            f"{exact_energy} versus {reference_energy}."
        )

    converged_bd = None
    dmrg_energy = None
    absolute_error = None
    noises = [0.0] * N_SWEEPS
    thrds = [1e-10] * N_SWEEPS

    for bd in BOND_DIMS:
        print(f"\nTesting MPS bond dimension {bd}")
        ket = driver.copy_mps(fci_mps, tag=f"KET-{system}-BD{bd}")
        dmrg_energy = float(
            driver.dmrg(
                mpo,
                ket,
                n_sweeps=N_SWEEPS,
                bond_dims=[bd] * N_SWEEPS,
                noises=noises,
                thrds=thrds,
                dav_max_iter=50,
                iprint=1,
            )
        )
        absolute_error = abs(dmrg_energy - reference_energy)
        converged = absolute_error <= ENERGY_TOLERANCE
        print(
            f"{system}: BD={bd}, energy={dmrg_energy:.12f}, "
            f"error={absolute_error:.3e}, converged={converged}"
        )
        if converged:
            converged_bd = bd
            break

    row = {
        "system": system,
        "ncore": ncore,
        "active_electrons": out["n_elec"],
        "active_orbitals": out["ncas"],
        "mpo_bond_dimension": out["largest_mpo_bond_dim"],
        "mps_converged_bond_dimension": (
            converged_bd if converged_bd is not None else ""
        ),
        "converged": converged_bd is not None,
        "reference_energy": reference_energy,
        "dmrg_energy": dmrg_energy,
        "absolute_error": absolute_error,
        "energy_tolerance": ENERGY_TOLERANCE,
    }

    scratch_obj = out.get("_scratch_obj")
    if scratch_obj is not None:
        scratch_obj.cleanup()
    return row


def write_results(rows):
    """Save after every system so completed results survive interruptions."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "ncore",
        "active_electrons",
        "active_orbitals",
        "mpo_bond_dimension",
        "mps_converged_bond_dimension",
        "converged",
        "reference_energy",
        "dmrg_energy",
        "absolute_error",
        "energy_tolerance",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    existing = {}
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            existing = {
                row["system"]: row for row in csv.DictReader(f)
            }

    rows = []
    for system in SYSTEMS:
        old_row = existing.get(system)
        if old_row is not None and old_row.get("converged") == "True":
            print(f"{system}: already converged in {OUTPUT_CSV}; skipping.")
            rows.append(old_row)
            write_results(rows)
            continue

        try:
            row = benchmark_system(system)
        except Exception as exc:
            print(f"{system}: FAILED: {exc}")
            row = {
                "system": system,
                "converged": False,
                "energy_tolerance": ENERGY_TOLERANCE,
            }
        rows.append(row)
        write_results(rows)
        print("Saved results to:", OUTPUT_CSV)

    print("\nCompleted all systems. Results:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
