#!/usr/bin/env python3
"""Benchmark several STO-3G orbital bases with optional Fiedler ordering.

Supported systems are full-space singlet N2 at R=1.4 Angstrom and linear H6
or H8 chains with a 2.0 Angstrom nearest-neighbor spacing. Four spatial-
orbital bases are available:

1. canonical RHF molecular orbitals;
2. Pipek--Mezey localization performed separately in the occupied and virtual
   spaces; and
3. Pipek--Mezey localization performed separately in every intersection of
   occupied/virtual space and D2h Abelian irrep; and
4. natural orbitals obtained by diagonalizing the canonical CISD spatial
   one-particle density matrix.

For each basis the script runs raw fermionic SU(2), raw Jordan--Wigner qubit,
full-rank HCT, and full-rank Beam-search DMRG curves. HCT and Beam candidates
are ranked by the CISD squared-commutator expectation. A bond dimension is
reported as converged only if it is within chemical accuracy of FCI and the
DMRG sweep-energy stopping criterion was also satisfied.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from openfermion import MolecularData, get_fermion_operator
from pyscf import ao2mo, fci, gto, lo, scf, symm

import _bootstrap  # noqa: F401

from quasisymmetries.block2_qubit_benchmark import (
    run_block2_qubit_dmrg_curve,
)
from quasisymmetries.bs.beam import (
    beam_search_symmetries,
    build_candidate_pool,
    validate_symmetry_generators,
)
from quasisymmetries.bs.utils import (
    jordan_wigner_pauli_stream,
    qubit_operator_terms,
    qubitops_to_masks,
)
from quasisymmetries.chemistry import run_restricted_cisd_from_molecular_data
from quasisymmetries.clifford_symmetry_optimized import (
    Clifford,
    permute_qubits_in_qubit_operator,
)
from quasisymmetries.dmrg_search import (
    lowest_accepted_bond_dimension,
    next_binary_bond_dimension,
)
from quasisymmetries.fiedler import (
    fiedler_order_from_state,
    fiedler_order_spatial_orbitals_from_sparse_state,
)
from quasisymmetries.metrics import GroupedSparsePauliCommutatorEvaluator
from quasisymmetries.mps_unitary import PermutationUnitary
from quasisymmetries.save import (
    decode_qubit_operator,
    encode_qubit_operator,
    load_json,
    read_csv,
    save_json,
    save_pauli_term_stream,
    save_sparse_qubit_state,
    write_csv,
)
from quasisymmetries.sym import HCT

import benchmark_raw_n2_dmrg as fermionic_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "saved" / "results"
CHEMICAL_ACCURACY_HARTREE = 1.6e-3
BOND_SEARCH_METHOD = "midpoint_integer_binary_v2"
DEFAULT_WARM_START_NOISES = (1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6)
ORBITAL_BASES = ("canonical", "split_pm", "split_pm_irrep", "natural")
ORDERINGS = ("standard", "fiedler")
FRAMES = ("raw_fermionic_su2", "raw_qubit", "HCT", "Beam")
SYSTEM_DEFAULTS = {
    "N2_corr": {"molecule": "N2", "bond_length_angstrom": 1.4},
    "H6_corr": {"molecule": "H6", "bond_length_angstrom": 2.0},
    "H8_corr": {"molecule": "H8", "bond_length_angstrom": 2.0},
}


class Tee:
    """Write Python text output to both a terminal stream and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return False


def load_dmrg_curve_rows(csv_path: Path, json_path: Path) -> list[dict]:
    """Load checkpointed DMRG rows with scientific scalar types restored.

    Parameters
    ----------
    csv_path, json_path
        Parallel human-readable CSV and typed JSON checkpoint paths. JSON is
        preferred. CSV provides compatibility with results written before the
        typed checkpoint was introduced.

    Returns
    -------
    rows
        DMRG records with bond dimensions represented as integers and Boolean
        convergence fields represented as actual Booleans, as required by the
        binary-search selector.
    """
    if json_path.is_file():
        return list(load_json(json_path))
    if not csv_path.is_file():
        return []
    rows = read_csv(csv_path)
    boolean_fields = (
        "within_dmrg_tolerance",
        "accepted_converged_bond_dimension",
        "sweep_converged",
        "reached_max_sweeps",
        "sweep_limit_reached_without_convergence",
    )
    for row in rows:
        row["bond_dim"] = int(row["bond_dim"])
        for field in boolean_fields:
            if row.get(field) in {"True", "False"}:
                row[field] = row[field] == "True"
            elif row.get(field) in {"", "None", None}:
                row[field] = None
    return rows


def save_dmrg_curve_rows(
    rows: list[dict], csv_path: Path, json_path: Path
) -> None:
    """Checkpoint typed DMRG rows to JSON and a readable copy to CSV."""
    save_json(json_path, rows)
    write_csv(csv_path, rows)


def molecular_geometry(system: str, bond_length_angstrom: float):
    """Return a centered linear molecular geometry in Angstrom.

    Parameters
    ----------
    system
        ``"N2"`` for the diatomic, or ``"H2"``, ``"H6"``, or ``"H8"``
        for an equally spaced linear hydrogen chain.
    bond_length_angstrom
        N--N distance for N2 or nearest-neighbor H--H spacing for a chain.

    Returns
    -------
    geometry
        PySCF/OpenFermion atom specification centered on the z axis.
    """
    if system == "N2":
        number_of_atoms = 2
        atom = "N"
    elif system in {"H2", "H6", "H8"}:
        number_of_atoms = int(system[1:])
        atom = "H"
    else:
        raise ValueError(f"unsupported molecular system {system!r}")
    spacing = float(bond_length_angstrom)
    center = 0.5 * (number_of_atoms - 1)
    return [
        (atom, (0.0, 0.0, (atom_index - center) * spacing))
        for atom_index in range(number_of_atoms)
    ]


def molecular_geometry_from_pyscf(molecule) -> list[tuple[str, tuple[float, ...]]]:
    """Return the PySCF molecule's actual geometry in Angstrom.

    Parameters
    ----------
    molecule
        PySCF ``Mole`` object, whose internal coordinates are in Bohr.

    Returns
    -------
    geometry
        OpenFermion-compatible atom symbols and Cartesian coordinates in
        Angstrom. This preserves every atom of H6 and H8 chains.
    """
    bohr_to_angstrom = 0.529177210903
    return [
        (
            molecule.atom_symbol(atom_index),
            tuple(
                float(value) * bohr_to_angstrom
                for value in molecule.atom_coord(atom_index)
            ),
        )
        for atom_index in range(molecule.natm)
    ]


def run_canonical_rhf(system: str, bond_length_angstrom: float):
    """Run the symmetry-adapted canonical RHF calculation.

    The linear molecule is represented in the Abelian D2h subgroup so orbital
    irrep labels can be used to define the restricted localization blocks.

    Returns
    -------
    molecule, mean_field
        Converged PySCF molecule and RHF objects.
    """
    molecule = gto.M(
        atom=molecular_geometry(system, bond_length_angstrom),
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry="D2h",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.max_cycle = 100
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("canonical RHF did not converge")
    return molecule, mean_field


def _fix_orbital_signs(coefficients: np.ndarray) -> np.ndarray:
    """Choose deterministic column signs without changing spatial orbitals."""
    coefficients = np.asarray(coefficients, dtype=float).copy()
    for column in range(coefficients.shape[1]):
        pivot = int(np.argmax(np.abs(coefficients[:, column])))
        if coefficients[pivot, column] < 0.0:
            coefficients[:, column] *= -1.0
    return coefficients


def _pipek_mezey_block(molecule, coefficients: np.ndarray) -> np.ndarray:
    """Localize one nonempty orbital block with Pipek--Mezey.

    Blocks containing zero or one orbital are returned unchanged because no
    nontrivial unitary rotation exists in those spaces.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape[1] <= 1:
        return coefficients.copy()
    localizer = lo.PM(molecule, coefficients)
    localizer.conv_tol = 1e-10
    localizer.max_cycle = 1000
    localizer.verbose = 0
    return np.asarray(localizer.kernel(), dtype=float)


def construct_orbital_bases(molecule, mean_field) -> dict[str, dict]:
    """Construct canonical and two split Pipek--Mezey orbital bases.

    The occupied/virtual partition is never mixed. ``split_pm_irrep`` further
    divides each partition according to the canonical orbital's D2h irrep.

    Returns
    -------
    orbital_bases
        Mapping containing AO-basis orbital coefficients, canonical-to-basis
        rotations, localization block definitions, and source irrep labels.
    """
    canonical = np.asarray(mean_field.mo_coeff, dtype=float)
    overlap = mean_field.get_ovlp()
    occupations = np.asarray(mean_field.mo_occ)
    occupied = np.flatnonzero(occupations > 0)
    virtual = np.flatnonzero(occupations == 0)
    irrep_labels = np.asarray(
        symm.label_orb_symm(
            molecule,
            molecule.irrep_name,
            molecule.symm_orb,
            canonical,
        )
    )

    block_definitions = {
        "canonical": [list(range(canonical.shape[1]))],
        "split_pm": [occupied.tolist(), virtual.tolist()],
        "split_pm_irrep": [],
    }
    for orbital_partition in (occupied, virtual):
        for irrep in molecule.irrep_name:
            block = orbital_partition[irrep_labels[orbital_partition] == irrep]
            if len(block):
                block_definitions["split_pm_irrep"].append(block.tolist())

    result = {}
    for basis_name in ("canonical", "split_pm", "split_pm_irrep"):
        coefficients = canonical.copy()
        if basis_name != "canonical":
            for block in block_definitions[basis_name]:
                coefficients[:, block] = _pipek_mezey_block(
                    molecule, canonical[:, block]
                )
        coefficients = _fix_orbital_signs(coefficients)
        orthonormality_error = float(
            np.max(
                np.abs(
                    coefficients.T @ overlap @ coefficients
                    - np.eye(coefficients.shape[1])
                )
            )
        )
        occupied_projector_error = float(
            np.max(
                np.abs(
                    coefficients[:, occupied] @ coefficients[:, occupied].T
                    - canonical[:, occupied] @ canonical[:, occupied].T
                )
            )
        )
        if orthonormality_error > 1e-9:
            raise RuntimeError(
                f"{basis_name}: localized orbitals are not orthonormal"
            )
        if occupied_projector_error > 1e-8:
            raise RuntimeError(
                f"{basis_name}: localization changed the occupied subspace"
            )
        result[basis_name] = {
            "coefficients": coefficients,
            "canonical_to_basis_rotation": canonical.T @ overlap @ coefficients,
            "blocks": block_definitions[basis_name],
            "source_irreps": irrep_labels.tolist(),
            "orthonormality_max_error": orthonormality_error,
            "occupied_projector_max_error": occupied_projector_error,
        }
    return result


def construct_cisd_natural_orbital_basis(
    molecule,
    mean_field,
    spatial_one_rdm: np.ndarray,
) -> dict:
    """Construct natural orbitals from a canonical-basis CISD one-RDM.

    Parameters
    ----------
    molecule, mean_field
        PySCF molecule and converged canonical RHF calculation defining the
        AO overlap matrix and canonical molecular-orbital coefficients.
    spatial_one_rdm
        Spin-summed CISD spatial one-particle density matrix in the canonical
        MO basis. It must have shape ``(n_orbitals, n_orbitals)`` and trace
        equal to the molecular electron count.

    Returns
    -------
    natural_basis
        Orbital-basis record compatible with :func:`run_basis`. Natural
        orbitals are ordered by decreasing occupation number. Column signs
        are fixed deterministically, but degenerate natural orbitals are not
        otherwise spatially sorted.

    Notes
    -----
    Unlike the split localization methods, this full diagonalization may mix
    canonical occupied and virtual orbitals. A new CISD calculation is
    therefore performed in the resulting natural-orbital basis.
    """
    canonical = np.asarray(mean_field.mo_coeff, dtype=float)
    one_rdm = np.asarray(spatial_one_rdm, dtype=float)
    expected_shape = (canonical.shape[1], canonical.shape[1])
    if one_rdm.shape != expected_shape:
        raise ValueError(
            f"CISD spatial one-RDM has shape {one_rdm.shape}; expected "
            f"{expected_shape}"
        )
    hermiticity_error = float(np.max(np.abs(one_rdm - one_rdm.T)))
    if hermiticity_error > 1e-10:
        raise ValueError(
            "CISD spatial one-RDM is not Hermitian within numerical "
            f"tolerance: max error={hermiticity_error:.3e}"
        )
    trace_error = abs(float(np.trace(one_rdm)) - float(molecule.nelectron))
    if trace_error > 1e-8:
        raise ValueError(
            "CISD spatial one-RDM trace does not equal the electron count: "
            f"error={trace_error:.3e}"
        )

    occupations, rotation = np.linalg.eigh(0.5 * (one_rdm + one_rdm.T))
    descending = np.argsort(-occupations, kind="stable")
    occupations = occupations[descending]
    rotation = rotation[:, descending]
    coefficients = _fix_orbital_signs(canonical @ rotation)
    overlap = mean_field.get_ovlp()
    orthonormality_error = float(
        np.max(
            np.abs(
                coefficients.T @ overlap @ coefficients
                - np.eye(coefficients.shape[1])
            )
        )
    )
    if orthonormality_error > 1e-9:
        raise RuntimeError("CISD natural orbitals are not orthonormal")
    occupied = np.flatnonzero(np.asarray(mean_field.mo_occ) > 0)
    occupied_projector_error = float(
        np.max(
            np.abs(
                coefficients[:, occupied] @ coefficients[:, occupied].T
                - canonical[:, occupied] @ canonical[:, occupied].T
            )
        )
    )
    return {
        "coefficients": coefficients,
        "canonical_to_basis_rotation": canonical.T @ overlap @ coefficients,
        "blocks": [list(range(canonical.shape[1]))],
        "source_irreps": ["not restricted"] * canonical.shape[1],
        "orthonormality_max_error": orthonormality_error,
        "occupied_projector_max_error": occupied_projector_error,
        "natural_occupations": occupations.tolist(),
        "one_rdm_hermiticity_max_error": hermiticity_error,
        "one_rdm_trace_error": trace_error,
    }


def molecular_data_in_orbital_basis(
    molecule,
    mean_field,
    orbital_coefficients: np.ndarray,
    filename: Path,
) -> tuple[MolecularData, np.ndarray, np.ndarray]:
    """Transform AO integrals and construct an OpenFermion molecule container.

    Returns
    -------
    molecular_data, one_body_integrals, chemist_two_body_integrals
        OpenFermion container plus spatial one-electron integrals ``h[p,q]``
        and chemists' two-electron integrals ``(p q | r s)``. The container's
        two-body tensor is converted to OpenFermion's convention.
    """
    coefficients = np.asarray(orbital_coefficients, dtype=float)
    number_of_orbitals = coefficients.shape[1]
    one_body = coefficients.T @ mean_field.get_hcore() @ coefficients
    two_body_chemist = ao2mo.restore(
        1,
        ao2mo.kernel(molecule, coefficients),
        number_of_orbitals,
    )
    occupied_coefficients = coefficients[:, : molecule.nelectron // 2]
    reference_density = 2.0 * (
        occupied_coefficients @ occupied_coefficients.T
    )
    effective_potential = mean_field.get_veff(molecule, reference_density)
    reference_fock = mean_field.get_fock(
        dm=reference_density, vhf=effective_potential
    )
    fock_diagonal = np.diag(coefficients.T @ reference_fock @ coefficients)
    reference_determinant_energy = float(
        mean_field.energy_tot(
            dm=reference_density, vhf=effective_potential
        )
    )
    data = MolecularData(
        geometry=molecular_geometry_from_pyscf(molecule),
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        filename=str(filename),
    )
    data.n_orbitals = number_of_orbitals
    data.n_qubits = 2 * number_of_orbitals
    data.n_electrons = int(molecule.nelectron)
    data.nuclear_repulsion = float(molecule.energy_nuc())
    data.hf_energy = reference_determinant_energy
    data.canonical_orbitals = coefficients
    data.orbital_energies = np.asarray(fock_diagonal, dtype=float)
    data.one_body_integrals = np.asarray(one_body, dtype=float)
    data.two_body_integrals = np.asarray(
        two_body_chemist.transpose(0, 2, 3, 1), dtype=float
    )
    data.save()
    return data, one_body, np.asarray(two_body_chemist, dtype=float)


def exact_fci_energy(
    molecule,
    one_body: np.ndarray,
    two_body_chemist: np.ndarray,
) -> tuple[float, float]:
    """Return total FCI energy and ``<S^2>`` in a spatial-orbital basis."""
    solver = fci.direct_spin1.FCI(molecule)
    solver.conv_tol = 1e-12
    solver.max_cycle = 200
    energy, vector = solver.kernel(
        one_body,
        two_body_chemist,
        one_body.shape[0],
        (molecule.nelectron // 2, molecule.nelectron // 2),
        ecore=molecule.energy_nuc(),
    )
    if not solver.converged:
        raise RuntimeError("FCI did not converge")
    spin_squared, _multiplicity = solver.spin_square(
        vector,
        one_body.shape[0],
        (molecule.nelectron // 2, molecule.nelectron // 2),
    )
    return float(energy), float(spin_squared)


def find_approximate_symmetries(
    hamiltonian,
    cisd_state,
    args,
) -> dict[str, dict]:
    """Find full-rank HCT and Beam sets using the CISD commutator cost.

    Beam maximizes the negative sum of singleton costs, which is equivalent to
    minimizing ``sum_S <CISD|[H,S]^dagger[H,S]|CISD>``. Its pool includes
    products of the 50 largest Hamiltonian terms and is capped at 1000 after
    sorting by this score.
    """
    number_of_qubits, terms = qubit_operator_terms(
        hamiltonian, cisd_state.n_qubits
    )
    evaluator = GroupedSparsePauliCommutatorEvaluator(
        hamiltonian, cisd_state
    )
    singleton_cost = lambda symmetry: evaluator.cost([symmetry])
    hct_start = perf_counter()
    hct_symmetries, thresholds = HCT(
        hamiltonian,
        n_sym=number_of_qubits,
        sym_metric_func=singleton_cost,
        use_coeffs_eps=True,
        tol=args.hct_term_tolerance,
        verbose=args.verbose,
    )
    hct_seconds = perf_counter() - hct_start

    hct_masks = qubitops_to_masks(hct_symmetries, number_of_qubits)
    base_pool = build_candidate_pool(
        terms,
        number_of_qubits,
        max_candidates_from_terms=args.max_candidates_from_terms,
        include_pairwise_products=True,
        pairwise_seed_terms=50,
        max_pauli_weight=args.max_pauli_weight,
    )
    pool = list(OrderedDict.fromkeys([*hct_masks, *base_pool]))
    pool_before_cap = len(pool)
    singleton_scores = {
        mask: -evaluator.cost_mask(mask) for mask in pool
    }
    original_position = {mask: position for position, mask in enumerate(pool)}
    pool.sort(
        key=lambda mask: (
            singleton_scores[mask],
            -original_position[mask],
        ),
        reverse=True,
    )
    pool = pool[: args.max_candidate_pool_size]

    def beam_score(generators) -> float:
        total = 0.0
        for mask in qubitops_to_masks(generators, number_of_qubits):
            if mask not in singleton_scores:
                singleton_scores[mask] = -evaluator.cost_mask(mask)
            total += singleton_scores[mask]
        return total

    beam_start = perf_counter()
    beam_symmetries = beam_search_symmetries(
        hamiltonian,
        pool,
        target_rank=number_of_qubits,
        n_qubits=number_of_qubits,
        beam_width=args.beam_width,
        heavy_core_fraction=args.heavy_core_fraction,
        score_func=beam_score,
        score_is_separable=True,
        separable_score_cache=singleton_scores.copy(),
        n_processes=args.symmetry_processes,
        mp_start_method=args.mp_start_method,
    )
    beam_seconds = perf_counter() - beam_start

    result = {}
    for name, generators, seconds, extra in (
        ("HCT", hct_symmetries, hct_seconds, {"thresholds": thresholds}),
        ("Beam", beam_symmetries, beam_seconds, {}),
    ):
        individual_costs = [evaluator.cost([generator]) for generator in generators]
        validation = validate_symmetry_generators(
            hamiltonian, generators, n_qubits=number_of_qubits
        )
        validation["target_rank"] = number_of_qubits
        if validation["independent_rank"] != number_of_qubits:
            raise RuntimeError(f"{name} did not produce a full-rank set")
        if not validation["pairwise_commuting"]:
            raise RuntimeError(f"{name} generators do not commute pairwise")
        result[name] = {
            "generators": generators,
            "symmetries": [encode_qubit_operator(item) for item in generators],
            "symmetry_strings": [str(item) for item in generators],
            "individual_costs": individual_costs,
            "total_cost": float(sum(individual_costs)),
            "score": float(-sum(individual_costs)),
            "seconds": seconds,
            "validation": validation,
            **extra,
        }
    result["settings"] = {
        "objective": "<CISD|[H,S]^dagger[H,S]|CISD>",
        "score_convention": "maximize negative singleton cost sum",
        "pairwise_seed_terms": 50,
        "pool_size_before_cap": pool_before_cap,
        "pool_size_after_cap": len(pool),
        "pool_cap": args.max_candidate_pool_size,
        "beam_width": args.beam_width,
        "heavy_core_fraction": args.heavy_core_fraction,
        "evaluator_preparation_seconds": evaluator.preparation_seconds,
    }
    return result


def compute_fiedler_frames(
    cisd_state,
    hamiltonian,
    symmetry_results: dict[str, dict],
) -> dict[str, dict]:
    """Construct CISD-mutual-information Fiedler frames for one basis.

    Parameters
    ----------
    cisd_state
        Normalized interleaved-spin ``SparseQubitState`` in the current
        spatial-orbital basis.
    hamiltonian
        Jordan--Wigner Pauli Hamiltonian in the same qubit ordering.
    symmetry_results
        Saved or newly generated HCT and Beam records. Each method record must
        contain decoded Pauli generators under ``generators``.

    Returns
    -------
    frames
        Frame definitions for fermionic, raw-qubit, HCT, and Beam DMRG.
        Every ``ordering`` follows ``new_position -> old_site``. Qubit frame
        records also contain the transformed Hamiltonian and the complete
        unitary sequence applied to the CISD warm start.
    """
    n_qubits = int(cisd_state.n_qubits)
    frames: dict[str, dict] = {}
    fermionic_info = fiedler_order_spatial_orbitals_from_sparse_state(
        cisd_state,
        n_spatial_orbitals=n_qubits // 2,
        base=2.0,
        mutual_info_convention="standard",
        component_order="total_weight",
    )
    frames["raw_fermionic_su2"] = {
        "fiedler": fermionic_info,
        "symmetries": [],
    }

    raw_info = fiedler_order_from_state(
        cisd_state,
        n_qubits=n_qubits,
        base=2.0,
        mutual_info_convention="standard",
        component_order="total_weight",
        mutual_information_method="sparse_bloch",
    )
    raw_permutation = tuple(int(value) for value in raw_info["old_to_new"])
    frames["raw_qubit"] = {
        "fiedler": raw_info,
        "hamiltonian": permute_qubits_in_qubit_operator(
            hamiltonian, raw_permutation
        ),
        "unitaries": (PermutationUnitary(raw_permutation),),
        "symmetries": [],
    }

    for method in ("HCT", "Beam"):
        generators = symmetry_results[method]["generators"]
        clifford = Clifford.from_symmetries(
            generators,
            n_qubits=n_qubits,
            symmetry_qubits_first=True,
            synthesis_basis="Z",
            generator_mapping="positive_z",
        )
        clifford_hamiltonian = clifford.transform(hamiltonian)
        transformed_state = clifford.transform_sparse_state(
            cisd_state, drop_tol=0.0
        )
        transformed_info = fiedler_order_from_state(
            transformed_state,
            n_qubits=n_qubits,
            base=2.0,
            mutual_info_convention="standard",
            component_order="total_weight",
            mutual_information_method="sparse_bloch",
        )
        permutation = tuple(
            int(value) for value in transformed_info["old_to_new"]
        )
        final_symmetries = [
            permute_qubits_in_qubit_operator(
                clifford.transform(generator), permutation
            )
            for generator in generators
        ]
        frames[method] = {
            "fiedler": transformed_info,
            "hamiltonian": permute_qubits_in_qubit_operator(
                clifford_hamiltonian, permutation
            ),
            "unitaries": (clifford, PermutationUnitary(permutation)),
            "symmetries": generators,
            "final_symmetry_strings": [
                str(item) for item in final_symmetries
            ],
            "transformed_cisd_determinants": int(transformed_state.nnz),
        }
    return frames


def portable_fiedler_record(frame: dict) -> dict:
    """Return JSON-compatible Fiedler diagnostics from a frame definition."""
    info = frame["fiedler"]
    keys = (
        "ordering",
        "old_to_new",
        "components",
        "mutual_information",
        "one_qubit_entropies",
        "two_qubit_entropies",
        "one_orbital_entropies",
        "two_orbital_entropies",
        "mutual_info_convention",
        "mutual_information_method",
        "entropy_base",
        "edge_tol",
        "tie_break",
        "component_order",
    )
    record = {key: info[key] for key in keys if key in info}
    if "final_symmetry_strings" in frame:
        record["final_symmetry_strings"] = frame[
            "final_symmetry_strings"
        ]
        record["transformed_cisd_determinants"] = frame[
            "transformed_cisd_determinants"
        ]
    return record


def compute_standard_frames(
    hamiltonian,
    n_qubits: int,
    symmetry_results: dict[str, dict],
) -> dict[str, dict]:
    """Construct frame definitions without an additional site permutation.

    Parameters
    ----------
    hamiltonian
        Jordan--Wigner Pauli Hamiltonian in the current orbital ordering.
    n_qubits
        Number of spin-orbital qubits.
    symmetry_results
        HCT and Beam records containing decoded Pauli generators.

    Returns
    -------
    frames
        Definitions for raw fermionic, raw qubit, HCT, and Beam calculations.
        Clifford frame records include the transformed Hamiltonian and the
        unitary applied to the CISD warm start.
    """
    frames = {
        "raw_fermionic_su2": {"symmetries": []},
        "raw_qubit": {
            "hamiltonian": hamiltonian,
            "unitaries": (),
            "symmetries": [],
        },
    }
    for method in ("HCT", "Beam"):
        generators = symmetry_results[method]["generators"]
        clifford = Clifford.from_symmetries(
            generators,
            n_qubits=n_qubits,
            symmetry_qubits_first=True,
            synthesis_basis="Z",
            generator_mapping="positive_z",
        )
        frames[method] = {
            "hamiltonian": clifford.transform(hamiltonian),
            "unitaries": (clifford,),
            "symmetries": generators,
        }
    return frames


def run_basis(orbital_basis: str, basis_data: dict, args) -> tuple[list[dict], dict]:
    """Prepare and benchmark all requested frames for one orbital basis."""
    basis_dir = args.output_dir / orbital_basis
    prepared_dir = basis_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    molecule_path = prepared_dir / "molecule"
    molecular_data, one_body, two_body = molecular_data_in_orbital_basis(
        basis_data["pyscf_molecule"],
        basis_data["mean_field"],
        basis_data["coefficients"],
        molecule_path,
    )
    fci_energy, spin_squared = exact_fci_energy(
        basis_data["pyscf_molecule"], one_body, two_body
    )
    cisd_energy, cisd_state, cisd_metadata = (
        run_restricted_cisd_from_molecular_data(
            molecular_data,
            frozen_core_orbitals=0,
            coefficient_tolerance=0.0,
            verbose=0,
        )
    )
    if abs(fci_energy - args.canonical_fci_energy) > 1e-8:
        raise RuntimeError(
            f"{orbital_basis}: FCI energy changed by orbital rotation: "
            f"{fci_energy - args.canonical_fci_energy:+.3e} Ha"
        )
    if spin_squared > 1e-7:
        raise RuntimeError(
            f"{orbital_basis}: FCI state is not a singlet; <S^2>={spin_squared}"
        )
    if (
        orbital_basis != "natural"
        and abs(cisd_energy - args.canonical_cisd_energy) > 1e-8
    ):
        raise RuntimeError(
            f"{orbital_basis}: CISD energy changed by split localization: "
            f"{cisd_energy - args.canonical_cisd_energy:+.3e} Ha"
        )

    fermion_hamiltonian = get_fermion_operator(
        molecular_data.get_molecular_hamiltonian()
    )
    hamiltonian = jordan_wigner_pauli_stream(
        fermion_hamiltonian,
        n_qubits=molecular_data.n_qubits,
    )
    save_sparse_qubit_state(prepared_dir / "cisd_state.npz", cisd_state)
    save_pauli_term_stream(prepared_dir / "hamiltonian.json", hamiltonian)
    np.savez_compressed(
        prepared_dir / "orbitals.npz",
        coefficients=basis_data["coefficients"],
        canonical_to_basis_rotation=basis_data["canonical_to_basis_rotation"],
    )

    symmetry_file = prepared_dir / "symmetries.json"
    if args.resume and symmetry_file.exists():
        saved_symmetries = load_json(symmetry_file)
        symmetry_results = {}
        for name in ("HCT", "Beam"):
            record = dict(saved_symmetries[name])
            record["generators"] = [
                decode_qubit_operator(item) for item in record["symmetries"]
            ]
            symmetry_results[name] = record
        symmetry_results["settings"] = saved_symmetries["settings"]
    else:
        symmetry_results = find_approximate_symmetries(
            hamiltonian, cisd_state, args
        )
        save_json(
            symmetry_file,
            {
                key: {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key != "generators"
                }
                if key in {"HCT", "Beam"}
                else value
                for key, value in symmetry_results.items()
            },
        )

    for method in ("HCT", "Beam"):
        method_result = symmetry_results[method]
        print(
            f"{orbital_basis}/{method}: selected "
            f"{len(method_result['generators'])} Pauli generators; "
            f"total CISD commutator cost="
            f"{method_result['total_cost']:.12e}",
            flush=True,
        )
        for index, (generator, cost) in enumerate(
            zip(
                method_result["generators"],
                method_result["individual_costs"],
                strict=True,
            ),
            start=1,
        ):
            print(
                f"  {method}[{index:02d}] cost={cost:.12e}  {generator}",
                flush=True,
            )

    metadata = {
        "system": args.system,
        "bond_length_angstrom": args.bond_length,
        "bond_length_definition": (
            "N--N distance" if args.molecule_name == "N2" else
            "nearest-neighbor H--H spacing"
        ),
        "geometry": molecular_geometry_from_pyscf(
            basis_data["pyscf_molecule"]
        ),
        "basis": "sto-3g",
        "orbital_basis": orbital_basis,
        "localization_blocks": basis_data["blocks"],
        "source_irreps": basis_data["source_irreps"],
        "orthonormality_max_error": basis_data["orthonormality_max_error"],
        "occupied_projector_max_error": basis_data[
            "occupied_projector_max_error"
        ],
        "n_spatial_orbitals": molecular_data.n_orbitals,
        "n_qubits": molecular_data.n_qubits,
        "n_electrons": molecular_data.n_electrons,
        "hf_energy": molecular_data.hf_energy,
        "cisd_energy": cisd_energy,
        "cisd_energy_difference_from_canonical": (
            cisd_energy - args.canonical_cisd_energy
        ),
        "fci_energy": fci_energy,
        "fci_spin_squared": spin_squared,
        "cisd": cisd_metadata,
        "symmetry_search": {
            key: {
                item_key: item_value
                for item_key, item_value in value.items()
                if item_key != "generators"
            }
            if key in {"HCT", "Beam"}
            else value
            for key, value in symmetry_results.items()
        },
    }
    if orbital_basis == "natural":
        metadata["natural_occupations"] = basis_data["natural_occupations"]
        metadata["natural_orbitals_source"] = "canonical CISD spatial one-RDM"
        metadata["natural_orbital_ordering"] = "decreasing occupation"
    save_json(prepared_dir / "metadata.json", metadata)

    benchmark_path = basis_dir / "benchmark.json"
    result = (
        load_json(benchmark_path)
        if args.resume and benchmark_path.exists()
        else {**metadata, "frames": {}}
    )
    curve_path = basis_dir / "dmrg_curves.csv"
    curve_json_path = basis_dir / "dmrg_curves.json"
    rows = (
        load_dmrg_curve_rows(curve_path, curve_json_path)
        if args.resume
        else []
    )

    standard_frames = compute_standard_frames(
        hamiltonian, molecular_data.n_qubits, symmetry_results
    )
    fiedler_frames = None
    if "fiedler" in args.orderings:
        fiedler_frames = compute_fiedler_frames(
            cisd_state, hamiltonian, symmetry_results
        )
        save_json(
            basis_dir / "fiedler_orderings.json",
            {
                name: portable_fiedler_record(frame)
                for name, frame in fiedler_frames.items()
            },
        )

    if args.prepare_only:
        save_json(benchmark_path, result)
        return rows, result

    for ordering in args.orderings:
        frame_definitions = (
            standard_frames if ordering == "standard" else fiedler_frames
        )
        for symmetry_method in args.frames:
            output_frame = (
                symmetry_method
                if ordering == "standard"
                else f"{symmetry_method}_Fiedler"
            )
            if output_frame in result["frames"]:
                print(
                    f"{orbital_basis}/{output_frame}: reusing completed frame",
                    flush=True,
                )
                continue
            frame = frame_definitions[symmetry_method]
            print(
                f"\n{orbital_basis}/{output_frame}: starting binary DMRG "
                f"search over [{args.min_bond_dim}, {args.max_bond_dim}]",
                flush=True,
            )
            if ordering == "fiedler":
                print(
                    f"{orbital_basis}/{output_frame}: ordering "
                    f"{frame['fiedler']['ordering']}",
                    flush=True,
                )

            other_rows = [
                row for row in rows if row.get("frame") != output_frame
            ]
            saved_frame_rows = {
                int(row["bond_dim"]): dict(row)
                for row in rows
                if row.get("frame") == output_frame
            }

            def decorate_row(row: dict) -> dict:
                decorated = {
                    **row,
                    "system": args.system,
                    "orbital_basis": orbital_basis,
                    "ordering": ordering,
                    "symmetry_method": symmetry_method,
                    "frame": output_frame,
                }
                if ordering == "fiedler":
                    decorated["fiedler_ordering"] = frame["fiedler"][
                        "ordering"
                    ]
                return decorated

            def checkpoint(row: dict) -> None:
                decorated = decorate_row(row)
                saved_frame_rows[int(decorated["bond_dim"])] = decorated
                save_dmrg_curve_rows(
                    other_rows
                    + [saved_frame_rows[key] for key in sorted(saved_frame_rows)],
                    curve_path,
                    curve_json_path,
                )

            prior_rows = tuple(
                saved_frame_rows[key] for key in sorted(saved_frame_rows)
            )

            def select_next(invocation_rows) -> int | None:
                return next_binary_bond_dimension(
                    prior_rows + tuple(invocation_rows),
                    args.min_bond_dim,
                    args.max_bond_dim,
                )

            if symmetry_method == "raw_fermionic_su2":
                orbital_ordering = (
                    list(range(molecular_data.n_orbitals))
                    if ordering == "standard"
                    else frame["fiedler"]["ordering"]
                )
                backend_args = SimpleNamespace(
                    output_dir=basis_dir,
                    frame_label=output_frame,
                    orbital_ordering=orbital_ordering,
                    bond_dims=(args.min_bond_dim, args.max_bond_dim),
                    bond_dim_selector=select_next,
                    bond_result_callback=checkpoint,
                    dmrg_sweeps=args.dmrg_sweeps,
                    sweep_tol=args.sweep_tolerance,
                    dmrg_tol=args.dmrg_tolerance,
                    initial_state="cisd",
                    save_tensor_networks=True,
                    full_curve=True,
                    n_threads=args.n_threads,
                    n_mkl_threads=args.n_mkl_threads,
                    stack_mem_gb=args.stack_mem_gb,
                    davidson_threshold=args.davidson_threshold,
                    warm_start_noises=args.warm_start_noises,
                    require_sweep_convergence_for_accuracy=True,
                    verbose=args.verbose,
                )
                frame_rows, summary = (
                    fermionic_backend.run_fermionic_dmrg_curve(
                        molecule=molecular_data,
                        warm_start_state=cisd_state,
                        warm_start_energy=cisd_energy,
                        fci_energy=fci_energy,
                        args=backend_args,
                    )
                )
            else:
                frame_rows, summary = run_block2_qubit_dmrg_curve(
                    label=output_frame,
                    hamiltonian=frame["hamiltonian"],
                    sparse_state=(cisd_state.indices, cisd_state.coeffs),
                    exact_energy=fci_energy,
                    warm_start_energy=cisd_energy,
                    n_qubits=molecular_data.n_qubits,
                    bond_dims=(args.min_bond_dim, args.max_bond_dim),
                    dmrg_sweeps=args.dmrg_sweeps,
                    dmrg_tolerance=args.dmrg_tolerance,
                    sweep_tolerance=args.sweep_tolerance,
                    mpo_cutoff=args.mpo_cutoff,
                    mpo_builder=args.mpo_builder,
                    sum_mpo_mod=args.sum_mpo_mod,
                    initial_state="cisd",
                    sparse_batch_size=args.sparse_batch_size,
                    sparse_compression_cutoff=args.mps_cutoff,
                    unitaries=frame["unitaries"],
                    transform_max_bond=args.max_bond_dim,
                    transform_cutoff=args.mps_cutoff,
                    validate_warm_start_energy=True,
                    full_curve=True,
                    n_threads=args.n_threads,
                    n_mkl_threads=args.n_mkl_threads,
                    stack_mem_gb=args.stack_mem_gb,
                    davidson_threshold=args.davidson_threshold,
                    warm_start_noises=args.warm_start_noises,
                    verbose=args.verbose,
                    scratch=(
                        None
                        if args.scratch is None
                        else args.scratch / orbital_basis / output_frame
                    ),
                    artifact_dir=(
                        basis_dir / "tensor_networks" / output_frame
                    ),
                    bond_result_callback=checkpoint,
                    bond_dim_selector=select_next,
                    require_sweep_convergence_for_accuracy=True,
                )

            for row in frame_rows:
                decorated = decorate_row(row)
                saved_frame_rows[int(decorated["bond_dim"])] = decorated
            completed_frame_rows = [
                saved_frame_rows[key] for key in sorted(saved_frame_rows)
            ]
            rows = other_rows + completed_frame_rows
            first_converged = lowest_accepted_bond_dimension(
                completed_frame_rows
            )
            first_converged_row = next(
                (
                    row
                    for row in completed_frame_rows
                    if int(row["bond_dim"]) == first_converged
                ),
                None,
            )
            summary["first_converged_bond_dim"] = first_converged
            summary["converged_within_grid"] = first_converged is not None
            summary["first_converged_dmrg_optimization_seconds"] = (
                None
                if first_converged_row is None
                else first_converged_row.get("dmrg_seconds")
            )
            summary["first_converged_per_sweep_seconds"] = (
                None
                if first_converged_row is None
                else first_converged_row.get("per_sweep_seconds")
            )
            summary["first_converged_sweep_energies"] = (
                None
                if first_converged_row is None
                else first_converged_row.get("sweep_energies")
            )
            summary["bond_search"] = BOND_SEARCH_METHOD
            summary["minimum_bond_dimension"] = args.min_bond_dim
            summary["maximum_bond_dimension"] = args.max_bond_dim
            summary["tested_bond_dimensions"] = [
                int(row["bond_dim"]) for row in completed_frame_rows
            ]
            summary["bond_dims_without_sweep_convergence"] = [
                int(row["bond_dim"])
                for row in completed_frame_rows
                if row.get("sweep_limit_reached_without_convergence") is True
            ]
            summary["ordering"] = ordering
            summary["symmetries"] = [
                encode_qubit_operator(item) for item in frame["symmetries"]
            ]
            summary["symmetry_strings"] = [
                str(item) for item in frame["symmetries"]
            ]
            if ordering == "fiedler":
                summary["fiedler"] = portable_fiedler_record(frame)
            if symmetry_method in {"HCT", "Beam"}:
                method_result = symmetry_results[symmetry_method]
                summary["symmetry_individual_costs"] = method_result[
                    "individual_costs"
                ]
                summary["symmetry_total_cost"] = method_result["total_cost"]
                summary["symmetry_score"] = method_result["score"]
                summary["symmetry_search_seconds"] = method_result["seconds"]
            result["frames"][output_frame] = summary
            save_dmrg_curve_rows(rows, curve_path, curve_json_path)
            save_json(benchmark_path, result)
            print(
                f"{orbital_basis}/{output_frame}: lowest converged bond "
                f"dimension = {first_converged}",
                flush=True,
            )
    return rows, result


def aggregate_results(args, basis_results: dict) -> None:
    """Write combined curve, summary, and full JSON outputs."""
    all_rows = []
    summary_rows = []
    for orbital_basis, result in basis_results.items():
        curve_path = args.output_dir / orbital_basis / "dmrg_curves.csv"
        curve_json_path = args.output_dir / orbital_basis / "dmrg_curves.json"
        all_rows.extend(load_dmrg_curve_rows(curve_path, curve_json_path))
        for frame, summary in result["frames"].items():
            summary_rows.append(
                {
                    "system": args.system,
                    "orbital_basis": orbital_basis,
                    "frame": frame,
                    "ordering": summary.get("ordering"),
                    "first_converged_bond_dim": summary.get(
                        "first_converged_bond_dim"
                    ),
                    "converged_within_grid": summary.get(
                        "first_converged_bond_dim"
                    )
                    is not None,
                    "mpo_bond_dimension": summary.get("mpo_bond_dimension"),
                    "symmetries": summary.get("symmetries", []),
                    "symmetry_strings": summary.get("symmetry_strings", []),
                    "symmetry_total_cost": summary.get(
                        "symmetry_total_cost"
                    ),
                    "symmetry_score": summary.get("symmetry_score"),
                    "symmetry_search_seconds": summary.get(
                        "symmetry_search_seconds"
                    ),
                    "bond_dims_without_sweep_convergence": summary.get(
                        "bond_dims_without_sweep_convergence", []
                    ),
                }
            )
    if all_rows:
        write_csv(args.output_dir / "dmrg_curves.csv", all_rows)
    if summary_rows:
        write_csv(args.output_dir / "dmrg_summary.csv", summary_rows)
    save_json(
        args.output_dir / "benchmark.json",
        {
            "settings": {
                "system": args.system,
                "bond_length_angstrom": args.bond_length,
                "basis": "sto-3g",
                "orbital_bases": list(args.orbital_bases),
                "orderings": list(args.orderings),
                "frames": list(args.frames),
                "bond_search": BOND_SEARCH_METHOD,
                "minimum_bond_dimension": args.min_bond_dim,
                "maximum_bond_dimension": args.max_bond_dim,
                "dmrg_sweeps": args.dmrg_sweeps,
                "sweep_tolerance": args.sweep_tolerance,
                "chemical_accuracy_hartree": args.dmrg_tolerance,
                "convergence_definition": (
                    "chemical accuracy and sweep-energy convergence"
                ),
            },
            "orbital_bases": basis_results,
            "summary_rows": summary_rows,
            "curve_rows": all_rows,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=tuple(SYSTEM_DEFAULTS),
        default="N2_corr",
        help="Molecular system and geometry regime to benchmark.",
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        help=(
            "Override the default N--N distance or adjacent H--H spacing, "
            "in Angstrom. H6_corr and H8_corr default to 2.0 Angstrom."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: saved/results/<system>_sto3g_orbital_dmrg.",
    )
    parser.add_argument(
        "--orbital-bases",
        nargs="+",
        choices=ORBITAL_BASES,
        default=list(ORBITAL_BASES),
        help="Spatial-orbital representations to test.",
    )
    parser.add_argument(
        "--orderings",
        nargs="+",
        choices=ORDERINGS,
        default=["standard"],
        help="Test the original site order, CISD Fiedler order, or both.",
    )
    parser.add_argument("--frames", nargs="+", choices=FRAMES, default=list(FRAMES))
    parser.add_argument("--min-bond-dim", type=int, default=1)
    parser.add_argument("--max-bond-dim", type=int, default=100)
    parser.add_argument("--dmrg-sweeps", type=int, default=100)
    parser.add_argument("--sweep-tolerance", type=float, default=1e-6)
    parser.add_argument("--dmrg-tolerance", type=float, default=CHEMICAL_ACCURACY_HARTREE)
    parser.add_argument("--davidson-threshold", type=float, default=1e-10)
    parser.add_argument("--warm-start-noises", nargs="*", type=float, default=list(DEFAULT_WARM_START_NOISES))
    parser.add_argument("--mps-cutoff", type=float, default=1e-13)
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument("--mpo-builder", choices=("blocked_sum", "expression"), default="blocked_sum")
    parser.add_argument("--sum-mpo-mod", type=int, default=10)
    parser.add_argument("--sparse-batch-size", type=int, default=32)
    parser.add_argument("--hct-term-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-candidates-from-terms", type=int, default=256)
    parser.add_argument("--max-candidate-pool-size", type=int, default=1000)
    parser.add_argument("--max-pauli-weight", type=int)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--heavy-core-fraction", type=float, default=0.95)
    parser.add_argument("--symmetry-processes", type=int, default=1)
    parser.add_argument("--mp-start-method")
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-mkl-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=4.0)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate orbitals, CISD/FCI data, Hamiltonians, and symmetries without DMRG.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the same workflow for H2/STO-3G at 0.74 Angstrom.",
    )
    return parser


def main() -> None:
    """Prepare requested orbital bases and run binary-search DMRG frames."""
    args = build_parser().parse_args()
    args.warm_start_noises = tuple(args.warm_start_noises)
    if args.dmrg_sweeps < 1 or args.dmrg_sweeps > 100:
        raise ValueError("--dmrg-sweeps must lie between 1 and 100")
    if args.min_bond_dim < 1:
        raise ValueError("--min-bond-dim must be positive")
    if args.max_bond_dim < args.min_bond_dim:
        raise ValueError(
            "--max-bond-dim must not be smaller than --min-bond-dim"
        )
    if args.max_candidate_pool_size < 1:
        raise ValueError("candidate pool cap must be positive")

    if args.smoke_test:
        args.system = "H2_smoke"
        args.molecule_name = "H2"
        args.bond_length = 0.74
    else:
        system_defaults = SYSTEM_DEFAULTS[args.system]
        args.molecule_name = system_defaults["molecule"]
        args.bond_length = (
            float(args.bond_length)
            if args.bond_length is not None
            else float(system_defaults["bond_length_angstrom"])
        )
    if args.bond_length <= 0.0:
        raise ValueError("--bond-length must be positive")
    if args.output_dir is None:
        directory_name = f"{args.system.lower()}_sto3g_orbital_dmrg"
        args.output_dir = DEFAULT_RESULTS_ROOT / directory_name
    args.output_dir = args.output_dir.expanduser().resolve()
    settings_path = args.output_dir / "settings.json"
    requested_settings = {
        "system": args.system,
        "bond_length_angstrom": args.bond_length,
        "orbital_bases": list(args.orbital_bases),
        "orderings": list(args.orderings),
        "frames": list(args.frames),
        "bond_search": BOND_SEARCH_METHOD,
        "minimum_bond_dimension": args.min_bond_dim,
        "maximum_bond_dimension": args.max_bond_dim,
        "dmrg_sweeps": args.dmrg_sweeps,
        "sweep_tolerance": args.sweep_tolerance,
        "dmrg_tolerance": args.dmrg_tolerance,
        "davidson_threshold": args.davidson_threshold,
        "warm_start_noises": list(args.warm_start_noises),
        "mps_cutoff": args.mps_cutoff,
        "mpo_cutoff": args.mpo_cutoff,
        "mpo_builder": args.mpo_builder,
        "sum_mpo_mod": args.sum_mpo_mod,
        "sparse_batch_size": args.sparse_batch_size,
        "hct_term_tolerance": args.hct_term_tolerance,
        "max_candidates_from_terms": args.max_candidates_from_terms,
        "max_candidate_pool_size": args.max_candidate_pool_size,
        "max_pauli_weight": args.max_pauli_weight,
        "beam_width": args.beam_width,
        "heavy_core_fraction": args.heavy_core_fraction,
        "symmetry_objective": "<CISD|[H,S]^dagger[H,S]|CISD>",
        "symmetry_rank": "number of qubits",
    }
    if settings_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{settings_path} exists; use --resume or a new --output-dir"
            )
        saved_settings = load_json(settings_path)
        if saved_settings.get("bond_search") == "integer_binary_v1":
            saved_settings["bond_search"] = BOND_SEARCH_METHOD
        if saved_settings != requested_settings:
            raise ValueError("saved settings differ; use a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(settings_path, requested_settings)

    log_path = args.output_dir / "benchmark.txt"
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
            molecule, mean_field = run_canonical_rhf(
                args.molecule_name, args.bond_length
            )
            orbital_bases = construct_orbital_bases(molecule, mean_field)

            canonical_molecule, canonical_h1, canonical_eri = molecular_data_in_orbital_basis(
                molecule,
                mean_field,
                orbital_bases["canonical"]["coefficients"],
                args.output_dir / "canonical_reference_molecule",
            )
            args.canonical_fci_energy, _ = exact_fci_energy(
                molecule, canonical_h1, canonical_eri
            )
            args.canonical_cisd_energy, _state, _metadata = (
                run_restricted_cisd_from_molecular_data(
                    canonical_molecule,
                    frozen_core_orbitals=0,
                    coefficient_tolerance=0.0,
                    include_spatial_one_rdm=(
                        "natural" in args.orbital_bases
                    ),
                    verbose=0,
                )
            )
            if "natural" in args.orbital_bases:
                orbital_bases["natural"] = (
                    construct_cisd_natural_orbital_basis(
                        molecule,
                        mean_field,
                        np.asarray(_metadata["spatial_one_rdm"]),
                    )
                )
                np.savez_compressed(
                    args.output_dir / "natural_orbitals_source.npz",
                    canonical_cisd_spatial_one_rdm=np.asarray(
                        _metadata["spatial_one_rdm"], dtype=float
                    ),
                    natural_occupations=np.asarray(
                        orbital_bases["natural"]["natural_occupations"],
                        dtype=float,
                    ),
                    canonical_to_natural_rotation=np.asarray(
                        orbital_bases["natural"][
                            "canonical_to_basis_rotation"
                        ],
                        dtype=float,
                    ),
                )
            for data in orbital_bases.values():
                data["pyscf_molecule"] = molecule
                data["mean_field"] = mean_field
            print(
                f"{args.system}: RHF={mean_field.e_tot:.12f}, "
                f"CISD={args.canonical_cisd_energy:.12f}, "
                f"FCI={args.canonical_fci_energy:.12f}",
                flush=True,
            )

            basis_results = {}
            for orbital_basis in args.orbital_bases:
                print(f"\n{'=' * 80}\nOrbital basis: {orbital_basis}\n{'=' * 80}", flush=True)
                _rows, result = run_basis(
                    orbital_basis, orbital_bases[orbital_basis], args
                )
                basis_results[orbital_basis] = result
                aggregate_results(args, basis_results)
            print(f"\nSaved results to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
