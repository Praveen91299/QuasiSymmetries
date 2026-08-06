"""Rerun every saved pyblock2 DMRG frame with initial warm-start noise.

This is the single DMRG-only entry point for the completed molecular
benchmark data.  It does not repeat chemistry, CISD, symmetry search,
Clifford synthesis, entropy, BLISS, or Fiedler calculations. It can either
reuse the saved FCI energy or replace it with one fixed-large-bond reference
DMRG calculation. It then
loads their portable saved inputs and reconstructs:

* raw fermionic SU(2) and raw Jordan--Wigner qubit references;
* HCT and beam-search n_q/2 and n_q Clifford frames;
* BLISS-HCT n_q;
* HCT n_q and beam n_q with their saved Fiedler permutations; and
* the orbital-pair seniority Clifford frame.

Each CISD-warm DMRG run uses a short initial noise schedule, followed by
zero-noise sweeps.  Per-frame partial outputs make ``--resume`` safe.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import gc
from collections import OrderedDict
from pathlib import Path

import numpy as np
from openfermion import MolecularData
from pyscf import ao2mo, lib

from quasisymmetries.block2_qubit_benchmark import (
    run_block2_qubit_dmrg_curve,
    run_block2_qubit_reference_dmrg,
)
from quasisymmetries.clifford_symmetry_optimized import (
    Clifford,
    permute_qubits_in_qubit_operator,
)
from quasisymmetries.fiedler import invert_ordering
from quasisymmetries.mps_unitary import PermutationUnitary
from quasisymmetries.save import (
    load_json,
    load_pauli_term_stream,
    load_sparse_qubit_state,
    read_csv,
    save_json,
    write_csv,
)
from quasisymmetries.sym import get_seniority_symmetries

import benchmark_raw_n2_dmrg as fermionic_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOTS = (
    ROOT / "saved" / "results" / "pyblock2_16_systems",
    ROOT / "saved" / "results" / "pyblock2_o2",
)
DEFAULT_SENIORITY_ROOT = (
    ROOT / "saved" / "results" / "pyblock2_seniority"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "saved" / "results" / "pyblock2_all_noisy"
)
DEFAULT_NOISES = (1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6)
CHEMICAL_ACCURACY = 1.6e-3

HCT_N2 = "HCT N/2 Comm"
HCT_N = "HCT N Comm"
BLISS_HCT_N = "Pauli BLISS+HCT N Comm"
BS_N2 = "BS N/2 Comm"
BS_N = "BS N Comm"

RAW_FERMION = "raw_fermionic_su2"
RAW_QUBIT = "raw_qubit"
SENIORITY = "seniority_Nover2"
HCT_N2_FRAME = "HCT_Nover2_Comm"
HCT_N_FRAME = "HCT_N_Comm"
BLISS_FRAME = "Pauli_BLISS+HCT_N_Comm"
BS_N2_FRAME = "BS_Nover2_Comm"
BS_N_FRAME = "BS_N_Comm"
HCT_FIEDLER_FRAME = "HCT_N_Comm_Fiedler"
BS_FIEDLER_FRAME = "BS_N_Comm_Fiedler"

ALL_FRAMES = (
    RAW_FERMION,
    RAW_QUBIT,
    SENIORITY,
    HCT_N2_FRAME,
    HCT_N_FRAME,
    BLISS_FRAME,
    BS_N2_FRAME,
    BS_N_FRAME,
    HCT_FIEDLER_FRAME,
    BS_FIEDLER_FRAME,
)

FRAME_DATASET = {
    HCT_N2_FRAME: HCT_N2,
    HCT_N_FRAME: HCT_N,
    BLISS_FRAME: BLISS_HCT_N,
    BS_N2_FRAME: BS_N2,
    BS_N_FRAME: BS_N,
    HCT_FIEDLER_FRAME: HCT_N,
    BS_FIEDLER_FRAME: BS_N,
}


def discover_inputs(input_roots) -> OrderedDict[str, Path]:
    discovered = OrderedDict()
    for raw_root in input_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            print(f"Skipping absent input root {root}", flush=True)
            continue
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            input_dir = child / "orbital_optimization_inputs"
            manifest_path = input_dir / "manifest.json"
            if not manifest_path.exists() or not (
                child / "benchmark.json"
            ).exists():
                continue
            system = str(load_json(manifest_path)["system"])
            if system in discovered:
                raise ValueError(f"Duplicate saved system {system!r}.")
            discovered[system] = input_dir
    if not discovered:
        raise FileNotFoundError("No completed saved benchmark inputs found.")
    return discovered


def choose_systems(discovered, requested):
    if requested is None:
        return list(discovered)
    missing = [system for system in requested if system not in discovered]
    if missing:
        raise ValueError(
            f"Unavailable systems {missing}; choose from {list(discovered)}"
        )
    return list(OrderedDict((system, None) for system in requested))


def input_file(input_dir: Path, manifest: dict, section: str, key: str):
    path = (input_dir / manifest[section][key]).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_saved_input(input_dir: Path) -> dict:
    manifest = load_json(input_dir / "manifest.json")
    n_qubits = int(manifest["n_qubits"])
    hamiltonian = load_pauli_term_stream(
        input_file(
            input_dir, manifest, "hamiltonian_files", "qubit_json"
        ),
        n_qubits=n_qubits,
    )
    cisd_state = load_sparse_qubit_state(
        input_file(input_dir, manifest, "state_files", "cisd")
    ).normalize()
    cliffords = load_json(
        input_dir / manifest["cliffords_and_orderings"]
    )
    if cisd_state.n_qubits != n_qubits:
        raise ValueError(f"{manifest['system']}: inconsistent qubit count.")
    return {
        "input_dir": input_dir,
        "manifest": manifest,
        "hamiltonian": hamiltonian,
        "cisd_state": cisd_state,
        "cliffords": cliffords,
        "n_qubits": n_qubits,
        "fci_energy": float(manifest["fci_energy"]),
        "cisd_energy": float(manifest["cisd_energy"]),
    }


def molecule_from_checkpoint(path: Path, manifest: dict) -> MolecularData:
    molecule = lib.chkfile.load_mol(str(path))
    scf_data = lib.chkfile.load(str(path), "scf")
    mo_coeff = np.asarray(scf_data["mo_coeff"])
    n_orbitals = int(mo_coeff.shape[1])
    one_body = np.linalg.multi_dot(
        (mo_coeff.T, molecule.intor("int1e_kin")
         + molecule.intor("int1e_nuc"), mo_coeff)
    )
    eri = ao2mo.restore(
        1, ao2mo.kernel(molecule, mo_coeff), n_orbitals
    )
    result = MolecularData(
        geometry="saved-checkpoint",
        basis=str(manifest["source"].get("basis", "unknown")),
        multiplicity=int(molecule.spin) + 1,
        charge=int(molecule.charge),
    )
    result.n_orbitals = n_orbitals
    result.n_qubits = 2 * n_orbitals
    result.n_electrons = int(molecule.nelectron)
    result.nuclear_repulsion = float(molecule.energy_nuc())
    result.one_body_integrals = np.asarray(one_body, dtype=float)
    result.two_body_integrals = np.asarray(
        eri.transpose(0, 2, 3, 1), dtype=float
    )
    return result


def load_fermionic_molecule(manifest: dict) -> MolecularData:
    source = manifest["source"]
    if source["kind"] == "openfermion_molecular_data":
        path = Path(source["molecular_data"]).expanduser().resolve()
        base = path.with_suffix("") if path.suffix == ".hdf5" else path
        molecule = MolecularData(filename=str(base))
        molecule.load()
        return molecule
    if source["kind"] in {
        "pyscf_checkpoint",
        "pyscf_rohf_checkpoint",
    }:
        return molecule_from_checkpoint(
            Path(source["checkpoint"]).expanduser().resolve(),
            manifest,
        )
    raise ValueError(
        f"Unsupported fermionic source kind {source['kind']!r}."
    )


def saved_clifford(data: dict, tag: str) -> Clifford:
    if tag not in data["cliffords"]:
        raise KeyError(
            f"{data['manifest']['system']}: no saved Clifford for {tag!r}"
        )
    return Clifford.from_dict(data["cliffords"][tag]["clifford"])


def build_qubit_frame(data: dict, frame: str, seniority_root: Path):
    hamiltonian = data["hamiltonian"]
    n_qubits = data["n_qubits"]
    if frame == RAW_QUBIT:
        return hamiltonian, (), ""
    if frame == SENIORITY:
        saved_path = (
            seniority_root
            / data["manifest"]["system"]
            / "prepared"
            / "clifford.json"
        )
        clifford = (
            Clifford.from_dict(load_json(saved_path))
            if saved_path.exists()
            else Clifford.from_symmetries(
                get_seniority_symmetries(n_qubits),
                n_qubits=n_qubits,
                symmetry_qubits_first=True,
                synthesis_basis="Z",
                generator_mapping="positive_z",
            )
        )
        return clifford.transform(hamiltonian), (clifford,), "Seniority N/2"

    tag = FRAME_DATASET[frame]
    clifford = saved_clifford(data, tag)
    transformed = clifford.transform(hamiltonian)
    if frame in {HCT_FIEDLER_FRAME, BS_FIEDLER_FRAME}:
        fiedler = data["cliffords"][tag].get("fiedler")
        if not fiedler:
            raise ValueError(
                f"{data['manifest']['system']}: missing saved {tag} "
                "Fiedler ordering."
            )
        ordering = [int(value) for value in fiedler["ordering"]]
        permutation = tuple(invert_ordering(ordering))
        transformed = permute_qubits_in_qubit_operator(
            transformed, permutation
        )
        return (
            transformed,
            (clifford, PermutationUnitary(permutation)),
            tag,
        )
    return transformed, (clifford,), tag


def partial_paths(system_dir: Path):
    return (
        system_dir / "dmrg_curves.partial.csv",
        system_dir / "dmrg_summaries.partial.json",
    )


def load_partial(system_dir: Path, resume: bool):
    curve_path, summary_path = partial_paths(system_dir)
    if not resume:
        return [], {}
    rows = read_csv(curve_path) if curve_path.exists() else []
    summaries = load_json(summary_path) if summary_path.exists() else {}
    return rows, summaries


def save_partial(system_dir: Path, rows, summaries):
    curve_path, summary_path = partial_paths(system_dir)
    if rows:
        write_csv(curve_path, rows)
    save_json(summary_path, summaries)


def run_system(system: str, input_dir: Path, args):
    print(f"\n{'=' * 80}\nStarting {system}\n{'=' * 80}", flush=True)
    system_dir = args.output_dir / system
    system_dir.mkdir(parents=True, exist_ok=True)
    data = load_saved_input(input_dir)
    rows, summaries = load_partial(system_dir, args.resume)

    if args.reference_method == "dmrg":
        reference_path = system_dir / "reference_dmrg.json"
        if args.resume and reference_path.exists():
            reference = load_json(reference_path)
            print(
                f"{system}: reusing fixed-bond DMRG reference "
                f"E={reference['energy']:.12f}.",
                flush=True,
            )
        else:
            _, reference = run_block2_qubit_reference_dmrg(
                label=system,
                hamiltonian=data["hamiltonian"],
                n_qubits=data["n_qubits"],
                bond_dim=args.reference_dmrg_bond_dim,
                sparse_state=(
                    data["cisd_state"].indices,
                    data["cisd_state"].coeffs,
                ),
                initial_state="cisd",
                dmrg_sweeps=args.reference_dmrg_sweeps,
                sweep_tolerance=args.reference_dmrg_sweep_tolerance,
                mps_cutoff=args.mps_cutoff,
                mpo_cutoff=args.mpo_cutoff,
                mpo_builder=args.mpo_builder,
                sum_mpo_mod=args.sum_mpo_mod,
                sparse_batch_size=args.sparse_batch_size,
                davidson_threshold=args.davidson_threshold,
                noises=args.warm_start_noises,
                n_threads=args.n_threads,
                stack_mem_gb=args.stack_mem_gb,
                verbose=args.verbose,
                artifact_dir=system_dir / "reference_tensor_networks",
            )
            save_json(reference_path, reference)
        reference_energy = float(reference["energy"])
    else:
        reference = {
            "method": "saved_fci",
            "energy": data["fci_energy"],
        }
        reference_energy = data["fci_energy"]

    if summaries and args.reference_method == "dmrg":
        incompatible = [
            frame
            for frame, summary in summaries.items()
            if summary.get("benchmark_reference", {}).get("method")
            != "pyblock2_qubit_reference_dmrg"
            or not np.isclose(
                summary.get("benchmark_reference", {}).get(
                    "energy", np.inf
                ),
                reference_energy,
                atol=1e-10,
            )
        ]
        if incompatible:
            raise ValueError(
                f"{system}: existing partial frames {incompatible} use a "
                "different reference; select a new --output-dir."
            )

    for frame in args.frames:
        if frame in summaries:
            print(f"Resuming: reusing completed {system}/{frame}", flush=True)
            continue
        rows = [row for row in rows if row.get("frame") != frame]
        if frame == RAW_FERMION:
            molecule = load_fermionic_molecule(data["manifest"])
            original_output = args.output_dir
            args.output_dir = system_dir
            args.sweep_tol = args.sweep_tolerance
            args.dmrg_tol = args.dmrg_tolerance
            args.initial_state = "cisd"
            try:
                frame_rows, summary = (
                    fermionic_backend.run_fermionic_dmrg_curve(
                        molecule=molecule,
                        warm_start_state=data["cisd_state"].to_dense(),
                        warm_start_energy=data["cisd_energy"],
                        fci_energy=reference_energy,
                        args=args,
                    )
                )
            finally:
                args.output_dir = original_output
            for row in frame_rows:
                row["dataset_tag"] = ""
                row["fiedler"] = False
                row["backend"] = "block2_su2"
        else:
            hamiltonian, unitaries, dataset_tag = build_qubit_frame(
                data, frame, args.seniority_root
            )
            frame_rows, summary = run_block2_qubit_dmrg_curve(
                label=frame,
                hamiltonian=hamiltonian,
                sparse_state=(
                    data["cisd_state"].indices,
                    data["cisd_state"].coeffs,
                ),
                exact_energy=reference_energy,
                warm_start_energy=data["cisd_energy"],
                n_qubits=data["n_qubits"],
                bond_dims=args.bond_dims,
                dmrg_sweeps=args.dmrg_sweeps,
                dmrg_tolerance=args.dmrg_tolerance,
                sweep_tolerance=args.sweep_tolerance,
                mps_cutoff=args.mps_cutoff,
                mpo_cutoff=args.mpo_cutoff,
                mpo_builder=args.mpo_builder,
                sum_mpo_mod=args.sum_mpo_mod,
                initial_state="cisd",
                sparse_batch_size=args.sparse_batch_size,
                sparse_compression_cutoff=args.mps_cutoff,
                unitaries=unitaries,
                transform_max_bond=max(args.bond_dims),
                transform_cutoff=args.mps_cutoff,
                full_curve=args.full_curve,
                n_threads=args.n_threads,
                stack_mem_gb=args.stack_mem_gb,
                davidson_threshold=args.davidson_threshold,
                warm_start_noises=args.warm_start_noises,
                verbose=args.verbose,
                artifact_dir=(
                    system_dir / "tensor_networks"
                    if args.save_tensor_networks
                    else None
                ),
            )
            for row in frame_rows:
                row["dataset_tag"] = dataset_tag
                row["fiedler"] = frame in {
                    HCT_FIEDLER_FRAME, BS_FIEDLER_FRAME
                }
                row["backend"] = "block2_pauli"
        for row in frame_rows:
            row["system"] = system
            row["reference_method"] = reference["method"]
            row["reference_energy"] = reference_energy
        summary["benchmark_reference"] = reference
        rows.extend(frame_rows)
        summaries[frame] = summary
        save_partial(system_dir, rows, summaries)
        if frame == RAW_FERMION:
            del molecule
        else:
            del hamiltonian, unitaries
        gc.collect()

    result = {
        "system": system,
        "source_input_dir": str(input_dir),
        "fci_energy": data["fci_energy"],
        "reference": reference,
        "reference_energy": reference_energy,
        "cisd_energy": data["cisd_energy"],
        "warm_start_noises": list(args.warm_start_noises),
        "frames": summaries,
    }
    write_csv(system_dir / "dmrg_curves.csv", rows)
    save_json(system_dir / "benchmark.json", result)
    return result, rows


def aggregate_outputs(args, settings):
    systems = OrderedDict()
    rows = []
    summaries = []
    for child in sorted(path for path in args.output_dir.iterdir() if path.is_dir()):
        benchmark_path = child / "benchmark.json"
        curve_path = child / "dmrg_curves.csv"
        if not benchmark_path.exists():
            continue
        result = load_json(benchmark_path)
        system = result["system"]
        systems[system] = result
        if curve_path.exists():
            rows.extend(read_csv(curve_path))
        for frame, summary in result["frames"].items():
            summaries.append(
                {
                    "system": system,
                    "frame": frame,
                    "first_converged_bond_dim": summary.get(
                        "first_converged_bond_dim"
                    ),
                    "converged_within_grid": summary.get(
                        "converged_within_grid", False
                    ),
                    "mpo_bond_dimension": summary.get(
                        "mpo_bond_dimension"
                    ),
                    "first_converged_dmrg_seconds": summary.get(
                        "first_converged_dmrg_optimization_seconds"
                    ),
                }
            )
    save_json(
        args.output_dir / "benchmark.json",
        {"settings": settings, "systems": systems},
    )
    if rows:
        write_csv(args.output_dir / "dmrg_curves.csv", rows)
    if summaries:
        write_csv(args.output_dir / "dmrg_summary.csv", summaries)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-roots",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INPUT_ROOTS),
    )
    parser.add_argument(
        "--seniority-root", type=Path, default=DEFAULT_SENIORITY_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument(
        "--frames", nargs="+", choices=ALL_FRAMES, default=list(ALL_FRAMES)
    )
    parser.add_argument("--list-systems", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-curve", action="store_true")
    parser.add_argument(
        "--reference-method",
        choices=("saved_fci", "dmrg"),
        default="saved_fci",
    )
    parser.add_argument("--reference-dmrg-bond-dim", type=int, default=400)
    parser.add_argument("--reference-dmrg-sweeps", type=int, default=100)
    parser.add_argument(
        "--reference-dmrg-sweep-tolerance", type=float, default=1e-8
    )
    parser.add_argument(
        "--bond-dims",
        type=int,
        nargs="+",
        default=(
            list(range(1, 11))
            + list(range(12, 21, 2))
            + list(range(30, 101, 10))
        ),
    )
    parser.add_argument("--dmrg-sweeps", type=int, default=100)
    parser.add_argument(
        "--dmrg-tolerance", type=float, default=CHEMICAL_ACCURACY
    )
    parser.add_argument("--sweep-tolerance", type=float, default=1e-6)
    parser.add_argument("--davidson-threshold", type=float, default=1e-10)
    parser.add_argument(
        "--warm-start-noises",
        type=float,
        nargs="*",
        default=list(DEFAULT_NOISES),
    )
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument("--mps-cutoff", type=float, default=1e-13)
    parser.add_argument(
        "--mpo-builder",
        choices=("blocked_sum", "expression"),
        default="blocked_sum",
    )
    parser.add_argument("--sum-mpo-mod", type=int, default=20)
    parser.add_argument("--sparse-batch-size", type=int, default=32)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=0.5)
    parser.add_argument(
        "--no-save-tensor-networks",
        dest="save_tensor_networks",
        action="store_false",
    )
    parser.set_defaults(save_tensor_networks=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(value < 0 for value in args.warm_start_noises):
        raise ValueError("Warm-start noises must be nonnegative.")
    if any(value < 1 for value in args.bond_dims):
        raise ValueError("Bond dimensions must be positive.")
    if args.reference_dmrg_bond_dim < 1:
        raise ValueError("Reference DMRG bond dimension must be positive.")
    if args.reference_dmrg_sweeps < 1:
        raise ValueError("Reference DMRG sweeps must be positive.")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.seniority_root = args.seniority_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_inputs(args.input_roots)
    systems = choose_systems(discovered, args.systems)
    if args.list_systems:
        for system in systems:
            print(f"{system}: {discovered[system]}")
        return

    settings = vars(args).copy()
    settings.update(
        {
            "selected_systems": systems,
            "selected_frames": list(args.frames),
            "dmrg_only": True,
            "saved_symmetries_reused": True,
            "saved_fiedler_orderings_reused": True,
        }
    )
    save_json(args.output_dir / "settings.json", settings)
    for system in systems:
        completed = args.output_dir / system / "benchmark.json"
        if args.resume and completed.exists():
            prior = load_json(completed)
            if all(frame in prior.get("frames", {}) for frame in args.frames):
                print(f"Resuming: reusing completed {system}", flush=True)
                continue
        run_system(system, discovered[system], args)
        aggregate_outputs(args, settings)
    aggregate_outputs(args, settings)
    print(f"Saved noisy DMRG rerun to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
