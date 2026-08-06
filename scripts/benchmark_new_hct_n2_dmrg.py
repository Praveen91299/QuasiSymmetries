"""Benchmark paper-faithful HCT symmetries for the three saved N2 geometries.

The calculation matches the saved pyblock2 molecular benchmark convention:

* full-active-space, 20-qubit N2 at 1.1, 1.5, and 2.0 Angstrom;
* CISD commutator-squared singleton ranking for HCT;
* Z-basis, positive-Z Clifford synthesis;
* CISD-warm Pauli-mode pyblock2 DMRG; and
* the first tested bond dimension within 1.6e-3 Ha of saved FCI.

Only the HCT(n/2) and HCT(n) frames are regenerated, using
``quasisymmetries.sym.HCT`` rather than the historical ``hct_mod``.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import gc
from pathlib import Path

from quasisymmetries.block2_qubit_benchmark import (
    run_block2_qubit_dmrg_curve,
)
from quasisymmetries.clifford_symmetry_optimized import Clifford
from quasisymmetries.metrics import PauliTermOverlapCommutatorEvaluator
from quasisymmetries.save import (
    load_json,
    load_pauli_term_stream,
    load_sparse_qubit_state,
    save_json,
    write_csv,
)
from quasisymmetries.sym import HCT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "saved" / "results" / "pyblock2_16_systems"
DEFAULT_OUTPUT_DIR = ROOT / "saved" / "results" / "pyblock2_new_hct_n2"
SYSTEMS = ("N2_eqm", "N2_corr", "N2_diss")
FRAME_SPECS = (
    ("new_HCT_Nover2_Comm", "N/2", lambda n_qubits: n_qubits // 2),
    ("new_HCT_N_Comm", "N", lambda n_qubits: n_qubits),
)
CHEMICAL_ACCURACY = 1.6e-3
DEFAULT_NOISES = (1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6)
DEFAULT_BOND_DIMS = (
    tuple(range(1, 11))
    + tuple(range(12, 21, 2))
    + tuple(range(30, 101, 10))
)


def load_input(input_root: Path, system: str) -> dict:
    input_dir = input_root / system / "orbital_optimization_inputs"
    manifest = load_json(input_dir / "manifest.json")
    n_qubits = int(manifest["n_qubits"])
    hamiltonian = load_pauli_term_stream(
        input_dir / manifest["hamiltonian_files"]["qubit_json"],
        n_qubits=n_qubits,
    )
    cisd_state = load_sparse_qubit_state(
        input_dir / manifest["state_files"]["cisd"]
    ).normalize()
    if cisd_state.n_qubits != n_qubits:
        raise ValueError(f"{system}: inconsistent saved qubit counts")
    return {
        "input_dir": input_dir,
        "manifest": manifest,
        "hamiltonian": hamiltonian,
        "cisd_state": cisd_state,
        "n_qubits": n_qubits,
        "fci_energy": float(manifest["fci_energy"]),
        "cisd_energy": float(manifest["cisd_energy"]),
    }


def symmetry_strings(symmetries) -> list[str]:
    return [str(symmetry) for symmetry in symmetries]


def save_system_outputs(system_dir: Path, rows, frames, metadata) -> None:
    if rows:
        write_csv(system_dir / "dmrg_curves.csv", rows)
    save_json(
        system_dir / "benchmark.json",
        {**metadata, "frames": frames},
    )


def run_system(system: str, args) -> tuple[dict, list[dict]]:
    print(f"\n{'=' * 80}\nStarting {system}\n{'=' * 80}", flush=True)
    system_dir = args.output_dir / system
    system_dir.mkdir(parents=True, exist_ok=True)
    data = load_input(args.input_root, system)
    prior_path = system_dir / "benchmark.json"
    prior = load_json(prior_path) if args.resume and prior_path.exists() else {}
    frames = dict(prior.get("frames", {}))
    rows = []
    prior_curve = system_dir / "dmrg_curves.csv"
    if args.resume and prior_curve.exists():
        from quasisymmetries.save import read_csv

        rows = read_csv(prior_curve)

    metadata = {
        "system": system,
        "source_input_dir": str(data["input_dir"]),
        "distance_angstrom": data["manifest"]["source"].get(
            "distance_angstrom"
        ),
        "n_qubits": data["n_qubits"],
        "fci_energy": data["fci_energy"],
        "cisd_energy": data["cisd_energy"],
        "hct_implementation": "quasisymmetries.sym.HCT",
        "hct_metric": "CISD commutator squared",
        "synthesis_basis": "Z",
        "generator_mapping": "positive_z",
    }

    evaluator = PauliTermOverlapCommutatorEvaluator(
        data["hamiltonian"], data["cisd_state"]
    )
    singleton_metric = lambda symmetry: evaluator.cost([symmetry])

    for frame, count_label, count_func in FRAME_SPECS:
        if args.counts and count_label not in args.counts:
            continue
        if frame in frames:
            print(f"Resuming: reusing completed {system}/{frame}", flush=True)
            continue

        n_sym = int(count_func(data["n_qubits"]))
        print(f"{system}/{frame}: finding {n_sym} HCT symmetries", flush=True)
        symmetries, epsilons = HCT(
            data["hamiltonian"],
            n_sym=n_sym,
            sym_metric_func=singleton_metric,
            use_coeffs_eps=True,
            verbose=False,
        )
        clifford = Clifford.from_symmetries(
            symmetries,
            n_qubits=data["n_qubits"],
            symmetry_qubits_first=True,
            synthesis_basis="Z",
            generator_mapping="positive_z",
        )
        transformed_hamiltonian = clifford.transform(data["hamiltonian"])

        rows = [row for row in rows if row.get("frame") != frame]
        frame_rows, summary = run_block2_qubit_dmrg_curve(
            label=frame,
            hamiltonian=transformed_hamiltonian,
            sparse_state=(
                data["cisd_state"].indices,
                data["cisd_state"].coeffs,
            ),
            exact_energy=data["fci_energy"],
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
            unitaries=(clifford,),
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
            row.update(
                {
                    "system": system,
                    "symmetry_count": count_label,
                    "n_symmetries": n_sym,
                    "reference_energy": data["fci_energy"],
                }
            )
        summary.update(
            {
                "symmetry_count": count_label,
                "n_symmetries": n_sym,
                "symmetries": symmetry_strings(symmetries),
                "hct_epsilons": epsilons,
                "clifford": clifford.to_dict(),
                "benchmark_reference": {
                    "method": "saved_fci",
                    "energy": data["fci_energy"],
                },
            }
        )
        rows.extend(frame_rows)
        frames[frame] = summary
        save_system_outputs(system_dir, rows, frames, metadata)
        del transformed_hamiltonian, clifford, symmetries
        gc.collect()

    save_system_outputs(system_dir, rows, frames, metadata)
    return {**metadata, "frames": frames}, rows


def aggregate(args) -> None:
    systems = {}
    all_rows = []
    summary_rows = []
    for system in args.systems:
        system_dir = args.output_dir / system
        benchmark_path = system_dir / "benchmark.json"
        if not benchmark_path.exists():
            continue
        result = load_json(benchmark_path)
        systems[system] = result
        curve_path = system_dir / "dmrg_curves.csv"
        if curve_path.exists():
            from quasisymmetries.save import read_csv

            all_rows.extend(read_csv(curve_path))
        for frame, summary in result.get("frames", {}).items():
            summary_rows.append(
                {
                    "system": system,
                    "distance_angstrom": result.get("distance_angstrom"),
                    "frame": frame,
                    "symmetry_count": summary.get("symmetry_count"),
                    "n_symmetries": summary.get("n_symmetries"),
                    "first_converged_bond_dim": summary.get(
                        "first_converged_bond_dim"
                    ),
                    "converged_within_grid": summary.get(
                        "converged_within_grid"
                    ),
                    "mpo_bond_dimension": summary.get(
                        "mpo_bond_dimension"
                    ),
                }
            )
    save_json(
        args.output_dir / "benchmark.json",
        {
            "settings": {
                "systems": args.systems,
                "counts": args.counts,
                "bond_dims": args.bond_dims,
                "dmrg_sweeps": args.dmrg_sweeps,
                "dmrg_tolerance": args.dmrg_tolerance,
                "sweep_tolerance": args.sweep_tolerance,
                "warm_start_noises": args.warm_start_noises,
            },
            "systems": systems,
        },
    )
    if all_rows:
        write_csv(args.output_dir / "dmrg_curves.csv", all_rows)
    if summary_rows:
        write_csv(args.output_dir / "dmrg_summary.csv", summary_rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--counts", nargs="+", choices=("N/2", "N"), default=["N/2", "N"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-curve", action="store_true")
    parser.add_argument("--bond-dims", type=int, nargs="+", default=list(DEFAULT_BOND_DIMS))
    parser.add_argument("--dmrg-sweeps", type=int, default=100)
    parser.add_argument("--dmrg-tolerance", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--sweep-tolerance", type=float, default=1e-6)
    parser.add_argument("--davidson-threshold", type=float, default=1e-10)
    parser.add_argument("--warm-start-noises", type=float, nargs="*", default=list(DEFAULT_NOISES))
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument("--mps-cutoff", type=float, default=1e-13)
    parser.add_argument("--mpo-builder", choices=("blocked_sum", "expression"), default="blocked_sum")
    parser.add_argument("--sum-mpo-mod", type=int, default=20)
    parser.add_argument("--sparse-batch-size", type=int, default=32)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=0.5)
    parser.add_argument("--save-tensor-networks", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if any(bond_dim < 1 for bond_dim in args.bond_dims):
        raise ValueError("Bond dimensions must be positive")
    if any(noise < 0 for noise in args.warm_start_noises):
        raise ValueError("Warm-start noises must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        run_system(system, args)
        aggregate(args)
    aggregate(args)
    print(f"Saved new-HCT N2 benchmarks to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
