"""Run BS(n) benchmarks with state-agnostic beam-search objectives.

This script compares two beam-search objectives for rank-n Pauli generators:

1. ``commuting_weight``:
   The default beam-search score, i.e. the total Hamiltonian coefficient
   1-norm retained by terms commuting with the whole candidate set.

2. ``summed_single_commuting_weight``:
   A separable state-agnostic score.  Each candidate generator is scored by
   the Hamiltonian coefficient 1-norm of terms commuting with that generator
   alone, and a candidate set is scored by the sum of these singleton scores.

For each system and objective, the script runs BS(n), benchmarks the resulting
symmetries with the standard ``BenchmarkData``/quimb DMRG path, and saves:

- a human-readable text log,
- portable JSON benchmark data plus search diagnostics,
- a CSV of DMRG convergence bond dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from openfermion import count_qubits, jordan_wigner


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quasisymmetries.benchmark import BenchmarkData, benchmark_syms
from quasisymmetries.bs.beam import (
    BeamSearch_Symmetries,
    validate_symmetry_generators,
)
from quasisymmetries.bs.utils import qubit_operator_terms, qubitops_to_masks


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


OBJECTIVES = {
    "commuting_weight": {
        "description": "Default retained Hamiltonian coefficient 1-norm commuting with the full generator set.",
        "score_is_separable": False,
    },
    "summed_single_commuting_weight": {
        "description": "Sum over generators of the Hamiltonian coefficient 1-norm commuting with each generator individually.",
        "score_is_separable": True,
    },
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        json.dump(jsonable(payload), file_obj, indent=2, allow_nan=False)
        file_obj.write("\n")


def symmetry_strings(symmetries) -> list[str]:
    return [str(symmetry) for symmetry in symmetries]


def make_singleton_commuting_weight_score(HQ, n_qubits: int):
    """Return score_func([sym]) = weight of terms commuting with sym."""
    _, terms = qubit_operator_terms(HQ, n_qubits=n_qubits)

    def score_func(symmetries):
        masks = qubitops_to_masks(symmetries, n_qubits)
        total = 0.0
        for term in terms:
            if all(
                ((int.bit_count(term.mask[0] & mask[1]) + int.bit_count(term.mask[1] & mask[0])) & 1) == 0
                for mask in masks
            ):
                total += term.abs_coeff
        return total

    return score_func


def load_hamiltonian_data(ham_dir: Path, system: str):
    with (ham_dir / f"{system}.pkl").open("rb") as file_obj:
        return pickle.load(file_obj)


def run_one_objective(
    *,
    objective: str,
    HQ,
    fci_gs,
    fci_e,
    n_qubits: int,
    args,
    text_path: Path,
):
    if objective == "commuting_weight":
        score_func = None
    elif objective == "summed_single_commuting_weight":
        score_func = make_singleton_commuting_weight_score(HQ, n_qubits)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    target_rank = n_qubits
    search_t0 = perf_counter()
    symmetries = BeamSearch_Symmetries(
        HQ,
        target_rank=target_rank,
        n_qubits=n_qubits,
        beam_width=args.beam_width,
        heavy_core_fraction=args.heavy_core_fraction,
        max_candidates_from_terms=args.max_candidates_from_terms,
        include_pairwise_products=args.include_pairwise_products,
        pairwise_seed_terms=args.pairwise_seed_terms,
        seed_with_exact_symmetries=args.seed_with_exact_symmetries,
        score_func=score_func,
        score_is_separable=OBJECTIVES[objective]["score_is_separable"],
        include_hct_symmetries=args.include_hct_symmetries,
        hct_n_sym=n_qubits,
        hct_use_coeffs_eps=True,
        do_local_refine=not args.no_local_refine,
        local_refine_passes=args.local_refine_passes,
        n_processes=args.n_processes,
        mp_start_method=args.mp_start_method,
    )
    search_seconds = perf_counter() - search_t0

    diagnostics = validate_symmetry_generators(
        HQ,
        symmetries,
        n_qubits=n_qubits,
    )
    diagnostics["requested_target_rank"] = target_rank
    diagnostics["search_seconds"] = search_seconds

    with text_path.open("a") as file_obj:
        print("\n" + "-" * 80, file=file_obj)
        print(f"Objective: {objective}", file=file_obj)
        print(OBJECTIVES[objective]["description"], file=file_obj)
        print(f"Search seconds: {search_seconds:.6f}", file=file_obj)
        print("Search diagnostics:", file=file_obj)
        for key, value in diagnostics.items():
            print(f"  {key}: {value}", file=file_obj)
        print("Symmetries found:", file=file_obj)
        for symmetry in symmetries:
            print(f"  {symmetry}", file=file_obj)

    if args.skip_dmrg:
        data = None
    else:
        data = benchmark_syms(
            symmetries,
            HQ,
            fci_gs,
            fci_e,
            n_qubits,
            N_2_sym=False,
            print_to_file=str(text_path),
            tag=f"BS N {objective}",
            verbose=False,
            synthesis_basis=args.synthesis_basis,
            generator_mapping=args.generator_mapping,
        )

    return symmetries, diagnostics, data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BS(n) state-agnostic cost-function benchmarks."
    )
    parser.add_argument("--ham-dir", type=Path, default=ROOT / "saved" / "hamiltonians")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "saved" / "results" / "JUL09")
    parser.add_argument("--systems", nargs="+", default=SYSTEMS)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=list(OBJECTIVES),
        default=list(OBJECTIVES),
    )
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--heavy-core-fraction", type=float, default=0.95)
    parser.add_argument("--max-candidates-from-terms", type=int, default=256)
    parser.add_argument("--pairwise-seed-terms", type=int, default=12)
    parser.add_argument("--include-pairwise-products", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-hct-symmetries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed-with-exact-symmetries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-local-refine", action="store_true")
    parser.add_argument("--local-refine-passes", type=int, default=10)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mp-start-method", default=None)
    parser.add_argument("--skip-dmrg", action="store_true")
    parser.add_argument("--synthesis-basis", choices=["X", "Z"], default="Z")
    parser.add_argument(
        "--generator-mapping",
        choices=["row_reduced", "positive_z"],
        default="positive_z",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_path = args.output_dir / "state_agnostic_bs_benchmarks.txt"
    datasets_path = args.output_dir / "state_agnostic_bs_benchmark_datasets"
    metrics_path = args.output_dir / "state_agnostic_bs_symmetries_metrics.json"
    bd_csv_path = args.output_dir / "state_agnostic_bs_dmrg_bond_dimensions.csv"

    all_datasets = []
    all_metrics: dict[str, dict] = {}
    bd_rows: list[dict] = []

    with text_path.open("w") as file_obj:
        print("State-agnostic BS(n) benchmarks", file=file_obj)
        print(f"Systems: {args.systems}", file=file_obj)
        print(f"Objectives: {args.objectives}", file=file_obj)
        print(f"Beam width: {args.beam_width}", file=file_obj)
        print(f"DMRG skipped: {args.skip_dmrg}", file=file_obj)
        print(f"Clifford synthesis basis: {args.synthesis_basis}", file=file_obj)
        print(f"Clifford generator mapping: {args.generator_mapping}", file=file_obj)

    for system in args.systems:
        print(f"Starting system: {system}", flush=True)
        H, fci_e, fci_gs, cisd_e, cisd_gs = load_hamiltonian_data(args.ham_dir, system)
        HQ = jordan_wigner(H)
        n_qubits = count_qubits(HQ)

        all_metrics[system] = {
            "n_qubits": n_qubits,
            "fci_energy": fci_e,
            "cisd_energy": cisd_e,
            "objectives": {},
        }
        bd_row = {"system": system}

        with text_path.open("a") as file_obj:
            print("\n\n" + "=" * 80, file=file_obj)
            print(system, file=file_obj)
            print("=" * 80, file=file_obj)
            print(f"n_qubits: {n_qubits}", file=file_obj)
            print(f"fci_energy: {fci_e}", file=file_obj)
            print(f"cisd_energy: {cisd_e}", file=file_obj)

        for objective in args.objectives:
            symmetries, diagnostics, data = run_one_objective(
                objective=objective,
                HQ=HQ,
                fci_gs=fci_gs,
                fci_e=fci_e,
                n_qubits=n_qubits,
                args=args,
                text_path=text_path,
            )

            all_metrics[system]["objectives"][objective] = {
                "description": OBJECTIVES[objective]["description"],
                "symmetries": symmetry_strings(symmetries),
                "search_diagnostics": diagnostics,
            }

            if data is not None:
                all_datasets.append(data)
                all_metrics[system]["objectives"][objective]["benchmark"] = data._to_json_dict()
                bd_row[objective] = data.dmrg_bd
                bd_row[f"{objective}_commuting_terms"] = data.num_commuting_terms
                bd_row[f"{objective}_non_commuting_l1"] = data.non_commuting_l1

            save_json(metrics_path, all_metrics)
            if all_datasets:
                BenchmarkData.save_datasets(all_datasets, datasets_path)

        bd_rows.append(bd_row)
        write_csv(bd_csv_path, bd_rows)

    save_json(metrics_path, all_metrics)
    if all_datasets:
        BenchmarkData.save_datasets(all_datasets, datasets_path)
    write_csv(bd_csv_path, bd_rows)
    print(f"Done. Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
