"""Compare X- and Z-native Clifford synthesis on saved beam symmetries.

The symmetry search is deliberately not repeated. This script loads the
MAY27 ``BS N/2 Comm`` and ``BS N Comm`` symmetry sets for the correlated H2O
and frozen-core N2 geometries, then benchmarks both Clifford conventions.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from openfermion import count_qubits, jordan_wigner

from quasisymmetries.benchmark import BenchmarkData, benchmark_syms


SYSTEMS = ("H2O_corr", "N2frozen_corr")
SYMMETRY_TAGS = ("BS N/2 Comm", "BS N Comm")
SYNTHESIS_BASES = ("X", "Z")


def _load_saved_beam_symmetries(results_dir: Path, system: str):
    path = results_dir / f"_nc_exp_cisd_MAY27{system}_datasets.pkl"
    datasets = BenchmarkData.load_datasets(path)
    by_tag = {dataset.tag: dataset for dataset in datasets}
    missing = [tag for tag in SYMMETRY_TAGS if tag not in by_tag]
    if missing:
        raise ValueError(f"{path} is missing symmetry datasets: {missing}")
    return [by_tag[tag] for tag in SYMMETRY_TAGS]


def run_comparison(
    hamiltonian_dir: Path,
    saved_results_dir: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "clifford_route_comparison.txt"
    text_path.write_text("", encoding="utf-8")
    all_results = []
    rows = []

    for system in SYSTEMS:
        with (hamiltonian_dir / f"{system}.pkl").open("rb") as file_obj:
            # These are existing repository-generated, trusted input files.
            H, fci_e, fci_gs, cisd_e, cisd_gs = pickle.load(file_obj)

        HQ = jordan_wigner(H)
        n_qubits = count_qubits(HQ)
        saved_symmetry_sets = _load_saved_beam_symmetries(
            saved_results_dir,
            system,
        )

        for saved_data in saved_symmetry_sets:
            for synthesis_basis in SYNTHESIS_BASES:
                tag = (
                    f"{system} | {saved_data.tag} | "
                    f"{synthesis_basis}-native"
                )
                result, processed = benchmark_syms(
                    saved_data.symmetries,
                    HQ,
                    fci_gs,
                    fci_e,
                    n_qubits,
                    N_2_sym=False,
                    verbose=False,
                    print_to_file=text_path,
                    tag=tag,
                    return_processed_data=True,
                    log_base=np.e,
                    synthesis_basis=synthesis_basis,
                )
                all_results.append(result)

                clifford = processed["clifford"]
                gates = clifford.factor_descriptions
                n_cnot = sum(gate.startswith("CNOT") for gate in gates)
                rows.append(
                    {
                        "system": system,
                        "symmetry_tag": saved_data.tag,
                        "synthesis_basis": synthesis_basis,
                        "num_symmetries": len(saved_data.symmetries),
                        "num_gates": len(gates),
                        "num_cnot": n_cnot,
                        "num_single_qubit_gates": len(gates) - n_cnot,
                        "max_cut_entropy": max(result.cut_entropies),
                        "sum_cut_entropy": sum(result.cut_entropies),
                        "cut_entropies": json.dumps(result.cut_entropies),
                        "dmrg_bd": result.dmrg_bd,
                    }
                )

    BenchmarkData.save_datasets(
        all_results,
        output_dir / "clifford_route_comparison",
    )
    pd.DataFrame(rows).to_csv(
        output_dir / "clifford_route_comparison.csv",
        index=False,
    )
    BenchmarkData.plot_cut_entropies(
        all_results,
        filename=output_dir / "clifford_route_cut_entropies.png",
    )
    return all_results, rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hamiltonian-dir",
        type=Path,
        default=Path("saved/hamiltonians"),
    )
    parser.add_argument(
        "--saved-results-dir",
        type=Path,
        default=Path("saved/results/MAY27"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saved/results/JUL04/clifford_routes"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_comparison(
        args.hamiltonian_dir,
        args.saved_results_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
