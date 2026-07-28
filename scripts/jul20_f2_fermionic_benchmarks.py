"""Run fermionic pyblock2 MPO/MPS bond-dimension benchmarks for JUL20 BeH2."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import test_mpo_ferm as fermionic  # noqa: E402


SYSTEMS = ["BeH2_eqm", "BeH2_corr", "BeH2_diss"]


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=SYSTEMS)
    parser.add_argument(
        "--ham-dir",
        type=Path,
        default=ROOT / "saved" / "results" / "Jul20" / "hamiltonians",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT
        / "saved"
        / "results"
        / "Jul20"
        / "jul20_beh2_fermionic_mpo_mps_bond_dimensions.csv",
    )
    parser.add_argument("--n-sweeps", type=int, default=fermionic.N_SWEEPS)
    return parser.parse_args()


def main():
    args = parse_args()

    fermionic.HAMILTONIAN_DIRECTORY = args.ham_dir
    fermionic.N_SWEEPS = args.n_sweeps

    rows = []
    for system in args.systems:
        try:
            row = fermionic.benchmark_system(system)
        except Exception as exc:
            print(f"{system}: FAILED: {exc}")
            row = {
                "system": system,
                "converged": False,
                "energy_tolerance": fermionic.ENERGY_TOLERANCE,
            }
        rows.append(row)
        write_results(args.output_csv, rows)
        print("Saved results to:", args.output_csv)

    print("\nCompleted BeH2 fermionic benchmarks:", args.output_csv)


if __name__ == "__main__":
    main()
