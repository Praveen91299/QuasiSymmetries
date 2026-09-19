#!/usr/bin/env python3
"""Plot raw-qubit DMRG entanglement for the three N2 orbital bases.

The script reads the first chemically accurate, sweep-converged MPS saved by
``benchmark_sto3g_orbital_dmrg.py``. It contracts each native Block2 MPS
to obtain the von Neumann entropy across every qubit bond. It also plots the
saved raw-qubit energy errors so a shared first accepted grid point can be
interpreted together with the entanglement profiles.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import _bootstrap  # noqa: F401

from quasisymmetries.block2_qubit_benchmark import load_block2_mps
from quasisymmetries.save import save_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "saved" / "results" / "n2_sto3g_localized_dmrg"
ORBITAL_BASES = ("canonical", "split_pm", "split_pm_irrep")
DISPLAY_NAMES = {
    "canonical": "Canonical",
    "split_pm": "Split PM",
    "split_pm_irrep": "Split PM within D2h irreps",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return its rows as string-valued dictionaries.

    Parameters
    ----------
    path
        CSV file to read.

    Returns
    -------
    rows
        Rows using the column names from the CSV header.
    """
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def find_saved_raw_qubit_mps(
    benchmark_dir: Path,
    orbital_basis: str,
    bond_dimension: int,
) -> Path:
    """Return the saved first-accepted raw-qubit MPS directory.

    Parameters
    ----------
    benchmark_dir
        Root output of ``benchmark_sto3g_orbital_dmrg.py``.
    orbital_basis
        One of ``canonical``, ``split_pm``, or ``split_pm_irrep``.
    bond_dimension
        First accepted bond dimension reported in ``dmrg_summary.csv``.

    Returns
    -------
    mps_directory
        Native Block2 restart directory for the requested MPS.
    """
    mps_directory = (
        benchmark_dir
        / orbital_basis
        / "tensor_networks"
        / "raw_qubit"
        / (
            "raw_qubit_first_chemical_accuracy_"
            f"bd{bond_dimension}_mps.block2"
        )
    )
    if not (mps_directory / "manifest.json").is_file():
        raise FileNotFoundError(mps_directory)
    return mps_directory


def block_entropies_from_saved_mps(
    mps_directory: Path,
    *,
    n_qubits: int,
    entropy_base: float = 2.0,
    n_threads: int = 1,
    n_mkl_threads: int = 1,
    stack_mem_gb: float = 1.0,
) -> np.ndarray:
    """Contract a saved Block2 Pauli MPS and return all cut entropies.

    Parameters
    ----------
    mps_directory
        Restart directory written by ``save_block2_mps``.
    n_qubits
        Number of spin-orbital qubits/sites in the saved MPS.
    entropy_base
        Logarithm base for the von Neumann entropy. The default, 2, reports
        entropy in bits.
    n_threads, n_mkl_threads
        Block2 and BLAS thread counts used for the identity-MPO contraction.
    stack_mem_gb
        Block2 stack-memory allowance in GiB.

    Returns
    -------
    entropies
        Length ``n_qubits - 1`` array. Element ``i`` is the entropy across the
        cut between zero-based qubit indices ``i`` and ``i + 1``.
    """
    if entropy_base <= 0.0 or np.isclose(entropy_base, 1.0):
        raise ValueError("entropy_base must be positive and different from 1")
    with tempfile.TemporaryDirectory(prefix="n2_raw_qubit_entropy_") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SGB,
            n_threads=int(n_threads),
            n_mkl_threads=int(n_mkl_threads),
            stack_mem=int(float(stack_mem_gb) * 1024**3),
            compressed_mps_storage=True,
        )
        driver.initialize_system(n_sites=int(n_qubits), pauli_mode=True)
        try:
            mps = load_block2_mps(driver, mps_directory)
            entropy_natural_log = np.asarray(
                driver.get_bipartite_entanglement(mps), dtype=float
            )
        finally:
            driver.finalize()
    if entropy_natural_log.shape != (n_qubits - 1,):
        raise RuntimeError(
            f"Expected {n_qubits - 1} cut entropies, got "
            f"{entropy_natural_log.shape}"
        )
    return entropy_natural_log / np.log(float(entropy_base))


def orbital_rotation_diagnostics(orbitals_file: Path) -> dict[str, float]:
    """Measure how strongly a localized basis mixes canonical orbitals.

    Parameters
    ----------
    orbitals_file
        ``orbitals.npz`` written by the localized-orbital benchmark.

    Returns
    -------
    diagnostics
        Frobenius norm and largest magnitude of the off-diagonal elements of
        the canonical-to-target orbital rotation, plus the mean canonical-
        orbital participation ratio of its columns. A participation ratio of
        one means no mixing of canonical orbitals.
    """
    with np.load(orbitals_file, allow_pickle=False) as archive:
        rotation = np.asarray(
            archive["canonical_to_basis_rotation"], dtype=float
        )
    off_diagonal = rotation - np.diag(np.diag(rotation))
    column_weights = np.abs(rotation) ** 2
    participation = 1.0 / np.sum(column_weights**2, axis=0)
    return {
        "rotation_off_diagonal_frobenius_norm": float(
            np.linalg.norm(off_diagonal)
        ),
        "rotation_max_abs_off_diagonal": float(
            np.max(np.abs(off_diagonal))
        ),
        "mean_canonical_orbital_participation_ratio": float(
            np.mean(participation)
        ),
        "max_canonical_orbital_participation_ratio": float(
            np.max(participation)
        ),
    }


def plot_entanglement(
    entropy_by_basis: dict[str, np.ndarray],
    output_path: Path,
    *,
    entropy_base: float,
) -> None:
    """Plot bond-resolved entropies and save the figure.

    Parameters
    ----------
    entropy_by_basis
        Mapping from orbital-basis name to cut-entropy array.
    output_path
        Destination PNG path.
    entropy_base
        Logarithm base used for the supplied entropies.

    Returns
    -------
    None
    """
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for orbital_basis in ORBITAL_BASES:
        entropies = entropy_by_basis[orbital_basis]
        cuts = np.arange(1, len(entropies) + 1)
        axis.plot(
            cuts,
            entropies,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=DISPLAY_NAMES[orbital_basis],
        )
    axis.set_xlabel("Bond cut (number of qubits on the left)")
    unit = "bits" if np.isclose(entropy_base, 2.0) else f"log base {entropy_base:g}"
    axis.set_ylabel(f"Von Neumann entropy ({unit})")
    axis.set_title("N$_2$ raw-qubit DMRG entanglement, $R=1.4$ Å")
    axis.set_xticks(np.arange(1, 20))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_energy_convergence(curve_rows: list[dict[str, str]], output_path: Path) -> None:
    """Plot raw-qubit absolute energy error against tested bond dimension.

    Parameters
    ----------
    curve_rows
        Rows from the aggregate ``dmrg_curves.csv``.
    output_path
        Destination PNG path.

    Returns
    -------
    None
    """
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for orbital_basis in ORBITAL_BASES:
        selected = [
            row
            for row in curve_rows
            if row["orbital_basis"] == orbital_basis
            and row["frame"] == "raw_qubit"
        ]
        selected.sort(key=lambda row: int(row["bond_dim"]))
        axis.plot(
            [int(row["bond_dim"]) for row in selected],
            [float(row["abs_energy_error"]) for row in selected],
            marker="o",
            linewidth=1.8,
            label=DISPLAY_NAMES[orbital_basis],
        )
    axis.axhline(1.6e-3, color="black", linestyle="--", linewidth=1.2,
                 label="Chemical accuracy")
    axis.set_yscale("log")
    axis.set_xlabel("Maximum MPS bond dimension")
    axis.set_ylabel("Absolute energy error (Ha)")
    axis.set_title("N$_2$ raw-qubit DMRG convergence, $R=1.4$ Å")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    """Load the three saved MPSs, analyze them, and write tables and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--entropy-base", type=float, default=2.0)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--n-mkl-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "raw_qubit_entanglement"
    )
    output_files = (
        output_dir / "bond_entropies.csv",
        output_dir / "entanglement_summary.json",
        output_dir / "bond_entanglement.png",
        output_dir / "energy_convergence.png",
    )
    if not args.force:
        existing = [path for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(
                f"Analysis outputs already exist: {existing}; pass --force to replace"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv_rows(input_dir / "dmrg_summary.csv")
    curve_rows = read_csv_rows(input_dir / "dmrg_curves.csv")
    n_qubits = 20
    entropy_by_basis = {}
    entropy_rows = []
    analysis_summary = {
        "input_directory": str(input_dir),
        "entropy_base": float(args.entropy_base),
        "n_qubits": n_qubits,
        "orbital_bases": {},
    }

    for orbital_basis in ORBITAL_BASES:
        matches = [
            row
            for row in summary_rows
            if row["orbital_basis"] == orbital_basis
            and row["frame"] == "raw_qubit"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one raw-qubit summary for {orbital_basis}, got {len(matches)}"
            )
        bond_dimension = int(matches[0]["first_converged_bond_dim"])
        mps_directory = find_saved_raw_qubit_mps(
            input_dir, orbital_basis, bond_dimension
        )
        entropies = block_entropies_from_saved_mps(
            mps_directory,
            n_qubits=n_qubits,
            entropy_base=args.entropy_base,
            n_threads=args.n_threads,
            n_mkl_threads=args.n_mkl_threads,
            stack_mem_gb=args.stack_mem_gb,
        )
        entropy_by_basis[orbital_basis] = entropies
        for cut, entropy in enumerate(entropies, start=1):
            entropy_rows.append(
                {
                    "orbital_basis": orbital_basis,
                    "first_converged_bond_dim": bond_dimension,
                    "bond_cut": cut,
                    "cut_after_qubit_index": cut - 1,
                    "left_qubits": cut,
                    "right_qubits": n_qubits - cut,
                    "von_neumann_entropy": float(entropy),
                    "entropy_base": float(args.entropy_base),
                }
            )
        raw_curve = [
            row
            for row in curve_rows
            if row["orbital_basis"] == orbital_basis
            and row["frame"] == "raw_qubit"
        ]
        accepted_row = next(
            row for row in raw_curve if int(row["bond_dim"]) == bond_dimension
        )
        analysis_summary["orbital_bases"][orbital_basis] = {
            "display_name": DISPLAY_NAMES[orbital_basis],
            "mps_directory": str(mps_directory),
            "first_converged_bond_dim": bond_dimension,
            "accepted_energy": float(accepted_row["energy"]),
            "accepted_abs_energy_error_hartree": float(
                accepted_row["abs_energy_error"]
            ),
            "maximum_entropy": float(np.max(entropies)),
            "maximum_entropy_bond_cut": int(np.argmax(entropies) + 1),
            "maximum_entropy_cut_after_qubit_index": int(
                np.argmax(entropies)
            ),
            "central_cut_entropy": float(entropies[n_qubits // 2 - 1]),
            "sum_of_cut_entropies": float(np.sum(entropies)),
            **orbital_rotation_diagnostics(
                input_dir / orbital_basis / "prepared" / "orbitals.npz"
            ),
        }

    write_csv(output_files[0], entropy_rows)
    save_json(output_files[1], analysis_summary)
    plot_entanglement(
        entropy_by_basis, output_files[2], entropy_base=args.entropy_base
    )
    plot_energy_convergence(curve_rows, output_files[3])

    for orbital_basis in ORBITAL_BASES:
        result = analysis_summary["orbital_bases"][orbital_basis]
        print(
            f"{DISPLAY_NAMES[orbital_basis]:28s} "
            f"BD={result['first_converged_bond_dim']:3d} "
            f"max S={result['maximum_entropy']:.8f} at cut "
            f"{result['maximum_entropy_bond_cut']:2d}; "
            f"central S={result['central_cut_entropy']:.8f}; "
            f"|dE|={result['accepted_abs_energy_error_hartree']:.3e} Ha"
        )
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
