"""Convert detailed pyblock2 DMRG curves into a compact comparison CSV.

Examples
--------
Summarize every completed or partial system under a benchmark directory::

    python scripts/summarize_dmrg_curves.py \
        saved/results/pyblock2_16_systems

Summarize one curve file explicitly::

    python scripts/summarize_dmrg_curves.py \
        saved/results/pyblock2_16_systems/H4chain_diss/dmrg_curves.partial.csv \
        --output h4chain_diss_dmrg_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

MINIMAL_COLUMNS = (
    "system",
    "benchmark",
    "fiedler",
    "backend",
    "within_chemical_accuracy",
    "converged_dmrg_bond_dimension",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a curve CSV without importing the benchmark dependencies."""
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a rectangular summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


def optional_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def optional_int(value):
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def discover_curve_files(inputs) -> list[Path]:
    """Resolve CSV inputs, preferring final over partial files per system."""
    files = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)

        system_files = []
        for child in sorted(item for item in path.iterdir() if item.is_dir()):
            final_path = child / "dmrg_curves.csv"
            partial_path = child / "dmrg_curves.partial.csv"
            if final_path.exists():
                system_files.append(final_path)
            elif partial_path.exists():
                system_files.append(partial_path)
        if system_files:
            files.extend(system_files)
            continue

        final_path = path / "dmrg_curves.csv"
        partial_path = path / "dmrg_curves.partial.csv"
        if final_path.exists():
            files.append(final_path)
        elif partial_path.exists():
            files.append(partial_path)
        else:
            raise FileNotFoundError(
                f"No dmrg_curves.csv or dmrg_curves.partial.csv under {path}"
            )

    # Preserve caller/discovery order while removing duplicate paths.
    return list(OrderedDict((path, None) for path in files))


def infer_system(row: dict, source: Path, override: str | None) -> str:
    if override:
        return override
    system = str(row.get("system", "")).strip()
    if system:
        return system
    if source.parent.name not in {"", "results", "saved"}:
        return source.parent.name
    return "unknown"


def benchmark_name(frame: str, symmetry_set: str, fiedler: bool) -> str:
    if frame == "raw_qubit":
        return "Original Qubit"
    if frame == "raw_fermionic_su2":
        return "Original Fermionic"
    label = symmetry_set or frame
    label = label.replace(" Comm", "")
    if fiedler and "Fiedler" not in label:
        label += " + Fiedler"
    return label


def summarize_curve_rows(
    sourced_rows: list[tuple[Path, dict]],
    *,
    system_override: str | None = None,
) -> list[dict]:
    """Return one summary row per system/backend/frame."""
    groups = OrderedDict()
    for order, (source, raw_row) in enumerate(sourced_rows):
        row = dict(raw_row)
        system = infer_system(row, source, system_override)
        frame = str(row.get("frame", "")).strip()
        backend = str(row.get("backend", "")).strip()
        if not backend and frame == "raw_fermionic_su2":
            backend = "block2_su2"
        symmetry_set = str(row.get("dataset_tag", "")).strip()
        fiedler = parse_bool(row.get("fiedler", False))
        key = (system, frame, backend, symmetry_set, fiedler)
        row["_source"] = str(source)
        row["_order"] = order
        groups.setdefault(key, []).append(row)

    summaries = []
    for (
        system,
        frame,
        backend,
        symmetry_set,
        fiedler,
    ), rows in groups.items():
        # A resumed file can contain a repeated bond dimension. Retain the
        # latest occurrence while preserving the benchmarked order.
        latest_by_bond = {}
        for row in rows:
            bond_dim = optional_int(row.get("bond_dim"))
            latest_by_bond[bond_dim] = row
        curve = sorted(
            latest_by_bond.values(),
            key=lambda row: optional_int(row.get("bond_dim")),
        )

        converged_rows = [
            row
            for row in curve
            if parse_bool(row.get("within_dmrg_tolerance", False))
        ]
        converged = converged_rows[0] if converged_rows else None
        best = min(
            curve,
            key=lambda row: (
                optional_float(row.get("abs_energy_error"))
                if optional_float(row.get("abs_energy_error")) is not None
                else float("inf")
            ),
        )
        final = curve[-1]
        selected = converged if converged is not None else best

        summaries.append(
            {
                "system": system,
                "benchmark": benchmark_name(
                    frame, symmetry_set, fiedler
                ),
                "fiedler": fiedler,
                "backend": backend,
                "within_chemical_accuracy": converged is not None,
                "converged_dmrg_bond_dimension": (
                    optional_int(converged.get("bond_dim"))
                    if converged is not None
                    else None
                ),
                "converged_energy": (
                    optional_float(converged.get("energy"))
                    if converged is not None
                    else None
                ),
                "converged_abs_energy_error": (
                    optional_float(converged.get("abs_energy_error"))
                    if converged is not None
                    else None
                ),
                "converged_dmrg_seconds": (
                    optional_float(converged.get("dmrg_seconds"))
                    if converged is not None
                    else None
                ),
                "sweep_converged_at_selected_bond": parse_bool(
                    selected.get("sweep_converged", False)
                ),
                "best_tested_bond_dimension": optional_int(
                    best.get("bond_dim")
                ),
                "best_tested_energy": optional_float(best.get("energy")),
                "best_abs_energy_error": optional_float(
                    best.get("abs_energy_error")
                ),
                "largest_tested_bond_dimension": max(
                    optional_int(row.get("bond_dim")) for row in curve
                ),
                "number_of_tested_bond_dimensions": len(curve),
                "last_tested_bond_dimension": optional_int(
                    final.get("bond_dim")
                ),
                "source_curve_csv": ";".join(
                    OrderedDict(
                        (row["_source"], None) for row in curve
                    )
                ),
            }
        )
    return summaries


def default_output_path(inputs, files) -> Path:
    if len(inputs) == 1 and Path(inputs[0]).expanduser().is_dir():
        return Path(inputs[0]).expanduser().resolve() / "dmrg_summary.csv"
    if len(files) == 1:
        return files[0].with_name(f"{files[0].stem}_summary.csv")
    return Path.cwd() / "dmrg_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="curve CSV file(s), or benchmark directories to scan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output CSV path; a nearby dmrg_summary.csv is used by default",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="system-name override for an input CSV lacking a system column",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help=(
            "write only system, benchmark, Fiedler, backend, "
            "convergence, and converged bond dimension"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = discover_curve_files(args.inputs)
    sourced_rows = [
        (path, row) for path in files for row in read_csv(path)
    ]
    if not sourced_rows:
        raise ValueError("The selected curve CSV files contain no rows.")
    summaries = summarize_curve_rows(
        sourced_rows, system_override=args.system
    )
    if args.minimal:
        summaries = [
            {column: row[column] for column in MINIMAL_COLUMNS}
            for row in summaries
        ]
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(args.inputs, files)
    )
    write_csv(output, summaries)
    print(
        f"Wrote {len(summaries)} DMRG frame summaries from "
        f"{len(files)} curve file(s) to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
