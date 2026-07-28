#!/usr/bin/env python3
"""Plot qubit mutual-information graphs of saved molecular wavefunctions."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quasisymmetries.fiedler import (  # noqa: E402
    fiedler_order_from_weights,
    infer_n_qubits_from_state,
    qubit_mutual_information_matrix_sparse_bloch,
)
from quasisymmetries.state_utils import SparseQubitState  # noqa: E402


HAMILTONIAN_DIR = PROJECT_ROOT / "saved" / "hamiltonians"
DEFAULT_SYSTEM = "H2O_corr"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "saved" / "results" / "mutual_information_graphs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load CISD/FCI states from a saved Hamiltonian tuple, import them "
            "as SparseQubitState objects, and plot their qubit-site "
            "mutual-information graphs."
        )
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=(
            "Saved system name, without the .pkl extension "
            f"(default: {DEFAULT_SYSTEM})."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Optional explicit pickle path containing "
            "(H, fci_e, fci_gs, cisd_e, cisd_gs). This overrides --system."
        ),
    )
    parser.add_argument(
        "--state",
        choices=("fci", "cisd", "both"),
        default="both",
        help="Which saved wavefunction(s) to plot (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the PNG and CSV outputs.",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=0.0,
        help=(
            "Drop amplitudes with absolute value <= this threshold when "
            "constructing SparseQubitState objects."
        ),
    )
    parser.add_argument(
        "--edge-tol",
        type=float,
        default=1.0e-12,
        help="Only mutual-information values above this tolerance are drawn.",
    )
    parser.add_argument(
        "--label-tol",
        type=float,
        default=5.0e-2,
        help="Label edges with mutual information at least this large.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_saved_states(path: Path) -> dict[str, tuple[float, np.ndarray]]:
    """Load CISD and FCI energies/states using the saved tuple format."""
    with path.open("rb") as file_obj:
        data = pickle.load(file_obj)

    if not isinstance(data, tuple) or len(data) != 5:
        raise ValueError(
            f"Expected a five-item saved Hamiltonian tuple in {path}, "
            f"received {type(data).__name__}."
        )

    _, fci_energy, fci_state, cisd_energy, cisd_state = data
    return {
        "fci": (
            float(np.real(fci_energy)),
            np.asarray(fci_state, dtype=complex).reshape(-1),
        ),
        "cisd": (
            float(np.real(cisd_energy)),
            np.asarray(cisd_state, dtype=complex).reshape(-1),
        ),
    }


def circular_fiedler_layout(ordering: list[int]) -> dict[int, np.ndarray]:
    """Place nodes on a circle in their mutual-information Fiedler order."""
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, len(ordering), endpoint=False)
    return {
        qubit: np.array([np.cos(angle), np.sin(angle)])
        for qubit, angle in zip(ordering, angles)
    }


def save_numerical_data(
    output_dir: Path,
    stem: str,
    mutual_information: np.ndarray,
    edge_tol: float,
) -> tuple[Path, Path]:
    matrix_path = output_dir / f"{stem}_mutual_information_matrix.csv"
    edge_path = output_dir / f"{stem}_mutual_information_edges.csv"

    np.savetxt(
        matrix_path,
        mutual_information,
        delimiter=",",
        header=",".join(f"q{i}" for i in range(mutual_information.shape[0])),
        comments="",
    )

    with edge_path.open("w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["qubit_i", "qubit_j", "mutual_information_bits"])
        for i in range(mutual_information.shape[0]):
            for j in range(i + 1, mutual_information.shape[0]):
                weight = mutual_information[i, j]
                if weight > edge_tol:
                    writer.writerow([i, j, f"{weight:.16g}"])

    return matrix_path, edge_path


def plot_graph(
    system: str,
    state_label: str,
    mutual_information: np.ndarray,
    one_qubit_entropies: np.ndarray,
    ordering: list[int],
    energy: float,
    output_path: Path,
    edge_tol: float,
    label_tol: float,
    dpi: int,
) -> None:
    positions = circular_fiedler_layout(ordering)
    edges = [
        (mutual_information[i, j], i, j)
        for i in range(mutual_information.shape[0])
        for j in range(i + 1, mutual_information.shape[0])
        if mutual_information[i, j] > edge_tol
    ]
    edges.sort()
    weights = np.array([edge[0] for edge in edges])
    if weights.size == 0:
        raise ValueError(f"No graph edges exceed edge_tol={edge_tol:g}.")

    max_weight = float(weights.max())
    positive_min = float(weights.min())
    # A fixed lower color scale keeps numerically tiny edges pale while retaining
    # every edge above edge_tol in the rendered graph.
    color_min = max(positive_min, max_weight * 1.0e-4)
    norm = LogNorm(vmin=color_min, vmax=max_weight)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(11.5, 10.0), constrained_layout=True)
    segments = [np.vstack((positions[i], positions[j])) for _, i, j in edges]
    widths = 0.12 + 6.5 * np.sqrt(weights / max_weight)
    colors = cmap(norm(np.maximum(weights, color_min)))
    colors[:, 3] = 0.10 + 0.85 * np.power(weights / max_weight, 0.25)
    ax.add_collection(
        LineCollection(segments, linewidths=widths, colors=colors, zorder=1)
    )

    node_xy = np.array([positions[i] for i in range(mutual_information.shape[0])])
    node_sizes = 1050.0 + 900.0 * one_qubit_entropies
    nodes = ax.scatter(
        node_xy[:, 0],
        node_xy[:, 1],
        s=node_sizes,
        c=one_qubit_entropies,
        cmap="Blues",
        vmin=0.0,
        vmax=max(1.0, float(one_qubit_entropies.max())),
        edgecolors="#17202a",
        linewidths=1.4,
        zorder=3,
    )
    for qubit, (x_coord, y_coord) in enumerate(node_xy):
        label_color = "white" if one_qubit_entropies[qubit] >= 0.55 else "#17202a"
        ax.text(
            x_coord,
            y_coord,
            str(qubit),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=label_color,
            zorder=4,
        )

    for weight, i, j in edges:
        if weight < label_tol:
            continue
        midpoint = 0.52 * positions[i] + 0.48 * positions[j]
        ax.text(
            midpoint[0],
            midpoint[1],
            f"{weight:.3f}",
            ha="center",
            va="center",
            fontsize=6.7,
            color="#202020",
            bbox={"boxstyle": "round,pad=0.13", "fc": "white", "ec": "none", "alpha": 0.74},
            zorder=2,
        )

    edge_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    edge_colorbar = fig.colorbar(edge_mappable, ax=ax, fraction=0.045, pad=0.03)
    edge_colorbar.set_label("Edge weight: mutual information $I(i:j)$ [bits]")
    node_colorbar = fig.colorbar(nodes, ax=ax, fraction=0.045, pad=0.09)
    node_colorbar.set_label("Node color/size: one-qubit entropy $S(i)$ [bits]")

    ax.set_title(
        f"{system} {state_label.upper()} qubit mutual-information graph\n"
        f"{mutual_information.shape[0]} qubits, "
        f"$E_{{{state_label.upper()}}}={energy:.10f}$ Hartree",
        fontsize=16,
        pad=16,
    )
    ax.text(
        0.5,
        -0.03,
        "Circular node order follows the mutual-information Fiedler ordering; "
        f"edge labels are shown for $I(i:j) \\geq {label_tol:g}$ bits.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
    )
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_sparse_mutual_information(
    state: np.ndarray,
    n_qubits: int,
    sparse_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SparseQubitState]:
    sparse_state = SparseQubitState.from_dense(
        state,
        n_qubits=n_qubits,
        threshold=sparse_threshold,
    ).normalize()
    mutual_information, one_qubit_entropies, two_qubit_entropies = (
        qubit_mutual_information_matrix_sparse_bloch(
            sparse_state,
            n_qubits=n_qubits,
            base=2.0,
            convention="standard",
        )
    )
    return (
        mutual_information,
        one_qubit_entropies,
        two_qubit_entropies,
        sparse_state,
    )


def write_state_outputs(
    system: str,
    stem: str,
    state_label: str,
    energy: float,
    dense_state: np.ndarray,
    output_dir: Path,
    edge_tol: float,
    label_tol: float,
    dpi: int,
    sparse_threshold: float,
) -> None:
    n_qubits = infer_n_qubits_from_state(dense_state)
    mutual_information, one_qubit_entropies, _, sparse_state = (
        compute_sparse_mutual_information(
            dense_state,
            n_qubits=n_qubits,
            sparse_threshold=sparse_threshold,
        )
    )
    fiedler_info = fiedler_order_from_weights(
        mutual_information,
        edge_tol=edge_tol,
        component_order="index",
    )

    output_stem = f"{stem}_{state_label}"
    image_path = output_dir / f"{output_stem}_qubit_mutual_information_graph.png"
    matrix_path, edge_path = save_numerical_data(
        output_dir,
        output_stem,
        mutual_information,
        edge_tol,
    )
    plot_graph(
        system,
        state_label,
        mutual_information,
        one_qubit_entropies,
        fiedler_info["ordering"],
        energy,
        image_path,
        edge_tol,
        label_tol,
        dpi,
    )

    edge_count = int(np.count_nonzero(np.triu(mutual_information, 1) > edge_tol))
    print(f"{state_label.upper()} energy: {energy:.12f} Hartree")
    print(
        f"{state_label.upper()} SparseQubitState nnz: "
        f"{sparse_state.nnz}/{sparse_state.dimension}"
    )
    print(f"{state_label.upper()} plotted edges: {edge_count}")
    print(f"{state_label.upper()} Fiedler ordering: {fiedler_info['ordering']}")
    print(f"{state_label.upper()} graph: {image_path}")
    print(f"{state_label.upper()} matrix: {matrix_path}")
    print(f"{state_label.upper()} edge list: {edge_path}")


def main() -> None:
    args = parse_args()
    if args.input is None:
        system = args.system.removesuffix(".pkl")
        input_path = (HAMILTONIAN_DIR / f"{system}.pkl").resolve()
    else:
        input_path = args.input.expanduser().resolve()
        system = input_path.stem

    if not input_path.is_file():
        raise FileNotFoundError(f"Saved system pickle not found: {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_states = load_saved_states(input_path)
    print(f"Loaded: {input_path}")

    labels = ("fci", "cisd") if args.state == "both" else (args.state,)
    for label in labels:
        energy, state = saved_states[label]
        write_state_outputs(
            system=system,
            stem=input_path.stem,
            state_label=label,
            energy=energy,
            dense_state=state,
            output_dir=output_dir,
            edge_tol=args.edge_tol,
            label_tol=args.label_tol,
            dpi=args.dpi,
            sparse_threshold=args.sparse_threshold,
        )


if __name__ == "__main__":
    main()
