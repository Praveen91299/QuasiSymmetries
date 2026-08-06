"""Iterate fixed-pool Beam search, Clifford transformation, and Fiedler DMRG.

For each selected full-space N2/STO-3G geometry this script:

1. builds and caps one candidate Pauli pool from the canonical Hamiltonian;
2. reuses that exact list of Pauli masks at every iteration;
3. runs Beam search on the Hamiltonian in the current frame;
4. maps the selected generators to single-qubit Z operators with a Clifford;
5. obtains a Fiedler ordering from either the transformed CISD state or a
   fixed-large-bond reference DMRG MPS;
6. applies the Clifford and ordering to the Hamiltonian and CISD warm start;
7. benchmarks the first MPS bond dimension reaching chemical accuracy; and
8. repeats from the newly transformed Hamiltonian.

"Same pool" means that the integer Pauli masks are unchanged between
iterations.  They are interpreted in the current qubit coordinates.  The
score is evaluated covariantly by mapping a candidate back through all prior
Clifford/permutation steps and contracting it with the original Hamiltonian
and CISD state.  This avoids forming a full sparse matrix for a transformed
20-qubit Hamiltonian.

Run from the repository root, for example::

    python -u scripts/iterative_n2_beam_fiedler_dmrg.py --iterations 3

Use ``--resume`` to continue a partially completed output directory.
Use ``--reference-method dmrg`` to avoid requiring an FCI reference energy or
an explicit FCI-like state for the entropy calculation.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import gc
import hashlib
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np

from quasisymmetries.block2_qubit_benchmark import (
    load_qubit_mps_arrays,
    run_block2_qubit_dmrg_curve,
    run_block2_qubit_reference_dmrg,
)
from quasisymmetries.bs.beam import (
    beam_search_symmetries,
    build_candidate_pool_hct,
    local_swap_refine,
    validate_symmetry_generators,
)
from quasisymmetries.bs.utils import mask_to_qubit_operator, qubit_operator_terms
from quasisymmetries.clifford_symmetry_optimized import (
    Clifford,
    invert_permutation,
    permute_qubits_in_qubit_operator,
)
from quasisymmetries.fiedler import (
    fiedler_order_from_mps,
    fiedler_order_from_state,
    invert_ordering,
)
from quasisymmetries.metrics import PauliTermOverlapCommutatorEvaluator
from quasisymmetries.mps_unitary import (
    PermutationUnitary,
    transform_qubit_mps_arrays,
)
from quasisymmetries.save import (
    encode_qubit_operator,
    load_json,
    load_pauli_term_stream,
    load_sparse_qubit_state,
    save_json,
    save_pauli_term_stream,
    save_sparse_qubit_state,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "saved" / "results" / "pyblock2_16_systems"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "saved" / "results" / "iterative_n2_beam_fiedler"
)
DEFAULT_SYSTEMS = ("N2_eqm", "N2_corr", "N2_diss")
DEFAULT_BOND_DIMS = (
    *range(1, 11),
    *range(12, 21, 2),
    *range(30, 101, 10),
)
DEFAULT_NOISES = (1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6)
CHEMICAL_ACCURACY = 1.6e-3


def _rss_message() -> str:
    try:
        import psutil

        gib = psutil.Process().memory_info().rss / 1024**3
        return f" RSS={gib:.3f} GiB"
    except Exception:
        return ""


def _input_file(input_dir: Path, manifest: dict, section: str, key: str):
    path = (input_dir / manifest[section][key]).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_n2_input(input_root: Path, system: str) -> dict:
    input_dir = input_root / system / "orbital_optimization_inputs"
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No saved input manifest for {system}: {manifest_path}"
        )
    manifest = load_json(manifest_path)
    if manifest.get("system") != system:
        raise ValueError(f"Manifest system does not match {system!r}.")
    if not bool(manifest.get("source", {}).get("full_active_space", False)):
        raise ValueError(f"{system} is not marked as a full-active-space input.")

    n_qubits = int(manifest["n_qubits"])
    hamiltonian = load_pauli_term_stream(
        _input_file(input_dir, manifest, "hamiltonian_files", "qubit_json"),
        n_qubits=n_qubits,
    )
    cisd_state = load_sparse_qubit_state(
        _input_file(input_dir, manifest, "state_files", "cisd")
    ).normalize()
    if cisd_state.n_qubits != n_qubits:
        raise ValueError(f"{system}: CISD state has the wrong qubit count.")
    return {
        "input_dir": input_dir,
        "manifest": manifest,
        "hamiltonian": hamiltonian,
        "cisd_state": cisd_state,
        "n_qubits": n_qubits,
        "fci_energy": float(manifest["fci_energy"]),
        "cisd_energy": float(manifest["cisd_energy"]),
    }


def map_operator_to_original_frame(operator, frame_steps):
    """Undo ``(Fiedler permutation) * (Clifford)`` steps in reverse order."""
    mapped = operator
    for clifford, permutation in reversed(frame_steps):
        mapped = permute_qubits_in_qubit_operator(
            mapped, invert_permutation(permutation)
        )
        mapped = clifford.inverse_transform(mapped)
    return mapped


class OriginalFrameCommutatorScore:
    """Score current-frame Paulis with an original-frame term-overlap metric."""

    def __init__(
        self,
        evaluator,
        frame_steps,
    ):
        self.evaluator = evaluator
        self.frame_steps = tuple(frame_steps)

    def __call__(self, symmetries):
        original_frame = [
            map_operator_to_original_frame(symmetry, self.frame_steps)
            for symmetry in symmetries
        ]
        return -self.evaluator.cost(original_frame)


def candidate_pool_digest(candidate_pool) -> str:
    encoded = ";".join(f"{int(x)}:{int(z)}" for x, z in candidate_pool)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def build_fixed_candidate_pool(data: dict, args, score_evaluator):
    n_qubits, terms = qubit_operator_terms(
        data["hamiltonian"], data["n_qubits"]
    )
    pool = build_candidate_pool_hct(
        terms,
        n_qubits,
        max_candidates_from_terms=args.max_candidates_from_terms,
        include_pairwise_products=args.include_pairwise_products,
        pairwise_seed_terms=args.pairwise_seed_terms,
        max_pauli_weight=args.max_pauli_weight,
        include_hct_symmetries=args.include_hct_symmetries,
        hct_n_sym=(
            n_qubits if args.hct_n_sym is None else args.hct_n_sym
        ),
        hct_use_coeffs_eps=True,
    )
    before_cap = len(pool)
    initial_cache = {}
    if args.max_candidate_pool_size is not None and len(pool) > args.max_candidate_pool_size:
        initial_score = OriginalFrameCommutatorScore(
            score_evaluator,
            (),
        )
        scored = []
        for position, mask in enumerate(pool):
            value = initial_score([mask_to_qubit_operator(mask, n_qubits)])
            initial_cache[mask] = value
            scored.append((value, -position, mask))
        scored.sort(reverse=True)
        pool = [item[2] for item in scored[: args.max_candidate_pool_size]]
        initial_cache = {mask: initial_cache[mask] for mask in pool}

    diagnostics = {
        "size_before_cap": before_cap,
        "size_after_cap": len(pool),
        "maximum_size": args.max_candidate_pool_size,
        "was_capped": len(pool) < before_cap,
        "ranking": "largest negative commutator-squared singleton score",
        "sha256": candidate_pool_digest(pool),
    }
    return pool, initial_cache, diagnostics


def encode_symmetries(symmetries):
    return [encode_qubit_operator(symmetry) for symmetry in symmetries]


def iteration_directory(system_dir: Path, iteration: int, baseline=False):
    suffix = "baseline" if baseline else "beam_fiedler"
    return system_dir / f"iteration_{iteration:03d}_{suffix}"


def run_dmrg(
    *,
    system: str,
    iteration: int,
    hamiltonian,
    data: dict,
    cumulative_unitaries,
    output_dir: Path,
    args,
    label_suffix: str,
):
    label = f"{system}_iter{iteration:03d}_{label_suffix}"
    rows, summary = run_block2_qubit_dmrg_curve(
        label=label,
        hamiltonian=hamiltonian,
        sparse_state=(
            data["cisd_state"].indices,
            data["cisd_state"].coeffs,
        ),
        exact_energy=data["reference_energy"],
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
        unitaries=tuple(cumulative_unitaries),
        transform_max_bond=args.transform_mps_max_bond,
        transform_cutoff=args.mps_cutoff,
        full_curve=args.full_curve,
        n_threads=args.n_threads,
        stack_mem_gb=args.stack_mem_gb,
        davidson_threshold=args.davidson_threshold,
        warm_start_noises=args.warm_start_noises,
        verbose=args.verbose,
        artifact_dir=(
            output_dir / "tensor_networks"
            if args.save_tensor_networks
            else None
        ),
    )
    for row in rows:
        row["system"] = system
        row["iteration"] = iteration
        row["representation"] = label_suffix
        row["reference_method"] = data["reference_summary"]["method"]
        row["reference_energy"] = data["reference_energy"]
    summary["benchmark_reference"] = data["reference_summary"]
    write_csv(output_dir / "dmrg_curve.csv", rows)
    return rows, summary


def replay_completed_iteration(result, current_hamiltonian, current_state):
    clifford = Clifford.from_dict(result["clifford"])
    ordering = [int(value) for value in result["fiedler"]["ordering"]]
    permutation = tuple(int(value) for value in result["fiedler"]["old_to_new"])
    current_hamiltonian = clifford.transform(current_hamiltonian)
    current_state = clifford.transform_sparse_state(current_state)
    current_hamiltonian = permute_qubits_in_qubit_operator(
        current_hamiltonian, permutation
    )
    current_state = current_state.reorder_qubits(ordering)
    return (
        current_hamiltonian,
        current_state,
        clifford,
        permutation,
    )


def run_system(system: str, args):
    data = load_n2_input(args.input_root, system)
    n_qubits = data["n_qubits"]
    target_rank = n_qubits if args.target_rank is None else args.target_rank
    if target_rank < 1 or target_rank > n_qubits:
        raise ValueError(f"target rank must lie in [1, {n_qubits}].")

    system_dir = args.output_root / system
    system_dir.mkdir(parents=True, exist_ok=True)
    settings_path = system_dir / "settings.json"
    run_signature = {
        "system": system,
        "input_dir": str(data["input_dir"]),
        "fixed_candidate_pool": True,
        "fixed_pool_coordinate_convention": (
            "unchanged Pauli masks interpreted in each current frame"
        ),
        "target_rank": target_rank,
        "beam_width": args.beam_width,
        "heavy_core_fraction": args.heavy_core_fraction,
        "max_candidates_from_terms": args.max_candidates_from_terms,
        "include_hct_symmetries": args.include_hct_symmetries,
        "hct_n_sym": n_qubits if args.hct_n_sym is None else args.hct_n_sym,
        "include_pairwise_products": args.include_pairwise_products,
        "pairwise_seed_terms": args.pairwise_seed_terms,
        "max_pauli_weight": args.max_pauli_weight,
        "max_candidate_pool_size": args.max_candidate_pool_size,
        "local_refine_passes": args.local_refine_passes,
        "fiedler_reference": (
            "DMRG reference MPS after the iteration Clifford"
            if args.reference_method == "dmrg"
            else "CISD state after the iteration Clifford"
        ),
        "fiedler_component_order": args.fiedler_component_order,
        "bond_dims": list(args.bond_dims),
        "dmrg_sweeps": args.dmrg_sweeps,
        "dmrg_tolerance": args.dmrg_tolerance,
        "sweep_tolerance": args.sweep_tolerance,
        "davidson_threshold": args.davidson_threshold,
        "warm_start_noises": list(args.warm_start_noises),
        "mps_cutoff": args.mps_cutoff,
        "mpo_cutoff": args.mpo_cutoff,
        "mpo_builder": args.mpo_builder,
        "sum_mpo_mod": args.sum_mpo_mod,
        "transform_mps_max_bond": args.transform_mps_max_bond,
        "reference_method": args.reference_method,
        "reference_dmrg_bond_dim": args.reference_dmrg_bond_dim,
        "reference_dmrg_sweeps": args.reference_dmrg_sweeps,
        "reference_dmrg_sweep_tolerance": (
            args.reference_dmrg_sweep_tolerance
        ),
        "reference_transform_max_bond": args.reference_transform_max_bond,
    }
    if settings_path.exists():
        previous = load_json(settings_path)
        if not args.resume:
            raise FileExistsError(
                f"{system_dir} already contains a run; use --resume or a "
                "different --output-root."
            )
        if previous.get("run_signature") != run_signature:
            raise ValueError(
                f"Cannot resume {system}: saved settings differ from this run."
            )
    else:
        save_json(
            settings_path,
            {
                "run_signature": run_signature,
                "requested_iterations": args.iterations,
            },
        )

    reference_tensors = None
    if args.reference_method == "dmrg":
        reference_path = system_dir / "reference_dmrg.json"
        portable_path = (
            system_dir
            / "reference_tensor_networks"
            / f"{system}_reference_mps.npz"
        )
        if args.resume and reference_path.exists() and portable_path.exists():
            reference_summary = load_json(reference_path)
            reference_tensors = load_qubit_mps_arrays(portable_path)
            print(
                f"{system}: reusing fixed-bond DMRG reference "
                f"E={reference_summary['energy']:.12f}.",
                flush=True,
            )
        else:
            reference_tensors, reference_summary = (
                run_block2_qubit_reference_dmrg(
                    label=system,
                    hamiltonian=data["hamiltonian"],
                    n_qubits=n_qubits,
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
                    artifact_dir=(
                        system_dir / "reference_tensor_networks"
                    ),
                )
            )
            save_json(reference_path, reference_summary)
        data["reference_energy"] = float(reference_summary["energy"])
        data["reference_summary"] = reference_summary
    else:
        data["reference_energy"] = data["fci_energy"]
        data["reference_summary"] = {
            "method": "saved_fci",
            "energy": data["fci_energy"],
        }

    print(
        f"\n{'=' * 80}\n{system}: preparing matrix-free score data"
        f"\n{'=' * 80}",
        flush=True,
    )
    sparse_start = perf_counter()
    score_evaluator = PauliTermOverlapCommutatorEvaluator(
        data["hamiltonian"], data["cisd_state"]
    )
    print(
        f"{system}: Pauli-term overlap matrix built in "
        f"{perf_counter() - sparse_start:.1f} s.{_rss_message()}",
        flush=True,
    )

    pool, initial_score_cache, pool_diagnostics = build_fixed_candidate_pool(
        data, args, score_evaluator
    )
    pool_payload = {
        "n_qubits": n_qubits,
        "masks": [[int(x), int(z)] for x, z in pool],
        "diagnostics": pool_diagnostics,
    }
    pool_path = system_dir / "fixed_candidate_pool.json"
    if args.resume and pool_path.exists():
        saved_pool = load_json(pool_path)
        if saved_pool["masks"] != pool_payload["masks"]:
            raise ValueError(f"{system}: reconstructed candidate pool changed.")
    else:
        save_json(pool_path, pool_payload)
    print(
        f"{system}: fixed pool has {len(pool)} candidates "
        f"(sha256={pool_diagnostics['sha256'][:12]}...).",
        flush=True,
    )

    current_hamiltonian = data["hamiltonian"]
    current_state = data["cisd_state"].copy()
    current_reference_tensors = reference_tensors
    frame_steps = []
    cumulative_unitaries = []
    all_rows = []
    iteration_summaries = []

    if not args.skip_baseline:
        baseline_dir = iteration_directory(system_dir, 0, baseline=True)
        result_path = baseline_dir / "result.json"
        if args.resume and result_path.exists():
            baseline = load_json(result_path)
            print(f"{system}: reusing completed baseline DMRG.", flush=True)
        else:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            rows, summary = run_dmrg(
                system=system,
                iteration=0,
                hamiltonian=current_hamiltonian,
                data=data,
                cumulative_unitaries=(),
                output_dir=baseline_dir,
                args=args,
                label_suffix="baseline",
            )
            baseline = {
                "system": system,
                "iteration": 0,
                "kind": "baseline",
                "dmrg": summary,
            }
            save_json(result_path, baseline)
            all_rows.extend(rows)
        iteration_summaries.append(baseline)

    for iteration in range(1, args.iterations + 1):
        output_dir = iteration_directory(system_dir, iteration)
        result_path = output_dir / "result.json"
        if args.resume and result_path.exists():
            result = load_json(result_path)
            (
                current_hamiltonian,
                current_state,
                clifford,
                permutation,
            ) = replay_completed_iteration(
                result, current_hamiltonian, current_state
            )
            frame_steps.append((clifford, permutation))
            cumulative_unitaries.extend(
                (clifford, PermutationUnitary(permutation))
            )
            if current_reference_tensors is not None:
                current_reference_tensors, _ = transform_qubit_mps_arrays(
                    current_reference_tensors,
                    unitaries=(
                        clifford,
                        PermutationUnitary(permutation),
                    ),
                    max_bond=args.reference_transform_max_bond,
                    cutoff=args.mps_cutoff,
                )
            iteration_summaries.append(result)
            print(
                f"{system}: replayed completed iteration {iteration}.",
                flush=True,
            )
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n{system}: Beam/Clifford/Fiedler iteration {iteration}"
            f"{_rss_message()}",
            flush=True,
        )
        score = OriginalFrameCommutatorScore(
            score_evaluator,
            frame_steps,
        )
        score_cache = (
            initial_score_cache.copy()
            if iteration == 1 and not frame_steps
            else {}
        )
        search_start = perf_counter()
        symmetries = beam_search_symmetries(
            current_hamiltonian,
            pool,
            target_rank=target_rank,
            n_qubits=n_qubits,
            beam_width=args.beam_width,
            heavy_core_fraction=args.heavy_core_fraction,
            score_func=score,
            score_is_separable=True,
            separable_score_cache=score_cache,
            n_processes=args.n_processes,
            mp_start_method=args.mp_start_method,
        )
        if args.local_refine_passes > 0:
            symmetries = local_swap_refine(
                current_hamiltonian,
                symmetries,
                pool,
                n_qubits=n_qubits,
                max_passes=args.local_refine_passes,
                score_func=score,
                score_is_separable=True,
                separable_score_cache=score_cache,
                n_processes=args.n_processes,
                mp_start_method=args.mp_start_method,
            )
        search_seconds = perf_counter() - search_start
        selected_score = score(symmetries)
        validation = validate_symmetry_generators(
            current_hamiltonian, symmetries, n_qubits=n_qubits
        )

        clifford = Clifford.from_symmetries(
            symmetries,
            n_qubits=n_qubits,
            symmetry_qubits_first=True,
            synthesis_basis="Z",
            generator_mapping="positive_z",
        )
        clifford_hamiltonian = clifford.transform(current_hamiltonian)
        clifford_state = clifford.transform_sparse_state(current_state)
        fiedler_start = perf_counter()
        reference_transform = None
        permutation_transform = None
        if current_reference_tensors is not None:
            clifford_reference_tensors, reference_transform = (
                transform_qubit_mps_arrays(
                    current_reference_tensors,
                    unitaries=(clifford,),
                    max_bond=args.reference_transform_max_bond,
                    cutoff=args.mps_cutoff,
                )
            )
            fiedler = fiedler_order_from_mps(
                clifford_reference_tensors,
                base=np.e,
                component_order=args.fiedler_component_order,
            )
        else:
            dense_clifford_state = clifford_state.to_dense()
            fiedler = fiedler_order_from_state(
                dense_clifford_state,
                n_qubits=n_qubits,
                base=np.e,
                component_order=args.fiedler_component_order,
                mutual_information_method="statevector",
            )
        fiedler_seconds = perf_counter() - fiedler_start
        ordering = [int(value) for value in fiedler["ordering"]]
        permutation = tuple(int(value) for value in invert_ordering(ordering))
        current_hamiltonian = permute_qubits_in_qubit_operator(
            clifford_hamiltonian, permutation
        )
        current_state = clifford_state.reorder_qubits(ordering)
        if current_reference_tensors is not None:
            current_reference_tensors, permutation_transform = (
                transform_qubit_mps_arrays(
                    clifford_reference_tensors,
                    unitaries=(PermutationUnitary(permutation),),
                    max_bond=args.reference_transform_max_bond,
                    cutoff=args.mps_cutoff,
                )
            )
        if not np.isclose(current_state.norm(), 1.0, atol=1e-10):
            raise RuntimeError(
                f"{system} iteration {iteration}: transformed state lost norm."
            )
        frame_steps.append((clifford, permutation))
        cumulative_unitaries.extend(
            (clifford, PermutationUnitary(permutation))
        )

        save_pauli_term_stream(
            output_dir / "transformed_hamiltonian.pauli_masks.json",
            current_hamiltonian,
        )
        save_sparse_qubit_state(
            output_dir / "transformed_cisd_state.npz", current_state
        )
        rows, dmrg_summary = run_dmrg(
            system=system,
            iteration=iteration,
            hamiltonian=current_hamiltonian,
            data=data,
            cumulative_unitaries=cumulative_unitaries,
            output_dir=output_dir,
            args=args,
            label_suffix="beam_fiedler",
        )
        all_rows.extend(rows)
        result = {
            "system": system,
            "iteration": iteration,
            "kind": "beam_clifford_fiedler",
            "fixed_pool_sha256": pool_diagnostics["sha256"],
            "selected_symmetries": encode_symmetries(symmetries),
            "selected_score": selected_score,
            "search_seconds": search_seconds,
            "search_validation": validation,
            "clifford": clifford.to_dict(),
            "fiedler": {
                "ordering": ordering,
                "old_to_new": list(permutation),
                "seconds": fiedler_seconds,
                "one_qubit_entropies": fiedler["one_qubit_entropies"],
                "mutual_information": fiedler["mutual_information"],
                "components": fiedler["components"],
                "reference_state": data["reference_summary"]["method"],
                "reference_clifford_transform": reference_transform,
                "reference_permutation_transform": permutation_transform,
            },
            "transformed_cisd_determinants": len(current_state.indices),
            "transformed_cisd_norm": current_state.norm(),
            "dmrg": dmrg_summary,
        }
        save_json(result_path, result)
        iteration_summaries.append(result)
        print(
            f"{system} iteration {iteration}: first chemical-accuracy bond "
            f"dimension={dmrg_summary.get('first_converged_bond_dim')}",
            flush=True,
        )
        if "dense_clifford_state" in locals():
            del dense_clifford_state
        if "clifford_reference_tensors" in locals():
            del clifford_reference_tensors
        del clifford_state, score
        gc.collect()

    rows_from_disk = []
    for child in sorted(system_dir.glob("iteration_*")):
        curve_path = child / "dmrg_curve.csv"
        if curve_path.exists():
            import csv

            with curve_path.open(newline="", encoding="utf-8") as file_obj:
                rows_from_disk.extend(list(csv.DictReader(file_obj)))
    if rows_from_disk:
        write_csv(system_dir / "dmrg_curves.csv", rows_from_disk)

    report_rows = []
    for result in iteration_summaries:
        dmrg = result.get("dmrg", {})
        report_rows.append(
            {
                "system": system,
                "iteration": result["iteration"],
                "kind": result["kind"],
                "first_converged_bond_dim": dmrg.get(
                    "first_converged_bond_dim"
                ),
                "converged_within_grid": dmrg.get(
                    "converged_within_grid", False
                ),
                "mpo_bond_dimension": dmrg.get("mpo_bond_dimension"),
                "search_seconds": result.get("search_seconds"),
                "fiedler_seconds": result.get("fiedler", {}).get("seconds"),
                "selected_score": result.get("selected_score"),
            }
        )
    write_csv(system_dir / "iteration_summary.csv", report_rows)
    save_json(
        system_dir / "benchmark.json",
        {
            "system": system,
            "candidate_pool": pool_diagnostics,
            "reference": data["reference_summary"],
            "iterations": iteration_summaries,
        },
    )
    del score_evaluator
    gc.collect()
    return report_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--systems", nargs="+", choices=DEFAULT_SYSTEMS, default=list(DEFAULT_SYSTEMS)
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--target-rank", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reference-method",
        choices=("saved_fci", "dmrg"),
        default="saved_fci",
        help="Select the saved FCI energy/CISD entropy state or a scalable "
        "fixed-bond DMRG reference energy and MPS.",
    )
    parser.add_argument("--reference-dmrg-bond-dim", type=int, default=400)
    parser.add_argument("--reference-dmrg-sweeps", type=int, default=100)
    parser.add_argument(
        "--reference-dmrg-sweep-tolerance", type=float, default=1e-8
    )
    parser.add_argument(
        "--reference-transform-max-bond", type=int, default=400
    )

    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--heavy-core-fraction", type=float, default=0.95)
    parser.add_argument("--max-candidates-from-terms", type=int, default=256)
    parser.add_argument(
        "--include-hct-symmetries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hct-n-sym", type=int, default=None)
    parser.add_argument(
        "--include-pairwise-products",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pairwise-seed-terms", type=int, default=50)
    parser.add_argument("--max-pauli-weight", type=int, default=None)
    parser.add_argument("--max-candidate-pool-size", type=int, default=1000)
    parser.add_argument("--local-refine-passes", type=int, default=10)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mp-start-method", default=None)
    parser.add_argument(
        "--fiedler-component-order",
        choices=("index", "total_weight", "size"),
        default="index",
    )

    parser.add_argument(
        "--bond-dims", type=int, nargs="+", default=list(DEFAULT_BOND_DIMS)
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
    parser.add_argument("--mps-cutoff", type=float, default=1e-13)
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument(
        "--mpo-builder",
        choices=("blocked_sum", "expression"),
        default="blocked_sum",
    )
    parser.add_argument("--sum-mpo-mod", type=int, default=20)
    parser.add_argument("--sparse-batch-size", type=int, default=32)
    parser.add_argument("--transform-mps-max-bond", type=int, default=100)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=0.5)
    parser.add_argument("--full-curve", action="store_true")
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
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if not 0 < args.heavy_core_fraction <= 1:
        raise ValueError("--heavy-core-fraction must lie in (0, 1].")
    if args.max_candidate_pool_size is not None and args.max_candidate_pool_size < 1:
        raise ValueError("--max-candidate-pool-size must be positive.")
    if args.local_refine_passes < 0:
        raise ValueError("--local-refine-passes cannot be negative.")
    if any(value < 1 for value in args.bond_dims):
        raise ValueError("All bond dimensions must be positive.")
    if args.reference_dmrg_bond_dim < 1:
        raise ValueError("--reference-dmrg-bond-dim must be positive.")
    if args.reference_dmrg_sweeps < 1:
        raise ValueError("--reference-dmrg-sweeps must be positive.")
    if args.reference_transform_max_bond < 1:
        raise ValueError("--reference-transform-max-bond must be positive.")
    if any(value < 0 for value in args.warm_start_noises):
        raise ValueError("Warm-start noises must be nonnegative.")
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    for system in args.systems:
        aggregate_rows.extend(run_system(system, args))
        write_csv(args.output_root / "iteration_summary.csv", aggregate_rows)
    save_json(
        args.output_root / "benchmark.json",
        {
            "systems": list(args.systems),
            "requested_iterations": args.iterations,
            "summary": aggregate_rows,
        },
    )
    print(f"Saved iterative N2 benchmark to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
