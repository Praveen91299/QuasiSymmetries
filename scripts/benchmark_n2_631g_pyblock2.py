#!/usr/bin/env python3
"""Checkpointed, FCI-free N2/6-31G qubit-DMRG benchmark.

The input is produced by ``probe_n2_631g_qubit_mpo.py``. This script prepares
a determinant-sparse CISD warm start, obtains and validates fixed-bond DMRG
reference calculations, finds scalable HCT and Beam symmetries, computes
MPS-based entropies and Fiedler orderings, and benchmarks one Hamiltonian frame
at a time with pyblock2. Every expensive stage is saved and reusable.
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np
from openfermion import MolecularData

import _bootstrap

from quasisymmetries.block2_qubit_benchmark import (
    load_qubit_mps_arrays,
    run_block2_qubit_dmrg_curve,
    run_block2_qubit_reference_dmrg,
)
from quasisymmetries.bs.beam import (
    beam_search_symmetries,
    build_candidate_pool,
    validate_symmetry_generators,
)
from quasisymmetries.bs.utils import (
    qubit_operator_terms,
    qubitops_to_masks,
    symplectic_commutes,
)
from quasisymmetries.chemistry import (
    run_restricted_cisd_from_molecular_data,
)
from quasisymmetries.clifford_symmetry_optimized import (
    Clifford,
    permute_qubits_in_qubit_operator,
)
from quasisymmetries.fiedler import (
    fiedler_order_from_mps,
    invert_ordering,
    qubit_mps_cut_entropies,
)
from quasisymmetries.mps_unitary import (
    PermutationUnitary,
    transform_qubit_mps_arrays,
)
from quasisymmetries.save import (
    decode_qubit_operator,
    encode_qubit_operator,
    load_json,
    load_pauli_term_stream,
    load_sparse_qubit_state,
    read_csv,
    save_json,
    save_sparse_qubit_state,
    to_jsonable,
    write_csv,
)
from quasisymmetries.sym import HCT, get_seniority_symmetries

import benchmark_raw_n2_dmrg as fermionic_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_DIR = ROOT / "saved" / "results" / "n2_631g_mpo_probe"
DEFAULT_OUTPUT_DIR = ROOT / "saved" / "results" / "n2_631g_benchmark"
CHEMICAL_ACCURACY = 1.6e-3
DEFAULT_NOISES = (1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6)
DEFAULT_BOND_DIMS = (10, 20, 30, 40, 60, 80, 100, 150, 200)

RAW_FERMION = "raw_fermionic_su2"
RAW = "raw_qubit"
SENIORITY = "seniority_Nover2"
HCT_HALF = "HCT_Nover2_CommL1"
HCT_FULL = "HCT_N_CommL1"
BEAM_HALF = "Beam_Nover2_CommL1"
BEAM_FULL = "Beam_N_CommL1"
HCT_FIEDLER = "HCT_N_CommL1_Fiedler"
BEAM_FIEDLER = "Beam_N_CommL1_Fiedler"
ALL_FRAMES = (
    RAW_FERMION,
    RAW,
    SENIORITY,
    HCT_HALF,
    HCT_FULL,
    BEAM_HALF,
    BEAM_FULL,
    HCT_FIEDLER,
    BEAM_FIEDLER,
)


def rss_gib() -> float | None:
    """Return current process resident memory in GiB.

    Returns
    -------
    rss
        Current resident memory in GiB, or ``None`` when ``psutil`` is not
        installed or the operating system denies process inspection.
    """
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024**3)
    except Exception:
        return None


def rss_message() -> str:
    """Return a compact resident-memory message for progress logging.

    Returns
    -------
    message
        Empty string when memory is unavailable, otherwise a leading-space
        message such as ``" RSS=0.250 GiB"``.
    """
    value = rss_gib()
    return "" if value is None else f" RSS={value:.3f} GiB"


def peak_rss_gib() -> float:
    """Return the process peak resident memory in GiB.

    Returns
    -------
    peak
        Maximum resident set size reported by the operating system. macOS
        reports ``ru_maxrss`` in bytes, whereas Linux reports KiB.
    """
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return value / divisor


def load_probe_input(probe_dir: Path, frozen_core_orbitals: int) -> dict:
    """Load the checkpointed molecular data and streamed Pauli Hamiltonian.

    Parameters
    ----------
    probe_dir
        Output directory created by ``probe_n2_631g_qubit_mpo.py``.
    frozen_core_orbitals
        Frozen-core count identifying ``probe_fcN.json`` and its Pauli stream.

    Returns
    -------
    data
        Probe manifest, loaded ``MolecularData``, packed Pauli stream, and
        active-space dimensions required by later stages.
    """
    manifest_path = probe_dir / f"probe_fc{int(frozen_core_orbitals)}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run probe_n2_631g_qubit_mpo.py first."
        )
    manifest = load_json(manifest_path)
    if manifest.get("status") not in {"hamiltonian_ready", "mpo_ready"}:
        raise ValueError(f"Probe is not ready: status={manifest.get('status')}")
    pauli_path = Path(manifest["pauli_stream"]).expanduser().resolve()
    n_qubits = int(manifest["active_space"]["n_qubits"])
    hamiltonian = load_pauli_term_stream(pauli_path, n_qubits=n_qubits)
    molecular_path = Path(manifest["molecular_data"]).expanduser().resolve()
    molecule = MolecularData(filename=str(molecular_path.with_suffix("")))
    molecule.load()
    return {
        "manifest_path": manifest_path.resolve(),
        "manifest": manifest,
        "molecule": molecule,
        "hamiltonian": hamiltonian,
        "n_qubits": n_qubits,
    }


def encode_symmetries(symmetries) -> list[dict]:
    """Encode single-Pauli OpenFermion generators for JSON storage.

    Parameters
    ----------
    symmetries
        Iterable of OpenFermion ``QubitOperator`` generators.

    Returns
    -------
    payload
        List of portable dictionaries accepted by :func:`decode_symmetries`.
    """
    return [encode_qubit_operator(symmetry) for symmetry in symmetries]


def decode_symmetries(payload) -> list:
    """Decode generators written by :func:`encode_symmetries`.

    Parameters
    ----------
    payload
        Iterable of encoded QubitOperator dictionaries.

    Returns
    -------
    symmetries
        List of OpenFermion ``QubitOperator`` generators.
    """
    return [decode_qubit_operator(item) for item in payload]


def commutator_l1_for_mask(mask, terms) -> float:
    """Return the Pauli coefficient 1-norm of ``[H, S]``.

    Parameters
    ----------
    mask
        Packed Pauli mask of the single-product candidate ``S``.
    terms
        Weighted packed Pauli terms of ``H``.

    Returns
    -------
    value
        ``2 * sum(abs(h_j))`` over Hamiltonian terms anticommuting with ``S``.
        This is minimized by the HCT ranking convention.
    """
    return 2.0 * sum(
        term.abs_coeff
        for term in terms
        if term.mask != (0, 0) and not symplectic_commutes(mask, term.mask)
    )


def prepare_cisd(data: dict, args) -> tuple[float, object, dict]:
    """Generate or reload the determinant-sparse CISD warm-start state.

    Parameters
    ----------
    data
        Dictionary returned by :func:`load_probe_input`.
    args
        Parsed benchmark settings including output path and CISD tolerances.

    Returns
    -------
    energy, state, metadata
        Total CISD energy, normalized ``SparseQubitState``, and saved
        preparation diagnostics.
    """
    directory = args.output_dir / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "cisd_state.npz"
    metadata_path = directory / "cisd.json"
    if state_path.exists() and metadata_path.exists() and not args.force_stage:
        metadata = load_json(metadata_path)
        if (
            metadata.get("jw_phase_convention")
            == "interleaved_spin_orbital_v1"
        ):
            state = load_sparse_qubit_state(state_path).normalize()
            return float(metadata["energy"]), state, metadata
        print(
            "Saved CISD state predates the interleaved-JW fermionic phase "
            "fix; regenerating only the CISD checkpoint.",
            flush=True,
        )

    print(f"Preparing determinant-sparse CISD state.{rss_message()}", flush=True)
    start = perf_counter()
    energy, state, metadata = run_restricted_cisd_from_molecular_data(
        data["molecule"],
        frozen_core_orbitals=args.frozen_core_orbitals,
        convergence_tolerance=args.cisd_tolerance,
        max_cycle=args.cisd_max_cycle,
        coefficient_tolerance=args.cisd_coefficient_tolerance,
        verbose=(4 if args.verbose else 0),
    )
    metadata["seconds"] = perf_counter() - start
    metadata["rss_gib_after"] = rss_gib()
    if state.n_qubits != data["n_qubits"]:
        raise RuntimeError("CISD and Hamiltonian active spaces differ")
    save_sparse_qubit_state(state_path, state)
    metadata["state_path"] = str(state_path.resolve())
    save_json(metadata_path, metadata)
    print(
        f"CISD: E={energy:.12f}, determinants={state.nnz}, "
        f"seconds={metadata['seconds']:.1f}.{rss_message()}",
        flush=True,
    )
    return energy, state, metadata


def prepare_references(
    data: dict,
    cisd_energy: float,
    cisd_state,
    args,
) -> tuple[list[np.ndarray], dict]:
    """Run/reload fixed-bond references and compare the two largest energies.

    Parameters
    ----------
    data
        Loaded Hamiltonian input.
    cisd_energy, cisd_state
        Sparse CISD warm-start energy and state.
    args
        Parsed DMRG and output settings.

    Returns
    -------
    tensors, validation
        Portable MPS tensors from the largest non-skipped reference bond
        dimension and validation metadata. With at least two dimensions,
        ``validation`` records their energy difference and whether it is below
        ``reference_validation_tolerance``.
    """
    reference_dir = args.output_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    largest_tensors = None
    skipped_dimensions = set(
        int(value) for value in args.skip_reference_bond_dims
    )
    dimensions = sorted(
        set(int(value) for value in args.reference_bond_dims)
        - skipped_dimensions
    )
    if not dimensions:
        raise ValueError("all requested reference bond dimensions were skipped")
    for bond_dim in dimensions:
        label = f"N2_631g_fc{args.frozen_core_orbitals}_M{bond_dim}"
        summary_path = reference_dir / f"reference_M{bond_dim}.json"
        summary = load_json(summary_path) if summary_path.exists() else None
        portable = (
            None
            if summary is None
            else summary.get("saved_tensor_networks", {}).get("portable_mps")
        )
        needs_portable_mps = bond_dim == dimensions[-1]
        portable_exists = bool(
            portable is not None and Path(portable["path"]).exists()
        )
        can_reuse = (
            summary is not None
            and not args.force_stage
            and (not needs_portable_mps or portable_exists)
        )
        if can_reuse:
            print(
                f"Reusing reference M={bond_dim}, "
                f"E={summary['energy']:.12f}.",
                flush=True,
            )
            if needs_portable_mps:
                largest_tensors = load_qubit_mps_arrays(portable["path"])
        else:
            if summary is not None and needs_portable_mps and not portable_exists:
                print(
                    f"Reference M={bond_dim} energy exists, but its portable "
                    "MPS was not saved; rerunning this bond dimension once.",
                    flush=True,
                )
            print(f"Running reference DMRG at M={bond_dim}.{rss_message()}", flush=True)
            tensors, summary = run_block2_qubit_reference_dmrg(
                label=label,
                hamiltonian=data["hamiltonian"],
                n_qubits=data["n_qubits"],
                bond_dim=bond_dim,
                sparse_state=(cisd_state.indices, cisd_state.coeffs),
                initial_state="cisd",
                dmrg_sweeps=args.reference_sweeps,
                sweep_tolerance=args.reference_sweep_tolerance,
                mps_cutoff=args.mps_cutoff,
                mpo_cutoff=args.mpo_cutoff,
                mpo_builder=args.mpo_builder,
                sum_mpo_mod=args.sum_mpo_mod,
                sparse_batch_size=args.sparse_batch_size,
                davidson_threshold=args.davidson_threshold,
                noises=args.warm_start_noises,
                n_threads=args.n_threads,
                n_mkl_threads=args.n_mkl_threads,
                stack_mem_gb=args.stack_mem_gb,
                verbose=args.verbose,
                artifact_dir=(reference_dir if bond_dim == dimensions[-1] else None),
            )
            summary["bond_dim"] = bond_dim
            summary["rss_gib_after"] = rss_gib()
            summary["process_peak_rss_gib_after"] = peak_rss_gib()
            save_json(summary_path, summary)
            if bond_dim == dimensions[-1]:
                largest_tensors = tensors
        summaries.append(summary)
        gc.collect()

    if largest_tensors is None:
        raise RuntimeError("largest reference MPS was not loaded")
    state_energy = float(summaries[-1]["energy"])
    energy_reference = prepare_fermionic_energy_reference(
        data,
        cisd_energy,
        cisd_state,
        provisional_energy=state_energy,
        args=args,
    )
    validation = {
        "method": "fermionic_su2_fixed_bond_validation",
        "calculations": summaries,
        "energy_reference": energy_reference,
        "requested_bond_dims": sorted(
            set(int(value) for value in args.reference_bond_dims)
        ),
        "skipped_bond_dims": sorted(skipped_dimensions),
        "state_selected_bond_dim": dimensions[-1],
        "state_energy": state_energy,
        "selected_bond_dim": energy_reference["selected_bond_dim"],
        "energy": energy_reference["energy"],
        "comparison_bond_dim": energy_reference["comparison_bond_dim"],
        "largest_two_energy_difference": energy_reference[
            "energy_difference"
        ],
        "validation_tolerance": args.reference_validation_tolerance,
        "validated": energy_reference["validated"],
        "reference_state_source": (
            "fixed-bond qubit DMRG MPS used for entropy and Fiedler analysis"
        ),
        "reference_energy_source": (
            "lower variational energy from the two SU(2) fermionic fixed-bond "
            "checks; validity additionally requires sweep convergence"
        ),
    }
    save_json(reference_dir / "reference_validation.json", validation)
    if not validation["validated"]:
        print(
            "WARNING: the fermionic SU(2) reference pair is not validated: "
            "both runs must converge in sweep energy and differ by at most "
            f"{args.reference_validation_tolerance:.3e} Ha.",
            flush=True,
        )
        if args.require_reference_validation:
            raise RuntimeError("reference DMRG validation requirement failed")
    return largest_tensors, validation


def select_fermionic_energy_reference(
    rows: list[dict],
    *,
    requested_bond_dims: tuple[int, int],
    validation_tolerance: float,
) -> dict:
    """Select and validate an energy from two fixed-bond SU(2) DMRG runs.

    Parameters
    ----------
    rows
        Fermionic DMRG result dictionaries. Each requested bond dimension must
        occur exactly once and provide ``energy`` and ``sweep_converged``.
    requested_bond_dims
        Pair ``(M_ref, M_ref + delta_M)`` used for the reference and its
        finite-bond check.
    validation_tolerance
        Maximum allowed absolute energy difference between the two runs.

    Returns
    -------
    reference
        Mapping containing the lower variational energy, its bond dimension,
        the other comparison dimension, the energy difference, individual
        sweep-convergence flags, and the combined validation flag.

    Notes
    -----
    The lower of the two energies is selected because both are variational
    upper bounds. Validation additionally requires both DMRG calculations to
    satisfy their sweep-energy stopping criterion.
    """
    requested = tuple(int(value) for value in requested_bond_dims)
    if len(requested) != 2 or requested[0] == requested[1]:
        raise ValueError("requested_bond_dims must contain two distinct values")
    selected_rows = {}
    for row in rows:
        bond_dim = int(row["bond_dim"])
        if bond_dim in requested:
            if bond_dim in selected_rows:
                raise ValueError(f"duplicate reference row for M={bond_dim}")
            selected_rows[bond_dim] = row
    missing = sorted(set(requested) - set(selected_rows))
    if missing:
        raise ValueError(f"missing fermionic reference rows for M={missing}")

    energies = {
        bond_dim: float(selected_rows[bond_dim]["energy"])
        for bond_dim in requested
    }
    best_bond_dim = min(requested, key=lambda value: energies[value])
    other_bond_dim = next(
        value for value in requested if value != best_bond_dim
    )
    sweep_converged = {
        str(bond_dim): bool(selected_rows[bond_dim]["sweep_converged"])
        for bond_dim in requested
    }
    difference = abs(energies[requested[1]] - energies[requested[0]])
    return {
        "energy": energies[best_bond_dim],
        "selected_bond_dim": best_bond_dim,
        "comparison_bond_dim": other_bond_dim,
        "requested_bond_dims": list(requested),
        "energies": {str(key): value for key, value in energies.items()},
        "energy_difference": difference,
        "validation_tolerance": float(validation_tolerance),
        "sweep_converged": sweep_converged,
        "validated": bool(
            all(sweep_converged.values())
            and difference <= float(validation_tolerance)
        ),
    }


def prepare_fermionic_energy_reference(
    data: dict,
    cisd_energy: float,
    cisd_state,
    *,
    provisional_energy: float,
    args,
) -> dict:
    """Run or reload the high-bond SU(2) energy-reference pair.

    Parameters
    ----------
    data
        Loaded molecular data and Hamiltonian metadata.
    cisd_energy, cisd_state
        Correctly phased sparse CISD energy and warm-start state.
    provisional_energy
        Quasi-exact qubit-MPS energy used only by the generic fermionic curve
        routine while the improved reference is being constructed.
    args
        Benchmark namespace supplying the base reference bond dimension,
        ``+delta_M`` check, sweep settings, threads, memory, and output path.

    Returns
    -------
    reference
        Cached payload containing both fixed-bond rows, their settings, and
        the selected/validated fermionic SU(2) reference energy.
    """
    reference_dir = args.output_dir / "reference"
    path = reference_dir / "fermionic_energy_reference.json"
    requested = (
        int(args.energy_reference_bond_dim),
        int(args.energy_reference_bond_dim)
        + int(args.energy_reference_bond_increment),
    )
    settings = {
        "requested_bond_dims": list(requested),
        "dmrg_sweeps": int(args.reference_sweeps),
        "sweep_tolerance": float(args.reference_sweep_tolerance),
        "validation_tolerance": float(args.reference_validation_tolerance),
        "davidson_threshold": float(args.davidson_threshold),
        "warm_start_noises": list(args.warm_start_noises),
        "jw_phase_convention": "interleaved_spin_orbital_v1",
    }
    if path.exists() and not args.force_stage:
        cached = load_json(path)
        if cached.get("settings") == settings:
            print(
                "Reusing fermionic energy reference at "
                f"M={requested[0]} and M={requested[1]}.",
                flush=True,
            )
            return cached

    print(
        "Running fermionic SU(2) energy reference at "
        f"M={requested[0]} and M={requested[1]}.",
        flush=True,
    )
    override_values = {
        "output_dir": reference_dir / "fermionic_energy_reference",
        "bond_dims": list(requested),
        "dmrg_sweeps": int(args.reference_sweeps),
        "sweep_tol": float(args.reference_sweep_tolerance),
        "dmrg_tol": float(args.dmrg_tolerance),
        "initial_state": "cisd",
        "save_tensor_networks": False,
        "full_curve": True,
    }
    missing = object()
    original_values = {
        name: getattr(args, name, missing) for name in override_values
    }
    for name, value in override_values.items():
        setattr(args, name, value)
    try:
        rows, summary = fermionic_backend.run_fermionic_dmrg_curve(
            molecule=data["molecule"],
            warm_start_state=cisd_state,
            warm_start_energy=cisd_energy,
            fci_energy=float(provisional_energy),
            args=args,
        )
    finally:
        for name, value in original_values.items():
            if value is missing:
                delattr(args, name)
            else:
                setattr(args, name, value)

    selected = select_fermionic_energy_reference(
        rows,
        requested_bond_dims=requested,
        validation_tolerance=args.reference_validation_tolerance,
    )
    payload = {
        **selected,
        "method": "fermionic_su2_fixed_bond_validation",
        "settings": settings,
        "rows": rows,
        "dmrg_summary": summary,
    }
    save_json(path, payload)
    print(
        "Fermionic energy reference: "
        f"E={payload['energy']:.12f} at M={payload['selected_bond_dim']}; "
        f"|E({requested[1]})-E({requested[0]})|="
        f"{payload['energy_difference']:.3e}; "
        f"validated={payload['validated']}.",
        flush=True,
    )
    return payload


def prepare_symmetries(data: dict, args) -> dict:
    """Find/reload HCT and Beam generators using scalable Pauli-only scores.

    Parameters
    ----------
    data
        Loaded packed Pauli Hamiltonian and qubit count.
    args
        HCT, Beam, pool-capping, and output settings.

    Returns
    -------
    payload
        Portable generator sets, scores, search timings, validations, and the
        exact capped candidate pool used by both Beam calculations.
    """
    directory = args.output_dir / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "symmetries.json"
    if path.exists() and not args.force_stage:
        return load_json(path)

    n_qubits, terms = qubit_operator_terms(
        data["hamiltonian"], data["n_qubits"]
    )
    print(f"Finding {n_qubits} HCT generators.{rss_message()}", flush=True)
    start = perf_counter()
    hct_full, hct_epsilons = HCT(
        data["hamiltonian"],
        n_sym=n_qubits,
        use_coeffs_eps=args.hct_use_coefficient_thresholds,
        num_intervals=args.hct_intervals,
        tol=args.hct_term_tolerance,
        verbose=args.verbose,
    )
    hct_seconds = perf_counter() - start
    hct_masks = qubitops_to_masks(hct_full, n_qubits)

    base_pool = build_candidate_pool(
        terms,
        n_qubits,
        max_candidates_from_terms=args.max_candidates_from_terms,
        include_pairwise_products=True,
        pairwise_seed_terms=args.pairwise_seed_terms,
        max_pauli_weight=args.max_pauli_weight,
    )
    ordered_pool = list(OrderedDict.fromkeys([*hct_masks, *base_pool]))
    pool_before_cap = len(ordered_pool)
    pool_scores = {
        mask: -commutator_l1_for_mask(mask, terms) for mask in ordered_pool
    }
    if len(ordered_pool) > args.max_candidate_pool_size:
        position = {mask: index for index, mask in enumerate(ordered_pool)}
        ordered_pool.sort(
            key=lambda mask: (pool_scores[mask], -position[mask]), reverse=True
        )
        ordered_pool = ordered_pool[: args.max_candidate_pool_size]
    print(
        f"Beam pool: {pool_before_cap} -> {len(ordered_pool)} candidates; "
        f"pairwise_seed_terms={args.pairwise_seed_terms}.",
        flush=True,
    )

    beam_sets = {}
    def beam_score(generators):
        masks = qubitops_to_masks(generators, n_qubits)
        total = 0.0
        for mask in masks:
            if mask not in pool_scores:
                pool_scores[mask] = -commutator_l1_for_mask(mask, terms)
            total += pool_scores[mask]
        return total

    for name, target in ((BEAM_HALF, n_qubits // 2), (BEAM_FULL, n_qubits)):
        print(f"Finding {target} Beam generators for {name}.", flush=True)
        start = perf_counter()
        generators = beam_search_symmetries(
            data["hamiltonian"],
            ordered_pool,
            target_rank=target,
            n_qubits=n_qubits,
            beam_width=args.beam_width,
            heavy_core_fraction=args.heavy_core_fraction,
            score_func=beam_score,
            score_is_separable=True,
            separable_score_cache=pool_scores.copy(),
            n_processes=args.n_processes,
            mp_start_method=args.mp_start_method,
        )
        validation = validate_symmetry_generators(
            data["hamiltonian"], generators, n_qubits=n_qubits
        )
        validation["target_rank"] = target
        beam_sets[name] = {
            "symmetries": encode_symmetries(generators),
            "score": beam_score(generators),
            "score_name": (
                "negative sum of individual Pauli commutator 1-norms"
            ),
            "score_convention": "maximize",
            "seconds": perf_counter() - start,
            "validation": validation,
        }

    hct_sets = {}
    for name, generators in (
        (HCT_HALF, hct_full[: n_qubits // 2]),
        (HCT_FULL, hct_full),
    ):
        masks = qubitops_to_masks(generators, n_qubits)
        validation = validate_symmetry_generators(
            data["hamiltonian"], generators, n_qubits=n_qubits
        )
        validation["target_rank"] = len(generators)
        hct_sets[name] = {
            "symmetries": encode_symmetries(generators),
            "individual_scores": [
                commutator_l1_for_mask(mask, terms) for mask in masks
            ],
            "total_score": sum(
                commutator_l1_for_mask(mask, terms) for mask in masks
            ),
            "score_name": "sum of Pauli 1-norms of individual commutators",
            "score_convention": "minimize",
            "epsilons": hct_epsilons[: len(generators)],
            "validation": validation,
        }

    payload = {
        "n_qubits": n_qubits,
        "hct": hct_sets,
        "beam": beam_sets,
        "hct_full_search_seconds": hct_seconds,
        "hct_threshold_schedule": (
            "coefficient_values"
            if args.hct_use_coefficient_thresholds
            else f"{args.hct_intervals} linear intervals"
        ),
        "candidate_pool": {
            "masks": [[int(x), int(z)] for x, z in ordered_pool],
            "size_before_cap": pool_before_cap,
            "size_after_cap": len(ordered_pool),
            "maximum_size": args.max_candidate_pool_size,
            "ranking_score": (
                "negative individual Pauli commutator 1-norm"
            ),
            "ranking_convention": "maximize",
            "pairwise_seed_terms": args.pairwise_seed_terms,
        },
    }
    save_json(path, payload)
    return payload


def _frame_symmetries(symmetry_data: dict, frame: str):
    """Return decoded generators for one non-raw benchmark frame."""
    if frame == SENIORITY:
        return None
    source = "hct" if frame.startswith("HCT") else "beam"
    key = frame.replace("_Fiedler", "")
    return decode_symmetries(symmetry_data[source][key]["symmetries"])


def prepare_frames(
    data: dict,
    symmetry_data: dict,
    reference_tensors,
    reference_validation: dict,
    args,
) -> dict:
    """Synthesize frame unitaries and calculate MPS entropy/Fiedler data.

    Parameters
    ----------
    data
        Loaded Hamiltonian input.
    symmetry_data
        Saved HCT and Beam output from :func:`prepare_symmetries`.
    reference_tensors
        Portable MPS tensors from the selected largest-bond reference.
    reference_validation
        Reference-energy/state provenance saved with every frame.
    args
        Clifford, MPS-transformation, Fiedler, and output settings.

    Returns
    -------
    frames
        Mapping from frame name to serialized Clifford/permutation data,
        reference-MPS cut entropies, and transformation diagnostics.
    """
    path = args.output_dir / "prepared" / "frames.json"
    if path.exists() and not args.force_stage:
        frames = load_json(path)
        # Energy refinement does not change the saved reference MPS,
        # Clifford circuits, entropies, or Fiedler permutations. Refresh only
        # their provenance instead of recomputing those expensive objects.
        for frame in frames.values():
            frame["reference"] = reference_validation
        save_json(path, frames)
        return frames

    n_qubits = data["n_qubits"]
    frames = {}
    if RAW in args.frames:
        raw_entropies = qubit_mps_cut_entropies(reference_tensors, base=np.e)
        frames[RAW] = {
            "unitaries": [],
            "reference_cut_entropies": raw_entropies,
            "reference": reference_validation,
        }
    requested = set(args.frames)
    symmetry_frames = []
    if SENIORITY in requested:
        symmetry_frames.append(SENIORITY)
    if HCT_HALF in requested:
        symmetry_frames.append(HCT_HALF)
    if HCT_FULL in requested or HCT_FIEDLER in requested:
        symmetry_frames.append(HCT_FULL)
    if BEAM_HALF in requested:
        symmetry_frames.append(BEAM_HALF)
    if BEAM_FULL in requested or BEAM_FIEDLER in requested:
        symmetry_frames.append(BEAM_FULL)
    for frame in symmetry_frames:
        print(f"Preparing Clifford and MPS entropies for {frame}.", flush=True)
        generators = (
            get_seniority_symmetries(n_qubits)
            if frame == SENIORITY
            else _frame_symmetries(symmetry_data, frame)
        )
        clifford = Clifford.from_symmetries(
            generators,
            n_qubits=n_qubits,
            symmetry_qubits_first=True,
            synthesis_basis="Z",
            generator_mapping="positive_z",
        )
        transformed, transform_info = transform_qubit_mps_arrays(
            reference_tensors,
            unitaries=(clifford,),
            max_bond=args.reference_transform_max_bond,
            cutoff=args.mps_cutoff,
        )
        frames[frame] = {
            "unitaries": [{"kind": "clifford", "data": clifford.to_dict()}],
            "n_symmetries": len(generators),
            "reference_cut_entropies": qubit_mps_cut_entropies(
                transformed, base=np.e
            ),
            "reference_transform": transform_info,
            "reference": reference_validation,
        }
        target = HCT_FIEDLER if frame == HCT_FULL else BEAM_FIEDLER
        if frame in {HCT_FULL, BEAM_FULL} and target in requested:
            fiedler = fiedler_order_from_mps(
                transformed,
                base=np.e,
                component_order=args.fiedler_component_order,
            )
            ordering = [int(value) for value in fiedler["ordering"]]
            permutation = tuple(int(value) for value in invert_ordering(ordering))
            reordered, reorder_info = transform_qubit_mps_arrays(
                transformed,
                unitaries=(PermutationUnitary(permutation),),
                max_bond=args.reference_transform_max_bond,
                cutoff=args.mps_cutoff,
            )
            frames[target] = {
                "unitaries": [
                    {"kind": "clifford", "data": clifford.to_dict()},
                    {"kind": "permutation", "old_to_new": list(permutation)},
                ],
                "n_symmetries": len(generators),
                "fiedler": {
                    "ordering": ordering,
                    "old_to_new": list(permutation),
                    "one_qubit_entropies": fiedler["one_qubit_entropies"],
                    "two_qubit_entropies": fiedler["two_qubit_entropies"],
                    "mutual_information": fiedler["mutual_information"],
                    "components": fiedler["components"],
                    "method": "direct contractions of transformed DMRG MPS",
                },
                "reference_cut_entropies": qubit_mps_cut_entropies(
                    reordered, base=np.e
                ),
                "reference_transform": transform_info,
                "reference_reordering_transform": reorder_info,
                "reference": reference_validation,
            }
            del reordered
        del transformed
        gc.collect()
    save_json(path, frames)
    return frames


def construct_frame(data: dict, frame_data: dict):
    """Construct a frame Hamiltonian and its ordered MPS-unitary objects.

    Parameters
    ----------
    data
        Loaded raw packed Hamiltonian.
    frame_data
        One entry from the saved ``frames.json`` mapping.

    Returns
    -------
    hamiltonian, unitaries
        Streamed Hamiltonian in the requested frame and the unitary sequence
        that maps the raw CISD warm-start MPS into the same frame.
    """
    hamiltonian = data["hamiltonian"]
    unitaries = []
    for item in frame_data["unitaries"]:
        if item["kind"] == "clifford":
            unitary = Clifford.from_dict(item["data"])
            hamiltonian = unitary.transform(hamiltonian)
        elif item["kind"] == "permutation":
            unitary = PermutationUnitary(tuple(item["old_to_new"]))
            hamiltonian = permute_qubits_in_qubit_operator(
                hamiltonian, unitary.permutation
            )
        else:
            raise ValueError(f"unknown frame unitary {item['kind']!r}")
        unitaries.append(unitary)
    return hamiltonian, tuple(unitaries)


def run_dmrg_frames(
    data: dict,
    cisd_energy: float,
    cisd_state,
    reference_validation: dict,
    frames: dict,
    args,
) -> tuple[list[dict], dict]:
    """Run requested frames sequentially and checkpoint after each one.

    Parameters
    ----------
    data
        Loaded raw Hamiltonian input.
    cisd_energy, cisd_state
        Raw-frame determinant-sparse CISD warm start.
    reference_validation
        Selected reference energy and its validation metadata.
    frames
        Prepared frame definitions from :func:`prepare_frames`.
    args
        DMRG grid, convergence, memory, artifact, and resume settings.

    Returns
    -------
    rows, summaries
        All per-bond benchmark rows and per-frame summary dictionaries.
    """
    benchmark_dir = args.output_dir / "dmrg"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = {}
    for frame in args.frames:
        frame_dir = benchmark_dir / frame
        result_path = frame_dir / "result.json"
        curve_path = frame_dir / "dmrg_curve.csv"
        if result_path.exists() and curve_path.exists() and not args.force_stage:
            cached_summary = load_json(result_path)
            cached_reference = cached_summary.get(
                "benchmark_reference", {}
            ).get("energy")
            if cached_reference is not None and np.isclose(
                float(cached_reference),
                float(reference_validation["energy"]),
                rtol=0.0,
                atol=1e-12,
            ):
                print(f"Reusing completed DMRG frame {frame}.", flush=True)
                summaries[frame] = cached_summary
                all_rows.extend(read_csv(curve_path))
                continue
            print(
                f"DMRG frame {frame} used the previous energy reference; "
                "rerunning its stopping curve with the refined reference.",
                flush=True,
            )

        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running DMRG frame {frame}.{rss_message()}", flush=True)
        rss_before = rss_gib()
        peak_before = peak_rss_gib()
        if frame == RAW_FERMION:
            if args.frozen_core_orbitals:
                raise NotImplementedError(
                    "the fermionic frame currently requires the full orbital space"
                )
            original_output = args.output_dir
            args.output_dir = frame_dir
            args.sweep_tol = args.sweep_tolerance
            args.dmrg_tol = args.dmrg_tolerance
            args.initial_state = "cisd"
            args.save_tensor_networks = True
            try:
                rows, summary = fermionic_backend.run_fermionic_dmrg_curve(
                    molecule=data["molecule"],
                    warm_start_state=cisd_state,
                    warm_start_energy=cisd_energy,
                    fci_energy=reference_validation["energy"],
                    args=args,
                )
            finally:
                args.output_dir = original_output
            hamiltonian = unitaries = None
        else:
            hamiltonian, unitaries = construct_frame(data, frames[frame])
            rows, summary = run_block2_qubit_dmrg_curve(
                label=frame,
                hamiltonian=hamiltonian,
                sparse_state=(cisd_state.indices, cisd_state.coeffs),
                exact_energy=reference_validation["energy"],
                warm_start_energy=cisd_energy,
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
                n_mkl_threads=args.n_mkl_threads,
                stack_mem_gb=args.stack_mem_gb,
                davidson_threshold=args.davidson_threshold,
                warm_start_noises=args.warm_start_noises,
                verbose=args.verbose,
                artifact_dir=(frame_dir / "tensor_networks"),
            )
        for row in rows:
            row.update(
                {
                    "system": "N2_631g",
                    "frame": frame,
                    "basis": data["manifest"]["basis"],
                    "n_qubits": data["n_qubits"],
                    "reference_energy": reference_validation["energy"],
                    "reference_validated": reference_validation["validated"],
                }
            )
        summary["benchmark_reference"] = reference_validation
        summary["memory_gib"] = {
            "rss_before": rss_before,
            "rss_after": rss_gib(),
            "process_peak_before": peak_before,
            "process_peak_after": peak_rss_gib(),
        }
        summary["frame_preparation"] = (
            {"representation": "spin-adapted SU(2) fermionic MPO"}
            if frame == RAW_FERMION
            else frames[frame]
        )
        write_csv(curve_path, rows)
        save_json(result_path, summary)
        all_rows.extend(rows)
        summaries[frame] = summary
        del hamiltonian, unitaries
        gc.collect()
    write_csv(args.output_dir / "dmrg_curves.csv", all_rows)
    save_json(args.output_dir / "dmrg_summaries.json", summaries)
    write_csv(
        args.output_dir / "dmrg_summary.csv",
        [
            {
                "system": "N2_631g",
                "frame": frame,
                "first_converged_bond_dim": summary.get(
                    "first_converged_bond_dim"
                ),
                "converged_within_grid": summary.get(
                    "converged_within_grid", False
                ),
                "mpo_bond_dimension": summary.get("mpo_bond_dimension"),
                "mpo_build_seconds": summary.get("mpo_build_seconds"),
                "first_converged_dmrg_seconds": summary.get(
                    "first_converged_dmrg_optimization_seconds"
                ),
            }
            for frame, summary in summaries.items()
        ],
    )
    return all_rows, summaries


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    args
        Populated benchmark ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frozen-core-orbitals", type=int, default=0)
    parser.add_argument(
        "--stage",
        choices=("cisd", "reference", "symmetries", "frames", "dmrg", "all"),
        default="all",
        help="Run this stage and any prerequisites; 'all' ends with DMRG.",
    )
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--frames", nargs="+", choices=ALL_FRAMES, default=list(ALL_FRAMES))

    parser.add_argument("--cisd-tolerance", type=float, default=1e-10)
    parser.add_argument("--cisd-max-cycle", type=int, default=100)
    parser.add_argument("--cisd-coefficient-tolerance", type=float, default=0.0)

    parser.add_argument("--reference-bond-dims", type=int, nargs="+", default=[100, 150])
    parser.add_argument(
        "--skip-reference-bond-dims",
        type=int,
        nargs="*",
        default=[150],
        help=(
            "requested fixed reference dimensions to omit; defaults to 150 "
            "because that calculation exceeds practical memory for N2/6-31G"
        ),
    )
    parser.add_argument("--reference-sweeps", type=int, default=100)
    parser.add_argument("--reference-sweep-tolerance", type=float, default=1e-8)
    parser.add_argument("--reference-validation-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--energy-reference-bond-dim",
        type=int,
        default=200,
        help="base SU(2) fermionic bond dimension for the energy reference",
    )
    parser.add_argument(
        "--energy-reference-bond-increment",
        type=int,
        default=10,
        help="additional bond dimension used to validate the energy reference",
    )
    parser.add_argument(
        "--require-reference-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "stop before benchmarks unless both high-bond reference runs "
            "converge and agree within the reference tolerance"
        ),
    )
    parser.add_argument("--reference-transform-max-bond", type=int, default=150)

    parser.add_argument("--hct-intervals", type=int, default=100)
    parser.add_argument("--hct-term-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--hct-use-coefficient-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--heavy-core-fraction", type=float, default=0.95)
    parser.add_argument("--max-candidates-from-terms", type=int, default=256)
    parser.add_argument("--pairwise-seed-terms", type=int, default=50)
    parser.add_argument("--max-candidate-pool-size", type=int, default=1000)
    parser.add_argument("--max-pauli-weight", type=int, default=None)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mp-start-method", default=None)
    parser.add_argument(
        "--fiedler-component-order",
        choices=("index", "total_weight", "size"),
        default="index",
    )

    parser.add_argument("--bond-dims", type=int, nargs="+", default=list(DEFAULT_BOND_DIMS))
    parser.add_argument("--dmrg-sweeps", type=int, default=100)
    parser.add_argument("--dmrg-tolerance", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--sweep-tolerance", type=float, default=1e-6)
    parser.add_argument("--davidson-threshold", type=float, default=1e-10)
    parser.add_argument("--warm-start-noises", type=float, nargs="*", default=list(DEFAULT_NOISES))
    parser.add_argument("--mps-cutoff", type=float, default=1e-13)
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument("--mpo-builder", choices=("blocked_sum", "expression"), default="blocked_sum")
    parser.add_argument("--sum-mpo-mod", type=int, default=10)
    parser.add_argument("--sparse-batch-size", type=int, default=32)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--n-mkl-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=0.5)
    parser.add_argument("--full-curve", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    """Validate settings whose errors would invalidate saved benchmark data.

    Parameters
    ----------
    args
        Parsed benchmark namespace returned by :func:`parse_args`.

    Returns
    -------
    None
        Raises ``ValueError`` when a dimension, pool size, frame list, or noise
        schedule is invalid.
    """
    if any(value < 1 for value in args.reference_bond_dims):
        raise ValueError("reference bond dimensions must be positive")
    if any(value < 1 for value in args.skip_reference_bond_dims):
        raise ValueError("skipped reference bond dimensions must be positive")
    if args.energy_reference_bond_dim < 1:
        raise ValueError("energy reference bond dimension must be positive")
    if args.energy_reference_bond_increment < 1:
        raise ValueError("energy reference bond increment must be positive")
    if any(value < 1 for value in args.bond_dims):
        raise ValueError("benchmark bond dimensions must be positive")
    if len(set(args.frames)) != len(args.frames):
        raise ValueError("frames must not contain duplicates")
    if args.max_candidate_pool_size < 1:
        raise ValueError("max candidate pool size must be positive")
    if args.pairwise_seed_terms < 0:
        raise ValueError("pairwise seed terms must be nonnegative")
    if any(value < 0 for value in args.warm_start_noises):
        raise ValueError("warm-start noises must be nonnegative")
    if args.n_threads < 1 or args.n_mkl_threads < 1:
        raise ValueError("n_threads and n_mkl_threads must be positive")


EXECUTION_ONLY_SETTING_KEYS = frozenset(
    {
        "n_threads",
        "n_mkl_threads",
        "n_processes",
        "mp_start_method",
        "stack_mem_gb",
        "verbose",
        "skip_reference_bond_dims",
        "require_reference_validation",
    }
)


def partition_checkpoint_settings(settings: dict) -> tuple[dict, dict]:
    """Separate result-defining settings from execution-only controls.

    Parameters
    ----------
    settings
        Complete JSON-compatible benchmark settings for one invocation.

    Returns
    -------
    preparation_settings, execution_settings
        ``preparation_settings`` contains values that define saved scientific
        data or requested benchmark results. ``execution_settings`` contains
        thread, process, memory, verbosity, and stage-selection controls that
        may change without invalidating unrelated reusable checkpoints.
    """
    preparation = {
        key: value
        for key, value in settings.items()
        if key not in EXECUTION_ONLY_SETTING_KEYS
    }
    execution = {
        key: settings[key]
        for key in EXECUTION_ONLY_SETTING_KEYS
        if key in settings
    }
    return preparation, execution


def checkpoint_settings_payload(settings: dict) -> dict:
    """Create the versioned settings record saved by this benchmark.

    Parameters
    ----------
    settings
        Complete JSON-compatible benchmark settings for the current run.

    Returns
    -------
    payload
        Versioned mapping with separate ``preparation`` and ``execution``
        sections, suitable for writing to ``settings.json``.
    """
    preparation, execution = partition_checkpoint_settings(settings)
    return {
        "format": "n2_631g_benchmark_settings_v2",
        "preparation": preparation,
        "execution": execution,
    }


def preparation_settings_from_saved(payload: dict) -> dict:
    """Read comparable preparation settings from old or new saved data.

    Parameters
    ----------
    payload
        Mapping loaded from ``settings.json``. Both the original flat format
        and the version-two partitioned format are accepted.

    Returns
    -------
    preparation_settings
        Result-defining settings with execution-only controls removed. This
        permits transparent migration of existing benchmark directories.
    """
    if payload.get("format") == "n2_631g_benchmark_settings_v2":
        preparation = dict(payload["preparation"])
    else:
        preparation = partition_checkpoint_settings(payload)[0]
    preparation.pop("require_reference_validation", None)
    # These defaults were introduced after the original N2/6-31G run. Treat
    # an absent value as the new default so existing scientific checkpoints
    # migrate without --force-stage.
    preparation.setdefault("energy_reference_bond_dim", 200)
    preparation.setdefault("energy_reference_bond_increment", 10)
    return preparation


def main() -> None:
    """Execute the selected checkpointed stage and its prerequisites.

    Returns
    -------
    None
        Results are printed and written below ``--output-dir``.
    """
    args = parse_args()
    validate_args(args)
    args.probe_dir = args.probe_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_probe_input(args.probe_dir, args.frozen_core_orbitals)

    invocation_settings = vars(args).copy()
    invocation_settings.pop("stage")
    invocation_settings.pop("force_stage")
    invocation_settings.update(
        {
            "system": "N2_631g",
            "n_qubits": data["n_qubits"],
            "probe_manifest": str(data["manifest_path"]),
            "fci_used": False,
            "symmetry_scoring_state": None,
            "frames_processed_sequentially": True,
        }
    )
    invocation_settings = to_jsonable(invocation_settings)
    settings = checkpoint_settings_payload(invocation_settings)
    settings_path = args.output_dir / "settings.json"
    if settings_path.exists() and not args.force_stage:
        prior = load_json(settings_path)
        if preparation_settings_from_saved(prior) != settings["preparation"]:
            raise ValueError(
                "Saved result-defining settings differ from this invocation; "
                "use a new --output-dir or --force-stage. Thread, process, "
                "memory, and verbosity settings may be changed freely."
            )
    # Always update execution controls and transparently migrate old flat
    # settings files after result-defining settings have been validated.
    save_json(settings_path, settings)

    cisd_energy = cisd_state = cisd_metadata = None
    reference_tensors = reference_validation = None
    symmetry_data = frame_data = None

    if args.stage in {"cisd", "reference", "frames", "dmrg", "all"}:
        cisd_energy, cisd_state, cisd_metadata = prepare_cisd(data, args)
    if args.stage == "cisd":
        return

    if args.stage in {"reference", "frames", "dmrg", "all"}:
        reference_tensors, reference_validation = prepare_references(
            data, cisd_energy, cisd_state, args
        )
    if args.stage == "reference":
        return

    searched_frames = {
        HCT_HALF,
        HCT_FULL,
        BEAM_HALF,
        BEAM_FULL,
        HCT_FIEDLER,
        BEAM_FIEDLER,
    }
    needs_symmetries = bool(set(args.frames) & searched_frames)
    if args.stage == "symmetries" or (
        args.stage in {"frames", "dmrg", "all"} and needs_symmetries
    ):
        symmetry_data = prepare_symmetries(data, args)
    if args.stage == "symmetries":
        return

    if args.stage in {"frames", "dmrg", "all"}:
        frame_data = prepare_frames(
            data,
            symmetry_data,
            reference_tensors,
            reference_validation,
            args,
        )
    if args.stage == "frames":
        return

    rows, summaries = run_dmrg_frames(
        data,
        cisd_energy,
        cisd_state,
        reference_validation,
        frame_data,
        args,
    )
    save_json(
        args.output_dir / "benchmark.json",
        {
            "system": "N2_631g",
            "settings": settings,
            "probe": data["manifest"],
            "cisd": cisd_metadata,
            "reference": reference_validation,
            "symmetries": symmetry_data,
            "frames": frame_data,
            "dmrg_summaries": summaries,
            "dmrg_rows": len(rows),
        },
    )
    print(f"Completed benchmark outputs in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
