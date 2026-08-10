#!/usr/bin/env python3
"""Run focused pyblock2 DMRG curves for saved N2/6-31G Beam symmetries.

The script consumes the corrected CISD states and ``Beam_N_CommSqCISD``
symmetry sets already prepared for the equilibrium, correlated, and
dissociated geometries.  It verifies their state and Hamiltonian fingerprints,
synthesizes a Z-basis Clifford, transforms the streamed Pauli Hamiltonian, and
uses the same Clifford to transform the CISD warm start.

Because these saved generators contain only Z Pauli factors, the Clifford is a
support-preserving basis circuit.  The warm start is therefore transformed by
direct bit-string and coefficient manipulations before MPS construction.  The
general tensor-gate route, including its robust SVD fallback, remains available
inside ``sparse_state_to_block2_pauli_mps`` if a future symmetry set requires
support-growing gates.

Before each qubit curve, independently initialized fermionic SU(2) calculations
at M=300 and M=400 establish the energy reference. The lower variational energy
is accepted only if both runs satisfy the sweep-energy convergence test and the
two energies agree within 1e-4 Hartree. Both runs are checkpointed separately.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

import _bootstrap

from quasisymmetries.block2_qubit_benchmark import (
    run_block2_qubit_dmrg_curve,
)
from quasisymmetries.clifford_symmetry_optimized import Clifford
from quasisymmetries.mps_unitary import (
    can_transform_sparse_state_without_support_growth,
)
from quasisymmetries.save import (
    decode_qubit_operator,
    load_json,
    load_sparse_qubit_state,
    save_json,
    write_csv,
)

import benchmark_n2_631g_pyblock2 as common


ROOT = Path(__file__).resolve().parents[1]
FRAME = "Beam_N_CommSqCISD"
CHEMICAL_ACCURACY = 1.6e-3
DEFAULT_NOISES = (1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6)
DEFAULT_OUTPUT_DIR = ROOT / "saved" / "results" / "n2_631g_beam_dmrg"
GEOMETRIES = {
    "eqm": {
        "bond_length_angstrom": 1.2,
        "corrected_state_fingerprint": (
            "6f26f0872a5f1509d96ba140b3fceea68d116aa21c3b0ebb4563013a1d5afdbb"
        ),
        "corrected_hamiltonian_fingerprint": (
            "ad620e7ccd4758dcec5efb7633346016bef14b706333789af82aa7f3fb266215"
        ),
        "probe_dir": ROOT / "saved" / "results" / "n2_631g_mpo_probe",
        "benchmark_dir": ROOT / "saved" / "results" / "n2_631g_benchmark",
    },
    "corr": {
        "bond_length_angstrom": 1.4,
        "corrected_state_fingerprint": (
            "1d531b24bf0c4733192983d7f13e790817f5d7daec2052cd6f5f741d8b6da4ff"
        ),
        "corrected_hamiltonian_fingerprint": (
            "8006ca0f3e6dfeed6e97e8750691f13bdb17ad4fba576ea08e73f0534541e8c3"
        ),
        "probe_dir": (
            ROOT
            / "saved"
            / "results"
            / "n2_631g_beam_geometries"
            / "N2_corr"
            / "probe"
        ),
        "benchmark_dir": (
            ROOT
            / "saved"
            / "results"
            / "n2_631g_beam_geometries"
            / "N2_corr"
            / "benchmark"
        ),
    },
    "diss": {
        "bond_length_angstrom": 2.2,
        "corrected_state_fingerprint": (
            "b79fceafe3f42d65f871bc46e4e7a60efefffda1378c0001b671a61ba77da730"
        ),
        "corrected_hamiltonian_fingerprint": (
            "162263f8dafc9f80d6cbaba673b6eb323526e923343ac5ad81db1bcd925ea7a9"
        ),
        "probe_dir": (
            ROOT
            / "saved"
            / "results"
            / "n2_631g_beam_geometries"
            / "N2_diss"
            / "probe"
        ),
        "benchmark_dir": (
            ROOT
            / "saved"
            / "results"
            / "n2_631g_beam_geometries"
            / "N2_diss"
            / "benchmark"
        ),
    },
}


def default_bond_dimensions(maximum: int) -> tuple[int, ...]:
    """Return the default DMRG bond-dimension grid through ``maximum``.

    Parameters
    ----------
    maximum
        Largest requested MPS bond dimension.  Values through 100 use the
        established coarse grid; values above 100 are added in steps of 20.

    Returns
    -------
    bond_dimensions
        Strictly increasing tuple that always includes ``maximum``.
    """
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError("max-bond-dim must be positive")
    low_grid = (10, 20, 30, 40, 60, 80, 100)
    values = [value for value in low_grid if value <= maximum]
    if maximum > 100:
        values.extend(range(120, maximum + 1, 20))
    if not values or values[-1] != maximum:
        values.append(maximum)
    return tuple(sorted(set(values)))


def next_adaptive_bond_dimension(
    completed_rows: list[dict] | tuple[dict, ...],
    candidates: tuple[int, ...],
) -> int | None:
    """Choose the next bond dimension in the adaptive accuracy search.

    Parameters
    ----------
    completed_rows
        Previously completed DMRG rows. ``within_dmrg_tolerance=True`` means
        that the bond dimension is sufficient for chemical accuracy.
    candidates
        Strictly increasing discrete bond dimensions. For the default grid the
        search begins at 10 and 100, then selects 60 after success at 100 or
        200 after failure at 100.

    Returns
    -------
    bond_dimension
        Next untested candidate, or ``None`` once the lowest sufficient grid
        point has been bracketed or the largest candidate has failed.

    Notes
    -----
    The search assumes that attainable variational accuracy is monotone with
    maximum MPS bond dimension. DMRG local-minimum effects can violate this in
    practice; all tested energies and sweep-convergence flags are retained so
    such violations remain visible.
    """
    candidates = tuple(int(value) for value in candidates)
    if not candidates or tuple(sorted(set(candidates))) != candidates:
        raise ValueError("candidates must be nonempty, increasing, and unique")
    outcomes: dict[int, bool] = {}
    for row in completed_rows:
        bond_dim = int(row["bond_dim"])
        if bond_dim in outcomes:
            raise ValueError(f"duplicate completed bond dimension {bond_dim}")
        if bond_dim not in candidates:
            raise ValueError(f"completed bond dimension {bond_dim} is off-grid")
        outcome = row.get("within_dmrg_tolerance")
        if outcome is None:
            raise ValueError(
                "adaptive bond search requires a validated reference energy"
            )
        outcomes[bond_dim] = bool(outcome)

    minimum = candidates[0]
    if minimum not in outcomes:
        return minimum
    if outcomes[minimum]:
        return None

    anchor = min(candidates, key=lambda value: (abs(value - 100), value))
    if anchor not in outcomes:
        return anchor

    successful = sorted(
        bond_dim for bond_dim, sufficient in outcomes.items() if sufficient
    )
    if successful:
        upper = successful[0]
        lower = max(
            bond_dim
            for bond_dim, sufficient in outcomes.items()
            if not sufficient and bond_dim < upper
        )
        interior = [
            value
            for value in candidates
            if lower < value < upper and value not in outcomes
        ]
        if not interior:
            return None
        target = (lower + upper) / 2.0
        return min(interior, key=lambda value: (abs(value - target), value))

    highest_failed = max(outcomes)
    remaining = [
        value
        for value in candidates
        if value > highest_failed and value not in outcomes
    ]
    if not remaining:
        return None
    target = (highest_failed + candidates[-1]) / 2.0
    return min(remaining, key=lambda value: (abs(value - target), value))


def parse_reference_energies(items: list[str]) -> dict[str, float]:
    """Parse ``GEOMETRY=ENERGY`` command-line reference specifications.

    Parameters
    ----------
    items
        Strings such as ``eqm=-109.1``. Geometry names must be one of
        ``eqm``, ``corr``, or ``diss``.

    Returns
    -------
    references
        Mapping from geometry name to finite reference energy in Hartree.
    """
    references: dict[str, float] = {}
    for item in items:
        try:
            geometry, value = item.split("=", 1)
            energy = float(value)
        except ValueError as exc:
            raise ValueError(
                f"invalid --reference-energy {item!r}; use GEOMETRY=ENERGY"
            ) from exc
        geometry = geometry.strip()
        if geometry not in GEOMETRIES:
            raise ValueError(f"unknown reference geometry {geometry!r}")
        if not np.isfinite(energy):
            raise ValueError("reference energies must be finite")
        references[geometry] = energy
    return references


def prepare_fermionic_su2_reference(
    data: dict,
    geometry_dir: Path,
    args,
) -> tuple[float | None, dict]:
    """Run/reload and validate the M=300 and M=400 fermionic references.

    Parameters
    ----------
    data
        Verified geometry inputs returned by :func:`load_verified_inputs`.
    geometry_dir
        Geometry-specific output directory. Each fixed-bond result is
        checkpointed below ``reference/`` immediately after it finishes.
    args
        Parsed execution, sweep, noise, and reference-validation options.

    Returns
    -------
    energy, validation
        The lower variational energy of the two runs when both runs satisfy
        the sweep stopping criterion and their energies differ by no more than
        ``reference_validation_tolerance``. Otherwise ``energy`` is ``None``;
        all rows and the failed validation diagnostics are still returned and
        saved.
    """
    reference_dir = geometry_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_settings = {
        "bond_dims": list(args.reference_bond_dims),
        "max_sweeps": int(args.max_sweeps),
        "sweep_tolerance": float(args.sweep_tolerance),
        "validation_tolerance": float(args.reference_validation_tolerance),
        "davidson_threshold": float(args.davidson_threshold),
        "warm_start_noises": list(args.warm_start_noises),
        "state_fingerprint": data["state_fingerprint"],
        "hamiltonian_fingerprint": data["hamiltonian_fingerprint"],
        "jw_phase_convention": data["cisd_metadata"].get(
            "jw_phase_convention"
        ),
    }
    rows = []
    run_summaries = {}
    for bond_dim in args.reference_bond_dims:
        result_path = reference_dir / f"fermionic_su2_M{bond_dim}.json"
        cached = load_json(result_path) if result_path.exists() else None
        if (
            cached is not None
            and not args.force_reference
            and cached.get("settings") == reference_settings
        ):
            print(
                f"N2 {data['geometry']}: reusing fermionic SU(2) "
                f"reference M={bond_dim}, E={cached['row']['energy']:.12f}.",
                flush=True,
            )
            row = cached["row"]
            summary = cached["summary"]
        else:
            print(
                f"N2 {data['geometry']}: running fermionic SU(2) "
                f"reference M={bond_dim}.",
                flush=True,
            )
            backend_args = SimpleNamespace(
                output_dir=reference_dir / f"M{bond_dim}",
                bond_dims=[int(bond_dim)],
                dmrg_sweeps=int(args.max_sweeps),
                sweep_tol=float(args.sweep_tolerance),
                dmrg_tol=float(args.dmrg_tolerance),
                initial_state="cisd",
                save_tensor_networks=False,
                full_curve=True,
                n_threads=int(args.n_threads),
                n_mkl_threads=int(args.n_mkl_threads),
                stack_mem_gb=float(args.stack_mem_gb),
                davidson_threshold=float(args.davidson_threshold),
                warm_start_noises=tuple(args.warm_start_noises),
                verbose=bool(args.verbose),
            )
            reference_rows, summary = common.fermionic_backend.run_fermionic_dmrg_curve(
                molecule=data["molecule"],
                warm_start_state=data["cisd_state"],
                warm_start_energy=data["cisd_energy"],
                fci_energy=None,
                args=backend_args,
            )
            if len(reference_rows) != 1:
                raise RuntimeError("expected one row from fixed-bond reference")
            row = reference_rows[0]
            save_json(
                result_path,
                {
                    "settings": reference_settings,
                    "row": row,
                    "summary": summary,
                },
            )
        rows.append(row)
        run_summaries[str(bond_dim)] = summary

    selected = common.select_fermionic_energy_reference(
        rows,
        requested_bond_dims=tuple(args.reference_bond_dims),
        validation_tolerance=args.reference_validation_tolerance,
    )
    validation = {
        **selected,
        "method": "fermionic_su2_fixed_bond_pair",
        "settings": reference_settings,
        "rows": rows,
        "run_summaries": run_summaries,
        "reference_energy_source": (
            "lower variational energy from independently converged SU(2) "
            "M=300 and M=400 calculations"
        ),
    }
    save_json(reference_dir / "reference_validation.json", validation)
    write_csv(reference_dir / "fermionic_su2_reference_curve.csv", rows)
    print(
        f"N2 {data['geometry']}: fermionic reference pair "
        f"|E(400)-E(300)|={validation['energy_difference']:.3e} Ha; "
        f"validated={validation['validated']}.",
        flush=True,
    )
    if not validation["validated"]:
        print(
            "WARNING: reference was not accepted because both fixed-bond runs "
            "must converge in sweep energy and agree within "
            f"{args.reference_validation_tolerance:.3e} Ha.",
            flush=True,
        )
        return None, validation
    return float(validation["energy"]), validation


def _single_z_term(symmetry) -> tuple:
    """Return and validate the sole Z-only Pauli term of a generator.

    Parameters
    ----------
    symmetry
        OpenFermion ``QubitOperator`` expected to contain one unit-magnitude,
        Z-only Pauli product.

    Returns
    -------
    term
        Tuple of ``(qubit, 'Z')`` factors.
    """
    if len(symmetry.terms) != 1:
        raise ValueError("each saved symmetry must contain one Pauli word")
    term, coefficient = next(iter(symmetry.terms.items()))
    if any(pauli != "Z" for _, pauli in term):
        raise ValueError("focused benchmark requires Z-only Beam symmetries")
    if not np.isclose(abs(complex(coefficient)), 1.0, atol=1e-12):
        raise ValueError("saved symmetry coefficient must have unit magnitude")
    return term


def load_verified_inputs(geometry: str) -> dict:
    """Load and cross-check one geometry's Hamiltonian, CISD state, and Beam set.

    Parameters
    ----------
    geometry
        One of ``eqm``, ``corr``, or ``diss``.

    Returns
    -------
    data
        Dictionary containing the streamed Hamiltonian, corrected sparse CISD
        state and energy, decoded generators, their saved metrics, file paths,
        and deterministic state/Hamiltonian fingerprints.
    """
    config = GEOMETRIES[geometry]
    probe = common.load_probe_input(Path(config["probe_dir"]), 0)
    prepared = Path(config["benchmark_dir"]) / "prepared"
    cisd_path = prepared / "cisd_state.npz"
    cisd_metadata_path = prepared / "cisd.json"
    symmetry_path = prepared / "symmetries_cisd_comm_sq.json"
    for path in (cisd_path, cisd_metadata_path, symmetry_path):
        if not path.exists():
            raise FileNotFoundError(f"missing required saved input: {path}")

    cisd_state = load_sparse_qubit_state(cisd_path)
    cisd_metadata = load_json(cisd_metadata_path)
    symmetry_payload = load_json(symmetry_path)
    n_qubits = int(probe["n_qubits"])
    if symmetry_payload.get("format") != common.SYMMETRY_CHECKPOINT_FORMAT:
        raise ValueError(
            f"{symmetry_path} uses stale/unknown format "
            f"{symmetry_payload.get('format')!r}"
        )
    if int(symmetry_payload.get("n_qubits", -1)) != n_qubits:
        raise ValueError("symmetry and Hamiltonian qubit counts differ")
    if int(cisd_state.n_qubits) != n_qubits:
        raise ValueError("CISD state and Hamiltonian qubit counts differ")

    state_fingerprint = common.sparse_state_fingerprint(cisd_state)
    hamiltonian_fingerprint = common.pauli_hamiltonian_fingerprint(
        probe["hamiltonian"], n_qubits
    )
    score_metadata = symmetry_payload.get("score", {})
    if score_metadata.get("state_fingerprint") != state_fingerprint:
        raise ValueError(
            f"CISD fingerprint mismatch for {geometry}; symmetry file is stale"
        )
    if score_metadata.get("hamiltonian_fingerprint") != hamiltonian_fingerprint:
        raise ValueError(
            f"Hamiltonian fingerprint mismatch for {geometry}; symmetry file is stale"
        )
    if state_fingerprint != config["corrected_state_fingerprint"]:
        raise ValueError(
            f"CISD state for {geometry} is a stale pre-fix version. Expected "
            f"corrected fingerprint {config['corrected_state_fingerprint']}, "
            f"got {state_fingerprint}. Update the repository's saved inputs "
            "before submitting the DMRG job."
        )
    if hamiltonian_fingerprint != config["corrected_hamiltonian_fingerprint"]:
        raise ValueError(
            f"Hamiltonian for {geometry} is a stale pre-fix version. Expected "
            f"corrected fingerprint "
            f"{config['corrected_hamiltonian_fingerprint']}, got "
            f"{hamiltonian_fingerprint}. Update the repository's saved inputs "
            "before submitting the DMRG job."
        )

    try:
        beam_record = symmetry_payload["beam"][FRAME]
    except KeyError as exc:
        raise KeyError(f"{symmetry_path} contains no {FRAME} result") from exc
    symmetries = [
        decode_qubit_operator(item) for item in beam_record["symmetries"]
    ]
    for symmetry in symmetries:
        _single_z_term(symmetry)
    validation = beam_record.get("validation", {})
    if len(symmetries) != n_qubits:
        raise ValueError(f"expected {n_qubits} full-rank Beam generators")
    if validation.get("independent_rank") != n_qubits:
        raise ValueError("saved Beam set is not independently full rank")
    if validation.get("pairwise_commuting") is not True:
        raise ValueError("saved Beam generators are not pairwise commuting")

    return {
        **probe,
        "geometry": geometry,
        "bond_length_angstrom": float(config["bond_length_angstrom"]),
        "benchmark_dir": Path(config["benchmark_dir"]),
        "cisd_state": cisd_state,
        "cisd_energy": float(cisd_metadata["energy"]),
        "cisd_metadata": cisd_metadata,
        "symmetries": symmetries,
        "beam_record": beam_record,
        "score_metadata": score_metadata,
        "state_fingerprint": state_fingerprint,
        "hamiltonian_fingerprint": hamiltonian_fingerprint,
        "source_paths": {
            "probe_manifest": str(probe["manifest_path"]),
            "cisd_state": str(cisd_path.resolve()),
            "cisd_metadata": str(cisd_metadata_path.resolve()),
            "symmetries": str(symmetry_path.resolve()),
        },
    }


def prepare_frame(data: dict) -> tuple[object, Clifford, dict]:
    """Construct the saved Beam frame's Clifford and transformed Hamiltonian.

    Parameters
    ----------
    data
        Verified geometry data returned by :func:`load_verified_inputs`.

    Returns
    -------
    transformed_hamiltonian, clifford, metadata
        Streamed Clifford-conjugated Hamiltonian, synthesized Clifford object,
        and construction timing/route diagnostics.
    """
    start = perf_counter()
    clifford = Clifford.from_symmetries(
        data["symmetries"],
        n_qubits=data["n_qubits"],
        symmetry_qubits_first=True,
        synthesis_basis="Z",
        generator_mapping="positive_z",
    )
    sparse_route = can_transform_sparse_state_without_support_growth(
        (clifford,), n_qubits=data["n_qubits"]
    )
    if not sparse_route:
        raise RuntimeError(
            "saved Z-only symmetries unexpectedly produced a support-growing "
            "Clifford; refusing the expensive tensor-circuit warm-start route"
        )
    transformed = clifford.transform(data["hamiltonian"])
    seconds = perf_counter() - start
    return transformed, clifford, {
        "seconds": seconds,
        "hamiltonian_terms_before": len(data["hamiltonian"].terms),
        "hamiltonian_terms_after": len(transformed.terms),
        "warm_start_route_expected": "sparse_state_then_mps",
        "support_preserving_sparse_transform": sparse_route,
        "clifford": clifford.to_dict(),
    }


def _calculation_settings(args, data: dict, reference_energy) -> dict:
    """Return result-defining settings used to validate resumed output.

    Parameters
    ----------
    args, data, reference_energy
        Parsed command-line options, verified input data, and resolved external
        reference energy.

    Returns
    -------
    settings
        JSON-serializable calculation settings. Execution-only thread, memory,
        scratch, and verbosity values are deliberately excluded.
    """
    return {
        "format": "n2_631g_beam_dmrg_v2",
        "geometry": data["geometry"],
        "frame": FRAME,
        "n_qubits": data["n_qubits"],
        "state_fingerprint": data["state_fingerprint"],
        "hamiltonian_fingerprint": data["hamiltonian_fingerprint"],
        "bond_dimensions": list(args.bond_dims),
        "bond_search": "adaptive_bracketed_binary_v1",
        "max_sweeps": args.max_sweeps,
        "sweep_tolerance": args.sweep_tolerance,
        "dmrg_tolerance": args.dmrg_tolerance,
        "davidson_threshold": args.davidson_threshold,
        "warm_start_noises": list(args.warm_start_noises),
        "sparse_batch_size": args.sparse_batch_size,
        "sparse_compression_cutoff": args.sparse_compression_cutoff,
        "transform_cutoff": args.transform_cutoff,
        "mpo_cutoff": args.mpo_cutoff,
        "mpo_builder": args.mpo_builder,
        "sum_mpo_mod": args.sum_mpo_mod,
        "reference_energy": reference_energy,
    }


def _decorate_row(row: dict, data: dict, reference_source: dict) -> dict:
    """Attach geometry, score, and provenance fields to a DMRG result row.

    Parameters
    ----------
    row, data, reference_source
        Backend per-bond result, verified geometry inputs, and reference-energy
        provenance.

    Returns
    -------
    decorated_row
        Copy suitable for the per-geometry and aggregate result tables.
    """
    return {
        "system": "N2_6-31G",
        "geometry": data["geometry"],
        "bond_length_angstrom": data["bond_length_angstrom"],
        "symmetry_method": FRAME,
        "number_of_symmetries": len(data["symmetries"]),
        "symmetry_score": data["beam_record"].get("score"),
        "symmetry_total_cost": data["beam_record"].get("total_cost"),
        "score_name": data["beam_record"].get("score_name"),
        "reference_energy_source": reference_source.get("source"),
        "reference_energy_validated": reference_source.get("validated", False),
        **row,
    }


def run_geometry(
    geometry: str,
    args,
    explicit_references: dict[str, float],
) -> tuple[list[dict], dict]:
    """Run or resume the adaptive Beam DMRG search for one N2 geometry.

    Parameters
    ----------
    geometry, args, explicit_references
        Geometry label, parsed command-line options, and explicit reference
        energy mapping.

    Returns
    -------
    rows, summary
        Complete decorated bond-dimension curve and geometry-level metadata.
    """
    geometry_start = perf_counter()
    data = load_verified_inputs(geometry)
    geometry_dir = args.output_dir / f"N2_{geometry}"
    curve_json = geometry_dir / "dmrg_curve.json"
    curve_csv = geometry_dir / "dmrg_curve.csv"
    settings_path = geometry_dir / "settings.json"

    # ``--force`` resets only the qubit-frame products. The much more
    # expensive SU(2) reference checkpoints have their own force flag.
    if geometry_dir.exists() and args.force:
        for path in (
            curve_json,
            curve_csv,
            settings_path,
            geometry_dir / "frame.json",
            geometry_dir / "result.json",
        ):
            path.unlink(missing_ok=True)
        tensor_networks = geometry_dir / "tensor_networks"
        if tensor_networks.exists():
            shutil.rmtree(tensor_networks)

    if geometry in explicit_references:
        reference_energy = explicit_references[geometry]
        reference_source = {
            "source": "command_line",
            "validated": True,
        }
    elif args.dry_run:
        reference_energy = None
        reference_source = {
            "source": "planned_fermionic_su2_M300_M400",
            "validated": False,
            "reason": "dry run does not execute DMRG",
        }
    else:
        reference_energy, reference_validation = (
            prepare_fermionic_su2_reference(data, geometry_dir, args)
        )
        reference_source = {
            "source": str(
                (
                    geometry_dir
                    / "reference"
                    / "reference_validation.json"
                ).resolve()
            ),
            "validated": bool(reference_validation["validated"]),
            "saved_payload": reference_validation,
        }
        if reference_energy is None:
            raise RuntimeError(
                f"adaptive bond search requires a validated reference for "
                f"{geometry}; inspect the saved M=300/M=400 reference rows"
            )

    calculation_settings = _calculation_settings(args, data, reference_energy)

    existing_rows: list[dict] = []
    if settings_path.exists():
        saved_settings = load_json(settings_path)
        if saved_settings != calculation_settings:
            raise ValueError(
                f"saved result-defining settings differ for {geometry}; "
                "use --force or a new --output-dir"
            )
        if curve_json.exists():
            existing_rows = list(load_json(curve_json))

    completed = {int(row["bond_dim"]) for row in existing_rows}
    next_bond_dim = next_adaptive_bond_dimension(
        existing_rows, args.bond_dims
    )
    print("\n" + "=" * 80, flush=True)
    print(
        f"N2 {geometry}: R={data['bond_length_angstrom']:.1f} Angstrom, "
        f"{data['n_qubits']} qubits, {len(data['symmetries'])} Beam generators",
        flush=True,
    )
    print(
        f"CISD determinants={data['cisd_state'].nnz}; "
        f"Beam score={data['beam_record'].get('score')}; "
        f"completed bond dimensions={sorted(completed)}; "
        f"next adaptive bond dimension={next_bond_dim}",
        flush=True,
    )
    if reference_energy is None:
        if args.dry_run:
            print(
                "Dry run: the planned SU(2) M=300/M=400 reference pair was "
                "not executed.",
                flush=True,
            )
        else:
            print(
                "The SU(2) pair was not validated: chemical-accuracy fields "
                "will be unavailable; sweep convergence is still assessed.",
                flush=True,
            )

    transformed_hamiltonian, clifford, frame_metadata = prepare_frame(data)
    frame_definition = {
        "frame": FRAME,
        "geometry": geometry,
        "bond_length_angstrom": data["bond_length_angstrom"],
        "symmetries": data["beam_record"]["symmetries"],
        "beam_search_result": data["beam_record"],
        "score_metadata": data["score_metadata"],
        "source_paths": data["source_paths"],
        "state_fingerprint": data["state_fingerprint"],
        "hamiltonian_fingerprint": data["hamiltonian_fingerprint"],
        "transformation": frame_metadata,
    }

    if args.dry_run:
        return existing_rows, {
            "status": "dry_run_validated",
            **frame_definition,
            "reference_energy": reference_energy,
            "reference_provenance": reference_source,
            "bond_dimensions_requested": list(args.bond_dims),
        }

    geometry_dir.mkdir(parents=True, exist_ok=True)
    save_json(settings_path, calculation_settings)
    save_json(geometry_dir / "frame.json", frame_definition)

    def checkpoint(raw_row: dict) -> None:
        """Persist one completed bond dimension before the next run starts."""
        decorated = _decorate_row(raw_row, data, reference_source)
        existing_rows.append(decorated)
        existing_rows.sort(key=lambda item: int(item["bond_dim"]))
        save_json(curve_json, existing_rows)
        write_csv(curve_csv, existing_rows)

    backend_summary = None
    invocation_start = perf_counter()
    if next_bond_dim is not None:
        scratch = (
            None
            if args.scratch is None
            else args.scratch / f"N2_{geometry}" / FRAME
        )
        prior_rows = tuple(dict(row) for row in existing_rows)

        def select_next(invocation_rows) -> int | None:
            """Choose the next point using saved and current-invocation rows."""
            return next_adaptive_bond_dimension(
                prior_rows + tuple(invocation_rows), args.bond_dims
            )

        _, backend_summary = run_block2_qubit_dmrg_curve(
            label=FRAME,
            hamiltonian=transformed_hamiltonian,
            sparse_state=(data["cisd_state"].indices, data["cisd_state"].coeffs),
            exact_energy=reference_energy,
            warm_start_energy=data["cisd_energy"],
            n_qubits=data["n_qubits"],
            bond_dims=args.bond_dims,
            dmrg_sweeps=args.max_sweeps,
            dmrg_tolerance=args.dmrg_tolerance,
            sweep_tolerance=args.sweep_tolerance,
            mpo_cutoff=args.mpo_cutoff,
            mpo_builder=args.mpo_builder,
            sum_mpo_mod=args.sum_mpo_mod,
            initial_state="cisd",
            sparse_batch_size=args.sparse_batch_size,
            sparse_compression_cutoff=args.sparse_compression_cutoff,
            unitaries=(clifford,),
            transform_max_bond=max(args.bond_dims),
            transform_cutoff=args.transform_cutoff,
            validate_warm_start_energy=True,
            full_curve=True,
            n_threads=args.n_threads,
            n_mkl_threads=args.n_mkl_threads,
            stack_mem_gb=args.stack_mem_gb,
            davidson_threshold=args.davidson_threshold,
            warm_start_noises=args.warm_start_noises,
            verbose=args.verbose,
            scratch=scratch,
            artifact_dir=geometry_dir / "tensor_networks",
            bond_result_callback=checkpoint,
            bond_dim_selector=select_next,
        )
    invocation_seconds = perf_counter() - invocation_start

    if reference_energy is not None:
        first_accurate = next(
            (
                row
                for row in existing_rows
                if row.get("within_dmrg_tolerance") is True
            ),
            None,
        )
    else:
        first_accurate = None
    best_energy = min(
        (float(row["energy"]) for row in existing_rows), default=None
    )
    for row in existing_rows:
        row["energy_above_best_in_curve"] = (
            None if best_energy is None else float(row["energy"]) - best_energy
        )
    save_json(curve_json, existing_rows)
    write_csv(curve_csv, existing_rows)

    summary = {
        "status": "complete",
        **frame_definition,
        "reference_energy": reference_energy,
        "reference_provenance": reference_source,
        "chemical_accuracy_assessed": reference_energy is not None,
        "converged_within_grid": first_accurate is not None,
        "first_chemically_accurate_bond_dim": (
            None if first_accurate is None else int(first_accurate["bond_dim"])
        ),
        "bond_dimensions_requested": list(args.bond_dims),
        "bond_search": "adaptive_bracketed_binary_v1",
        "bond_dimensions_completed": [
            int(row["bond_dim"]) for row in existing_rows
        ],
        "best_energy_in_curve": best_energy,
        "bond_dims_without_sweep_convergence": [
            int(row["bond_dim"])
            for row in existing_rows
            if row.get("sweep_limit_reached_without_convergence") is True
        ],
        "total_dmrg_seconds_across_curve": sum(
            float(row["dmrg_seconds"]) for row in existing_rows
        ),
        "total_recorded_sweep_seconds_across_curve": sum(
            sum(float(value) for value in row.get("per_sweep_seconds", []))
            for row in existing_rows
        ),
        "latest_invocation_seconds": invocation_seconds,
        "latest_geometry_wall_seconds": perf_counter() - geometry_start,
        "latest_backend_summary": backend_summary,
        "execution_settings": {
            "n_threads": args.n_threads,
            "n_mkl_threads": args.n_mkl_threads,
            "stack_mem_gb": args.stack_mem_gb,
            "scratch": None if args.scratch is None else str(args.scratch),
            "verbose": args.verbose,
        },
    }
    save_json(geometry_dir / "result.json", summary)
    return existing_rows, summary


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser for this benchmark script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(GEOMETRIES),
        default=list(GEOMETRIES),
        help="N2 geometries to benchmark (default: all three).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-bond-dim",
        type=int,
        default=300,
        help="Default grid endpoint; above 100 the spacing is 20 (default: 300).",
    )
    parser.add_argument(
        "--bond-dims",
        type=int,
        nargs="+",
        default=None,
        help="Explicit increasing bond grid, overriding --max-bond-dim.",
    )
    parser.add_argument("--max-sweeps", type=int, default=100)
    parser.add_argument("--sweep-tolerance", type=float, default=1e-6)
    parser.add_argument("--dmrg-tolerance", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--davidson-threshold", type=float, default=1e-10)
    parser.add_argument(
        "--warm-start-noises",
        type=float,
        nargs="*",
        default=list(DEFAULT_NOISES),
    )
    parser.add_argument("--sparse-batch-size", type=int, default=32)
    parser.add_argument("--sparse-compression-cutoff", type=float, default=1e-13)
    parser.add_argument("--transform-cutoff", type=float, default=1e-13)
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument(
        "--mpo-builder",
        choices=("blocked_sum", "expression"),
        default="blocked_sum",
    )
    parser.add_argument("--sum-mpo-mod", type=int, default=10)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-mkl-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=32.0)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument(
        "--reference-energy",
        action="append",
        default=[],
        metavar="GEOMETRY=ENERGY",
        help="Validated external reference energy; may be repeated.",
    )
    parser.add_argument(
        "--reference-bond-dims",
        type=int,
        nargs=2,
        default=(300, 400),
        metavar=("M_LOW", "M_HIGH"),
        help="Two SU(2) reference bond dimensions (default: 300 400).",
    )
    parser.add_argument(
        "--reference-validation-tolerance",
        type=float,
        default=1e-4,
        help="Maximum energy difference for accepting the reference pair.",
    )
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--worker-mode",
        action="store_true",
        help=(
            "Run requested geometries and save only geometry-local files; "
            "defer combined-table writes to a separate aggregation process."
        ),
    )
    execution_mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate combined result files without running DMRG.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def aggregate_saved_results(output_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Load completed geometry checkpoints and rewrite combined result files.

    Parameters
    ----------
    output_dir
        Root containing ``N2_eqm``, ``N2_corr``, and ``N2_diss`` result
        directories produced by independent worker processes.

    Returns
    -------
    rows, summaries
        Concatenated per-bond rows and geometry-keyed result summaries. Only
        geometries with both a curve and completed ``result.json`` are
        included. Combined CSV/JSON files are written atomically where the
        persistence helper supports it.
    """
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for geometry in GEOMETRIES:
        geometry_dir = output_dir / f"N2_{geometry}"
        curve_path = geometry_dir / "dmrg_curve.json"
        result_path = geometry_dir / "result.json"
        if curve_path.exists() and result_path.exists():
            all_rows.extend(load_json(curve_path))
            summaries[geometry] = load_json(result_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    if all_rows:
        write_csv(output_dir / "dmrg_curves.csv", all_rows)
    compact_summaries = []
    for geometry, summary in summaries.items():
        compact_summaries.append(
            {
                "system": "N2_6-31G",
                "geometry": geometry,
                "bond_length_angstrom": summary["bond_length_angstrom"],
                "frame": FRAME,
                "symmetry_score": summary["beam_search_result"].get("score"),
                "reference_energy": summary["reference_energy"],
                "chemical_accuracy_assessed": summary[
                    "chemical_accuracy_assessed"
                ],
                "first_chemically_accurate_bond_dim": summary[
                    "first_chemically_accurate_bond_dim"
                ],
                "best_energy_in_curve": summary["best_energy_in_curve"],
                "bond_dims_without_sweep_convergence": summary[
                    "bond_dims_without_sweep_convergence"
                ],
                "total_dmrg_seconds_across_curve": summary[
                    "total_dmrg_seconds_across_curve"
                ],
                "total_recorded_sweep_seconds_across_curve": summary[
                    "total_recorded_sweep_seconds_across_curve"
                ],
            }
        )
    if compact_summaries:
        write_csv(output_dir / "dmrg_summary.csv", compact_summaries)
    save_json(
        output_dir / "benchmark.json",
        {
            "format": "n2_631g_beam_dmrg_v2",
            "frame": FRAME,
            "systems": list(summaries),
            "incomplete_systems": [
                geometry for geometry in GEOMETRIES if geometry not in summaries
            ],
            "summaries": summaries,
        },
    )
    return all_rows, summaries


def main() -> None:
    """Validate arguments, run requested geometries, and save aggregate data."""
    args = build_parser().parse_args()
    if args.max_sweeps < 1 or args.max_sweeps > 100:
        raise ValueError("--max-sweeps must lie between 1 and the cap of 100")
    if args.bond_dims is None:
        args.bond_dims = default_bond_dimensions(args.max_bond_dim)
    else:
        args.bond_dims = tuple(int(value) for value in args.bond_dims)
        if any(value < 1 for value in args.bond_dims):
            raise ValueError("bond dimensions must be positive")
        if tuple(sorted(set(args.bond_dims))) != args.bond_dims:
            raise ValueError("--bond-dims must be strictly increasing and unique")
    if args.n_threads < 1 or args.n_mkl_threads < 1:
        raise ValueError("thread counts must be positive")
    args.reference_bond_dims = tuple(
        int(value) for value in args.reference_bond_dims
    )
    if (
        any(value < 1 for value in args.reference_bond_dims)
        or args.reference_bond_dims[0] >= args.reference_bond_dims[1]
    ):
        raise ValueError(
            "--reference-bond-dims must be two increasing positive values"
        )
    if args.reference_validation_tolerance < 0:
        raise ValueError("reference validation tolerance must be nonnegative")
    explicit_references = parse_reference_energies(args.reference_energy)

    if args.aggregate_only:
        rows, summaries = aggregate_saved_results(args.output_dir)
        print(
            f"Aggregated {len(rows)} bond results from "
            f"{len(summaries)} completed geometries in "
            f"{args.output_dir.resolve()}."
        )
        return

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for geometry in args.systems:
        rows, summary = run_geometry(geometry, args, explicit_references)
        all_rows.extend(rows)
        summaries[geometry] = summary

    if args.dry_run:
        print("\nDry run succeeded; no DMRG calculations or files were written.")
        for geometry, summary in summaries.items():
            print(
                f"  {geometry}: score={summary['beam_search_result'].get('score')}, "
                f"terms={summary['transformation']['hamiltonian_terms_after']}, "
                "warm-start route=sparse_state_then_mps"
            )
        return

    if args.worker_mode:
        print(
            "\nWorker completed its geometry-local checkpoints; combined "
            "tables were intentionally deferred."
        )
        return

    rows, completed = aggregate_saved_results(args.output_dir)
    print(
        f"\nSaved {len(rows)} aggregate bond results for "
        f"{len(completed)} geometries to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
