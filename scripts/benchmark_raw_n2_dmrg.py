"""Benchmark DMRG for canonical N2 orbitals without a Clifford transform.

This is the raw-basis control for ``benchmark_saved_oo_n2.py``.  It uses the
same N2 1.1000 Angstrom checkpoint, full-CISD MPS initialization, bond-
dimension grid, DMRG settings, and chemical-accuracy stopping rule, but it
applies neither the saved orbital rotation nor any Clifford transformation.
It runs both the raw Jordan--Wigner qubit Hamiltonian in Pauli-mode pyblock2
and the spatial-integral Hamiltonian directly with SU(2)-adapted pyblock2.

Run from ``QuasiSymmetries``:

    python scripts/benchmark_raw_n2_dmrg.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from openfermion import MolecularData

import benchmark_saved_oo_n2 as shared


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OO_PROJECT_ROOT = WORKSPACE_ROOT / "quasisymmetry"
DEFAULT_MOLPATH = (
    OO_PROJECT_ROOT / "hamiltonians" / "N2" / "N2_1.1000_D2h.chk"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "saved" / "results" / "N2_1.1000_raw_dmrg"
)

if str(OO_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(OO_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molpath", type=Path, default=DEFAULT_MOLPATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bond-dims",
        type=int,
        nargs="+",
        default=(
            list(range(1, 11))
            + list(range(12, 21, 2))
            + list(range(30, 101, 10))
        ),
        help="maximum MPS bond dimensions to benchmark",
    )
    parser.add_argument("--dmrg-sweeps", type=int, default=100)
    parser.add_argument("--dmrg-tol", type=float, default=1.6e-3)
    parser.add_argument("--sweep-tol", type=float, default=1e-6)
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument(
        "--block2-mpo-builder",
        choices=("blocked_sum", "expression"),
        default="blocked_sum",
    )
    parser.add_argument("--sum-mpo-mod", type=int, default=20)
    parser.add_argument("--mps-cutoff", type=float, default=1e-13)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--qubit-backend",
        choices=("block2",),
        default="block2",
        help="retained for CLI compatibility; pyblock2 is the only backend",
    )
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--n-mkl-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=0.5)
    parser.add_argument("--davidson-threshold", type=float, default=1e-10)
    parser.add_argument(
        "--skip-qubit-dmrg",
        action="store_true",
        help="run only the direct fermionic Block2 benchmark",
    )
    parser.add_argument(
        "--skip-fermionic-dmrg",
        action="store_true",
        help="run only the raw Jordan-Wigner pyblock2 benchmark",
    )
    parser.add_argument(
        "--full-curve",
        action="store_true",
        help=(
            "continue through every requested bond dimension after reaching "
            "chemical accuracy; by default stop at the first converged bond "
            "dimension"
        ),
    )
    parser.add_argument(
        "--initial-state",
        choices=("cisd", "exact_fci", "random"),
        default="cisd",
        help=(
            "warm start from the full CISD state (default), exact FCI state, "
            "or a random MPS"
        ),
    )
    parser.add_argument("--sparse-batch-size", type=int, default=32)
    parser.add_argument(
        "--sparse-compression-cutoff", type=float, default=1e-13
    )
    parser.add_argument(
        "--no-save-tensor-networks",
        dest="save_tensor_networks",
        action="store_false",
        help="do not save the constructed warm-start MPS and Hamiltonian MPO",
    )
    parser.set_defaults(save_tensor_networks=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def openfermion_molecule_from_ffsim(moldata) -> MolecularData:
    """Build the integral container expected by the fermionic Block2 helper."""
    molecule = MolecularData(
        geometry="FCIDUMP",
        basis="unknown",
        multiplicity=abs(int(moldata.nelec[0]) - int(moldata.nelec[1])) + 1,
        charge=0,
    )
    molecule.n_orbitals = int(moldata.norb)
    molecule.n_qubits = 2 * int(moldata.norb)
    molecule.n_electrons = int(sum(moldata.nelec))
    molecule.nuclear_repulsion = float(
        np.real(moldata.hamiltonian.constant)
    )
    molecule.one_body_integrals = np.asarray(
        moldata.hamiltonian.one_body_tensor, dtype=float
    )
    molecule.two_body_integrals = np.transpose(
        np.asarray(moldata.hamiltonian.two_body_tensor, dtype=float),
        (0, 2, 3, 1),
    )
    return molecule


def run_fermionic_dmrg_curve(
    *,
    molecule: MolecularData,
    warm_start_state,
    warm_start_energy: float,
    fci_energy: float | None,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """Run SU(2)-adapted quantum-chemistry DMRG directly on the integrals.

    Parameters
    ----------
    molecule
        OpenFermion molecular integral container.
    warm_start_state, warm_start_energy
        Selected-CI state and its variational energy, used to construct and
        validate the initial SU(2) MPS.
    fci_energy
        Optional external reference energy. When absent, sweep convergence is
        still reported, while absolute-error and chemical-accuracy fields are
        returned as ``None``.
    args
        DMRG bond grid, convergence, noise, execution, and output settings.

    Returns
    -------
    rows, summary
        Per-bond measurements and curve-level MPO, warm-start, timing, and
        convergence metadata.
    """
    try:
        from pyblock2.driver.core import SymmetryTypes
        from quasisymmetries.mpo import (
            build_qc_mpo_from_openfermion_molecule,
            cleanup_qc_mpo_result,
        )
        from test_mpo_ferm import (
            determinants_to_su2_pyblock_mps,
            pyblock_su2_to_block2_mps,
            statevector_to_sz_determinants,
        )
    except ImportError as exc:
        raise ImportError(
            "The direct fermionic benchmark requires pyblock2."
        ) from exc

    n_sites = int(molecule.n_orbitals)
    n_electrons = int(molecule.n_electrons)
    spin = int(molecule.multiplicity) - 1
    active_orbitals = list(range(n_sites))
    orb_sym = [1] * n_sites

    determinants, coefficients, projection_norm = (
        statevector_to_sz_determinants(
            warm_start_state,
            n_spatial_orbitals=n_sites,
            active_orbitals=active_orbitals,
            n_electrons=n_electrons,
            spin=spin,
            cutoff=0.0,
        )
    )
    py_su2_warm_mps = determinants_to_su2_pyblock_mps(
        determinants,
        coefficients,
        n_sites=n_sites,
        n_electrons=n_electrons,
        spin=spin,
        orb_sym=orb_sym,
        stack_mem=int(args.stack_mem_gb * 1024**3),
    )

    mpo_start = perf_counter()
    mpo_result = build_qc_mpo_from_openfermion_molecule(
        molecule,
        ncore=0,
        active_orbitals=active_orbitals,
        symm_type=SymmetryTypes.SU2,
        n_threads=args.n_threads,
        n_mkl_threads=getattr(args, "n_mkl_threads", 1),
        stack_mem=int(args.stack_mem_gb * 1024**3),
        iprint=2 if args.verbose else 0,
        orb_sym=orb_sym,
    )
    mpo_seconds = perf_counter() - mpo_start
    driver = mpo_result["driver"]
    mpo = mpo_result["mpo"]

    try:
        from quasisymmetries.block2_qubit_benchmark import (
            Block2SweepTimer,
            block2_dmrg_sweep_status,
            save_block2_mpo,
            save_block2_mps,
        )

        warm_mps = pyblock_su2_to_block2_mps(
            py_su2_warm_mps,
            driver,
            tag="WARM-SU2-N2-RAW",
        )
        identity_mpo = driver.get_identity_mpo()
        exact_norm = driver.expectation(warm_mps, identity_mpo, warm_mps)
        imported_warm_energy = float(
            driver.expectation(warm_mps, mpo, warm_mps) / exact_norm
        )
        if abs(imported_warm_energy - warm_start_energy) > 1e-8:
            raise RuntimeError(
                "fermionic warm-start MPS energy check failed: "
                f"{imported_warm_energy} versus {warm_start_energy}"
            )
        artifacts = {}
        if args.save_tensor_networks:
            artifact_dir = args.output_dir / "tensor_networks"
            artifacts["mpo"] = save_block2_mpo(
                mpo,
                artifact_dir
                / "raw_fermionic_su2_hamiltonian_mpo.block2.bin",
            )
            artifacts["warm_start_mps"] = save_block2_mps(
                warm_mps,
                artifact_dir
                / (
                    "raw_fermionic_su2_"
                    f"{args.initial_state}_warm_start_mps.block2"
                ),
            )
            print(
                "raw_fermionic_su2: saved Hamiltonian MPO to "
                f"{artifacts['mpo']['path']}",
                flush=True,
            )
            print(
                "raw_fermionic_su2: saved "
                f"{args.initial_state} warm-start MPS to "
                f"{artifacts['warm_start_mps']['path']}",
                flush=True,
            )

        rows = []
        first_converged_bd = None
        thresholds = [
            float(getattr(args, "davidson_threshold", 1e-10))
        ] * args.dmrg_sweeps
        for bond_dim in args.bond_dims:
            tag = f"N2-RAW-FERM-BD{bond_dim}"
            if args.initial_state != "random":
                ket = driver.copy_mps(warm_mps, tag=tag)
                warm_start_noises = tuple(
                    float(value)
                    for value in getattr(args, "warm_start_noises", ())
                )
                noises = list(
                    warm_start_noises[: args.dmrg_sweeps]
                )
                noises += [0.0] * (
                    args.dmrg_sweeps - len(noises)
                )
            else:
                ket = driver.get_random_mps(
                    tag=tag, bond_dim=int(bond_dim), nroots=1
                )
                conventional_noise = [
                    1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6
                ]
                noises = (
                    conventional_noise[: args.dmrg_sweeps]
                    + [0.0]
                    * max(0, args.dmrg_sweeps - len(conventional_noise))
                )

            start = perf_counter()
            sweep_timer = Block2SweepTimer()
            driver.set_callback(sweep_timer)
            energy = float(
                driver.dmrg(
                    mpo,
                    ket,
                    n_sweeps=args.dmrg_sweeps,
                    tol=args.sweep_tol,
                    bond_dims=[int(bond_dim)] * args.dmrg_sweeps,
                    noises=noises,
                    thrds=thresholds,
                    dav_max_iter=50,
                    iprint=1 if args.verbose else 0,
                )
            )
            seconds = perf_counter() - start
            sweep_status = block2_dmrg_sweep_status(
                driver,
                requested_sweeps=args.dmrg_sweeps,
                energy_tolerance=args.sweep_tol,
                noises=noises,
                sweep_seconds=sweep_timer.sweep_seconds,
            )
            error = (
                None if fci_energy is None else abs(energy - fci_energy)
            )
            converged = (
                None if error is None else error <= args.dmrg_tol
            )
            if converged is True and first_converged_bd is None:
                first_converged_bd = int(bond_dim)
                if args.save_tensor_networks:
                    artifacts["first_chemically_accurate_mps"] = (
                        save_block2_mps(
                            ket,
                            artifact_dir
                            / (
                                "raw_fermionic_su2_first_chemical_accuracy_"
                                f"bd{int(bond_dim)}_mps.block2"
                            ),
                        )
                    )
            row = {
                "frame": "raw_fermionic_su2",
                "bond_dim": int(bond_dim),
                "energy": energy,
                "abs_energy_error": error,
                "within_dmrg_tolerance": converged,
                "dmrg_seconds": seconds,
                "max_result_mps_bond": int(bond_dim),
                **sweep_status,
            }
            rows.append(row)
            print(
                f"{'raw_fermionic_su2':18s} bond_dim={bond_dim:3d} "
                f"E={energy:.12f} |dE|="
                f"{'unavailable' if error is None else f'{error:.3e}'} "
                f"seconds={seconds:.1f}",
                flush=True,
            )
            if sweep_status["sweep_limit_reached_without_convergence"]:
                delta = sweep_status["last_sweep_energy_change"]
                delta_text = "unavailable" if delta is None else f"{delta:.3e}"
                print(
                    "WARNING: raw_fermionic_su2 "
                    f"bond_dim={bond_dim} exhausted {args.dmrg_sweeps} "
                    "sweeps without sweep-energy convergence; "
                    f"last |delta E_sweep|={delta_text}, "
                    f"tolerance={args.sweep_tol:.3e}.",
                    flush=True,
                )
            if converged is True and not args.full_curve:
                print(
                    "raw_fermionic_su2: reached chemical accuracy at "
                    f"bond_dim={bond_dim}; stopping this frame.",
                    flush=True,
                )
                break

        first_converged_row = next(
            (
                row
                for row in rows
                if row["within_dmrg_tolerance"] is True
            ),
            None,
        )
        summary = {
            "frame": "raw_fermionic_su2",
            "chemical_accuracy_assessed": fci_energy is not None,
            "reference_energy": fci_energy,
            "first_converged_bond_dim": first_converged_bd,
            "converged_within_grid": (
                None if fci_energy is None else first_converged_bd is not None
            ),
            "first_converged_dmrg_optimization_seconds": (
                None
                if first_converged_row is None
                else first_converged_row["dmrg_seconds"]
            ),
            "first_converged_per_sweep_seconds": (
                None
                if first_converged_row is None
                else first_converged_row["per_sweep_seconds"]
            ),
            "first_converged_sweep_energies": (
                None
                if first_converged_row is None
                else first_converged_row["sweep_energies"]
            ),
            "mpo_bond_dimension": mpo_result["largest_mpo_bond_dim"],
            "mpo_build_seconds": mpo_seconds,
            "warm_start_kind": args.initial_state,
            "warm_start_determinants_retained": len(determinants),
            "warm_start_projection_norm": projection_norm,
            "imported_warm_start_mps_energy": imported_warm_energy,
            "expected_warm_start_energy": warm_start_energy,
            "saved_tensor_networks": artifacts,
            "symmetry_type": "SU2",
            "ncore": 0,
            "active_electrons": n_electrons,
            "active_orbitals": n_sites,
            "sweep_energy_tolerance": args.sweep_tol,
            "warm_start_noises": list(
                getattr(args, "warm_start_noises", ())
            ),
            "bond_dims_without_sweep_convergence": [
                row["bond_dim"]
                for row in rows
                if row["sweep_limit_reached_without_convergence"]
            ],
        }
        return rows, summary
    finally:
        driver.finalize()
        cleanup_qc_mpo_result(mpo_result)


def main() -> None:
    args = parse_args()
    if args.skip_qubit_dmrg and args.skip_fermionic_dmrg:
        raise ValueError(
            "--skip-qubit-dmrg and --skip-fermionic-dmrg cannot both be set"
        )
    molpath = args.molpath.expanduser().resolve()
    if not molpath.exists():
        raise FileNotFoundError(molpath)

    try:
        from chemistry import load_moldata
    except ImportError as exc:
        raise ImportError(
            "This workflow requires ffsim, pyscf, openfermion, and "
            "openfermionpyscf from the sibling quasisymmetry project."
        ) from exc

    moldata = load_moldata(str(molpath))
    norb = int(moldata.norb)
    n_qubits = 2 * norb
    nelec = tuple(int(value) for value in moldata.nelec)

    print(f"Hamiltonian: {molpath}", flush=True)
    print(
        f"Raw benchmark: norb={norb}, n_qubits={n_qubits}, nelec={nelec}",
        flush=True,
    )
    print("Orbital rotation: none", flush=True)
    print("Clifford transformation: none", flush=True)

    fci_energy, fci_ci = shared.solve_fci(molpath)
    raw_fci_state = shared.expand_ci_state(fci_ci, norb, nelec)
    if not np.isclose(np.linalg.norm(raw_fci_state), 1.0, atol=1e-10):
        raise RuntimeError("expanded raw FCI state is not normalized")
    cisd_energy = None
    if args.initial_state == "cisd":
        cisd_energy, cisd_ci = shared.solve_cisd(molpath)
        warm_start_state = shared.expand_ci_state(
            cisd_ci, norb, nelec, cutoff=0.0
        )
    elif args.initial_state == "exact_fci":
        warm_start_state = raw_fci_state
    else:
        warm_start_state = None
    warm_start_energy = (
        cisd_energy if args.initial_state == "cisd" else fci_energy
    )
    raw_sparse_warm_state = (
        shared.sparse_state(warm_start_state)
        if warm_start_state is not None
        else None
    )
    if raw_sparse_warm_state is not None:
        print(
            "Warm-start determinants: "
            f"{len(raw_sparse_warm_state[0])}",
            flush=True,
        )

    raw_jw_hamiltonian = shared.molecular_hamiltonian_to_jw(
        moldata.hamiltonian, nelec
    )
    fermionic_molecule = openfermion_molecule_from_ffsim(moldata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "benchmark.json"
    curve_path = args.output_dir / "dmrg_curve.csv"
    result = {
        "input": {"molpath": str(molpath)},
        "system": {
            "norb": norb,
            "n_qubits": n_qubits,
            "nelec": nelec,
            "fci_energy": fci_energy,
            "cisd_energy": cisd_energy,
        },
        "basis": {
            "orbital_rotation": False,
            "clifford_transformation": False,
            "jordan_wigner_terms": len(raw_jw_hamiltonian.terms),
            "fermionic_active_space": f"CAS({sum(nelec)}e,{norb}o)",
            "fermionic_symmetry": "SU2",
            "frozen_core_orbitals": 0,
            "bond_dimension_note": (
                "SU2-adapted fermionic and two-level qubit MPS bond dimensions "
                "use different local bases/symmetry structure and should be "
                "reported side by side, not interpreted as identical units."
            ),
        },
        "dmrg_settings": {
            "bond_dims": args.bond_dims,
            "dmrg_sweeps": args.dmrg_sweeps,
            "dmrg_tolerance": args.dmrg_tol,
            "sweep_tolerance": args.sweep_tol,
            "mpo_cutoff": args.mpo_cutoff,
            "block2_mpo_builder": args.block2_mpo_builder,
            "sum_mpo_mod": args.sum_mpo_mod,
            "mps_cutoff": args.mps_cutoff,
            "initial_state": args.initial_state,
            "noise": args.noise,
            "seed": args.seed,
            "full_curve": args.full_curve,
            "n_threads": args.n_threads,
            "n_mkl_threads": getattr(args, "n_mkl_threads", 1),
            "stack_mem_gb": args.stack_mem_gb,
            "qubit_backend": args.qubit_backend,
            "davidson_threshold": args.davidson_threshold,
            "sparse_batch_size": args.sparse_batch_size,
            "sparse_compression_cutoff": args.sparse_compression_cutoff,
            "save_tensor_networks": args.save_tensor_networks,
            "skip_qubit_dmrg": args.skip_qubit_dmrg,
            "skip_fermionic_dmrg": args.skip_fermionic_dmrg,
        },
        "warm_start": {
            "kind": args.initial_state,
            "energy": warm_start_energy,
            "nonzero_determinants": (
                len(raw_sparse_warm_state[0])
                if raw_sparse_warm_state is not None
                else None
            ),
            "determinant_coefficient_truncation": False,
            "mps_numerical_rank_cutoff": (
                args.sparse_compression_cutoff
                if args.qubit_backend == "block2"
                else args.mps_cutoff
            ),
        },
        "dmrg_summary": {},
        "dmrg_curve": [],
    }
    shared.save_json(result_path, result)

    all_rows = []
    if not args.skip_qubit_dmrg:
        rows, summary = shared.run_qubit_dmrg_curve(
            label="raw_canonical_qubit",
            hamiltonian=raw_jw_hamiltonian,
            exact_state=warm_start_state,
            sparse_warm_state=raw_sparse_warm_state,
            exact_energy=fci_energy,
            warm_start_energy=warm_start_energy,
            n_qubits=n_qubits,
            args=args,
        )
        all_rows.extend(rows)
        result["dmrg_curve"] = all_rows
        result["dmrg_summary"]["raw_canonical_qubit"] = summary
        shared.write_csv(curve_path, all_rows)
        shared.save_json(result_path, result)

    if not args.skip_fermionic_dmrg:
        rows, summary = run_fermionic_dmrg_curve(
            molecule=fermionic_molecule,
            warm_start_state=(
                warm_start_state
                if warm_start_state is not None
                else raw_fci_state
            ),
            warm_start_energy=warm_start_energy,
            fci_energy=fci_energy,
            args=args,
        )
        all_rows.extend(rows)
        result["dmrg_curve"] = all_rows
        result["dmrg_summary"]["raw_fermionic_su2"] = summary
        shared.write_csv(curve_path, all_rows)
        shared.save_json(result_path, result)

    for label, summary in result["dmrg_summary"].items():
        print(
            f"{label} first converged bond dimension: "
            f"{summary['first_converged_bond_dim']}",
            flush=True,
        )
    print(f"Wrote results to {result_path}", flush=True)
    print(f"Wrote DMRG curve to {curve_path}", flush=True)


if __name__ == "__main__":
    main()
