#!/usr/bin/env python3
"""Generate full-space N2/6-31G data and probe its raw qubit MPO.

This is intentionally a preprocessing and MPO-construction probe. It does
not run FCI, CISD, symmetry search, or DMRG. Molecular data and the packed
Jordan--Wigner Pauli stream are saved before pyblock2 MPO construction begins,
so they remain reusable if native MPO construction runs out of memory.
"""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

import _bootstrap

from openfermion import MolecularData
from openfermion.transforms import get_fermion_operator
from openfermionpyscf import run_pyscf

from quasisymmetries.block2_qubit_benchmark import (
    build_block2_qubit_mpo,
    save_block2_mpo,
)
from quasisymmetries.bs.utils import jordan_wigner_pauli_stream
from quasisymmetries.save import (
    load_pauli_term_stream,
    save_json,
    save_pauli_term_stream,
)


def n2_geometry(bond_length: float) -> list[tuple[str, tuple[float, float, float]]]:
    """Return a centered linear N2 geometry.

    Parameters
    ----------
    bond_length
        Nitrogen--nitrogen distance in Angstrom.

    Returns
    -------
    geometry
        Two OpenFermion atom records centered on the Cartesian origin.
    """
    half = float(bond_length) / 2.0
    return [("N", (0.0, 0.0, -half)), ("N", (0.0, 0.0, half))]


def current_rss_gib() -> float | None:
    """Return current resident memory in GiB.

    Returns
    -------
    rss
        Current resident memory in GiB, or ``None`` without usable ``psutil``.
    """
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024**3)
    except Exception:
        return None


def peak_rss_gib() -> float:
    """Return this process's peak resident memory in GiB.

    Returns
    -------
    peak
        Maximum resident set size reported for this process.

    ``ru_maxrss`` is reported in bytes on macOS and KiB on Linux.
    """
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return peak / divisor


def rss_text() -> str:
    """Return a short current/peak resident-memory status string.

    Returns
    -------
    message
        Human-readable current and peak resident-memory values.
    """
    current = current_rss_gib()
    current_text = "unavailable" if current is None else f"{current:.3f} GiB"
    return f"RSS={current_text}, peak_RSS={peak_rss_gib():.3f} GiB"


def molecular_basename(output_dir: Path, bond_length: float, basis: str) -> Path:
    """Return the extension-free OpenFermion MolecularData filename.

    Parameters
    ----------
    output_dir
        Root directory for this probe's artifacts.
    bond_length
        Nitrogen--nitrogen distance in Angstrom.
    basis
        Gaussian basis-set name.

    Returns
    -------
    path
        Extension-free path used by ``MolecularData``; OpenFermion writes an
        HDF5 file by appending ``.hdf5``.
    """
    safe_basis = basis.lower().replace("+", "p").replace("*", "s")
    return output_dir / "molecule" / f"N2_{bond_length:.4f}_{safe_basis}"


def generate_or_load_molecule(args, output_dir: Path):
    """Generate or reload the SCF-processed N2 molecule.

    Parameters
    ----------
    args
        Parsed command-line namespace containing geometry, basis, charge,
        multiplicity, and ``force_scf`` settings.
    output_dir
        Root artifact directory.

    Returns
    -------
    molecule, seconds, generated
        Loaded ``MolecularData``, elapsed wall time, and whether PySCF was run
        during this invocation.
    """
    basename = molecular_basename(output_dir, args.bond_length, args.basis)
    basename.parent.mkdir(parents=True, exist_ok=True)
    hdf5_path = Path(f"{basename}.hdf5")
    start = perf_counter()
    if hdf5_path.exists() and not args.force_scf:
        molecule = MolecularData(filename=str(basename))
        molecule.load()
        generated = False
    else:
        molecule = MolecularData(
            geometry=n2_geometry(args.bond_length),
            basis=args.basis,
            multiplicity=args.multiplicity,
            charge=args.charge,
            filename=str(basename),
            description=f"R{args.bond_length:.4f}",
        )
        molecule = run_pyscf(
            molecule,
            run_scf=True,
            run_mp2=False,
            run_cisd=False,
            run_ccsd=False,
            run_fci=False,
        )
        molecule.filename = str(basename)
        molecule.save()
        generated = True
    return molecule, perf_counter() - start, generated


def active_space_definition(molecule, frozen_core_orbitals: int) -> dict:
    """Construct the requested leading-core frozen-orbital active space.

    Parameters
    ----------
    molecule
        SCF-processed OpenFermion ``MolecularData`` object.
    frozen_core_orbitals
        Number of lowest-energy doubly occupied spatial orbitals to freeze.

    Returns
    -------
    definition
        Dictionary containing frozen and active orbital indices, active
        electron count, active spatial-orbital count, and qubit count.
    """
    ncore = int(frozen_core_orbitals)
    if ncore < 0 or 2 * ncore > int(molecule.n_electrons):
        raise ValueError("invalid number of frozen core orbitals")
    if ncore >= int(molecule.n_orbitals):
        raise ValueError("freezing all spatial orbitals is not allowed")
    frozen = list(range(ncore))
    active = list(range(ncore, int(molecule.n_orbitals)))
    return {
        "frozen_core_orbitals": frozen,
        "active_orbitals": active,
        "active_electrons": int(molecule.n_electrons) - 2 * ncore,
        "active_spatial_orbitals": len(active),
        "n_qubits": 2 * len(active),
    }


def build_fermion_hamiltonian(molecule, active_space: dict):
    """Construct the active-space OpenFermion fermionic Hamiltonian.

    Parameters
    ----------
    molecule
        SCF-processed OpenFermion ``MolecularData`` object.
    active_space
        Dictionary returned by :func:`active_space_definition`.

    Returns
    -------
    hamiltonian
        Normal-ordered OpenFermion ``FermionOperator`` including the nuclear
        and frozen-core scalar contributions.
    """
    frozen = active_space["frozen_core_orbitals"]
    active = active_space["active_orbitals"]
    if frozen:
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=frozen,
            active_indices=active,
        )
    else:
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    return get_fermion_operator(molecular_hamiltonian)


def pauli_stream_statistics(stream) -> dict:
    """Summarize the size and coefficient distribution of a Pauli stream.

    Parameters
    ----------
    stream
        Packed ``PauliTermStream`` to summarize.

    Returns
    -------
    statistics
        Counts, coefficient norms, maximum imaginary component, and Pauli-word
        weights. Identity terms are included in ``term_count``.
    """
    coefficients = np.asarray(
        [item.signed_coefficient for item in stream.terms], dtype=complex
    )
    weights = [
        int((item.mask[0] | item.mask[1]).bit_count())
        for item in stream.terms
    ]
    identity = next(
        (
            item.signed_coefficient
            for item in stream.terms
            if item.mask == (0, 0)
        ),
        0.0j,
    )
    return {
        "term_count": len(stream.terms),
        "nonidentity_term_count": sum(
            item.mask != (0, 0) for item in stream.terms
        ),
        "identity_coefficient": identity,
        "pauli_l1_norm": float(np.sum(np.abs(coefficients))),
        "largest_coefficient": float(
            np.max(np.abs(coefficients)) if len(coefficients) else 0.0
        ),
        "largest_imaginary_coefficient": float(
            np.max(np.abs(coefficients.imag)) if len(coefficients) else 0.0
        ),
        "maximum_pauli_weight": max(weights, default=0),
        "mean_pauli_weight": float(np.mean(weights) if weights else 0.0),
    }


def mpo_bond_dimensions(mpo) -> list[int] | None:
    """Return all reported pyblock2 MPO bond dimensions when available.

    Parameters
    ----------
    mpo
        Constructed pyblock2 MPO or MPO wrapper.

    Returns
    -------
    dimensions
        Integer bond dimensions from ``get_bond_dims()``, or ``None`` when
        this pyblock2 version does not expose them.
    """
    try:
        return [int(value) for value in mpo.get_bond_dims()]
    except Exception:
        pass
    try:
        return [int(value) for value in mpo.prim_mpo.get_bond_dims()]
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """Parse command-line settings.

    Returns
    -------
    args
        Populated probe ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.2)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--freeze-core-orbitals",
        type=int,
        default=0,
        help="Freeze this many lowest doubly occupied spatial orbitals.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _bootstrap.PROJECT_ROOT
            / "saved"
            / "results"
            / "n2_631g_mpo_probe"
        ),
    )
    parser.add_argument("--force-scf", action="store_true")
    parser.add_argument("--force-pauli", action="store_true")
    parser.add_argument(
        "--hamiltonian-only",
        action="store_true",
        help="Generate and save the Pauli stream but do not build its MPO.",
    )
    parser.add_argument(
        "--mpo-builder",
        choices=("blocked_sum", "expression"),
        default="blocked_sum",
    )
    parser.add_argument("--pauli-tolerance", type=float, default=1e-12)
    parser.add_argument("--mpo-cutoff", type=float, default=1e-10)
    parser.add_argument("--sum-mpo-mod", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-mkl-threads", type=int, default=1)
    parser.add_argument("--stack-mem-gb", type=float, default=0.25)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--iprint", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Generate/reload the Hamiltonian, construct its raw MPO, and save data.

    Returns
    -------
    None
        Results are printed and written below ``--output-dir``.
    """
    args = parse_args()
    if args.threads < 1 or args.n_mkl_threads < 1:
        raise ValueError("threads and n_mkl_threads must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    active_tag = f"fc{args.freeze_core_orbitals}"
    manifest_path = output_dir / f"probe_{active_tag}.json"
    pauli_path = output_dir / f"raw_jordan_wigner_{active_tag}_pauli_stream.json"
    mpo_path = output_dir / f"raw_qubit_{active_tag}_hamiltonian_mpo.block2.bin"

    print("N2/6-31G raw qubit-MPO feasibility probe", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Initial {rss_text()}", flush=True)

    molecule, scf_seconds, generated = generate_or_load_molecule(
        args, output_dir
    )
    active_space = active_space_definition(
        molecule, args.freeze_core_orbitals
    )
    print(
        "Molecule: "
        f"n_orbitals={molecule.n_orbitals}, "
        f"n_electrons={molecule.n_electrons}, "
        f"active_orbitals={active_space['active_spatial_orbitals']}, "
        f"n_qubits={active_space['n_qubits']}",
        flush=True,
    )
    print(f"SCF/reload seconds={scf_seconds:.3f}; {rss_text()}", flush=True)

    fermion_seconds = 0.0
    jw_seconds = 0.0
    fermion_term_count = None
    if pauli_path.exists() and not args.force_pauli:
        start = perf_counter()
        stream = load_pauli_term_stream(
            pauli_path, n_qubits=active_space["n_qubits"]
        )
        pauli_load_seconds = perf_counter() - start
        print(
            f"Loaded saved Pauli stream in {pauli_load_seconds:.3f} s; "
            f"{rss_text()}",
            flush=True,
        )
    else:
        start = perf_counter()
        fermion_hamiltonian = build_fermion_hamiltonian(
            molecule, active_space
        )
        fermion_seconds = perf_counter() - start
        fermion_term_count = len(fermion_hamiltonian.terms)
        print(
            f"Fermion Hamiltonian: {fermion_term_count} terms in "
            f"{fermion_seconds:.3f} s; {rss_text()}",
            flush=True,
        )
        start = perf_counter()
        stream = jordan_wigner_pauli_stream(
            fermion_hamiltonian,
            n_qubits=active_space["n_qubits"],
            tolerance=args.pauli_tolerance,
        )
        jw_seconds = perf_counter() - start
        save_pauli_term_stream(pauli_path, stream)
        pauli_load_seconds = None
        print(
            f"Jordan--Wigner stream: {len(stream.terms)} combined terms in "
            f"{jw_seconds:.3f} s; {rss_text()}",
            flush=True,
        )
        print(f"Saved Pauli stream: {pauli_path}", flush=True)

    statistics = pauli_stream_statistics(stream)
    manifest = {
        "status": "hamiltonian_ready",
        "system": "N2",
        "geometry_angstrom": n2_geometry(args.bond_length),
        "bond_length_angstrom": args.bond_length,
        "basis": args.basis,
        "multiplicity": args.multiplicity,
        "charge": args.charge,
        "molecular_data": str(
            Path(
                f"{molecular_basename(output_dir, args.bond_length, args.basis)}.hdf5"
            )
        ),
        "hartree_fock_energy": float(molecule.hf_energy),
        "full_spatial_orbitals": int(molecule.n_orbitals),
        "full_electrons": int(molecule.n_electrons),
        "active_space": active_space,
        "fermion_term_count": fermion_term_count,
        "pauli_stream": str(pauli_path),
        "pauli_statistics": statistics,
        "settings": {
            "pauli_tolerance": args.pauli_tolerance,
            "mpo_builder": args.mpo_builder,
            "mpo_cutoff": args.mpo_cutoff,
            "sum_mpo_mod": args.sum_mpo_mod,
            "threads": args.threads,
            "n_mkl_threads": args.n_mkl_threads,
            "stack_mem_gb": args.stack_mem_gb,
        },
        "timings_seconds": {
            "scf_or_molecule_reload": scf_seconds,
            "fermion_hamiltonian": fermion_seconds,
            "jordan_wigner": jw_seconds,
            "pauli_stream_load": pauli_load_seconds,
        },
        "memory_gib": {
            "current_after_hamiltonian": current_rss_gib(),
            "peak_after_hamiltonian": peak_rss_gib(),
        },
        "molecule_generated_this_run": generated,
    }
    save_json(manifest_path, manifest)
    print(f"Hamiltonian checkpoint: {manifest_path}", flush=True)

    if args.hamiltonian_only:
        print("Stopped after Hamiltonian generation as requested.", flush=True)
        return

    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError as exc:
        raise ImportError("MPO construction requires pyblock2") from exc

    scratch_context = None
    if args.scratch is None:
        scratch_context = tempfile.TemporaryDirectory(
            prefix="n2_631g_mpo_probe_"
        )
        scratch = Path(scratch_context.name)
    else:
        scratch = args.scratch.expanduser().resolve()
        scratch.mkdir(parents=True, exist_ok=True)

    driver = DMRGDriver(
        scratch=str(scratch),
        symm_type=SymmetryTypes.SGB,
        n_threads=int(args.threads),
        n_mkl_threads=int(args.n_mkl_threads),
        stack_mem=int(args.stack_mem_gb * 1024**3),
        min_mpo_mem=True,
        compressed_mps_storage=True,
    )
    driver.initialize_system(
        n_sites=active_space["n_qubits"], pauli_mode=True
    )
    print(
        f"Building raw Pauli MPO with stack_mem={args.stack_mem_gb:.3f} GiB, "
        f"sum_mpo_mod={args.sum_mpo_mod}; {rss_text()}",
        flush=True,
    )
    try:
        start = perf_counter()
        mpo = build_block2_qubit_mpo(
            driver,
            stream,
            n_qubits=active_space["n_qubits"],
            builder=args.mpo_builder,
            cutoff=args.mpo_cutoff,
            sum_mpo_mod=args.sum_mpo_mod,
            iprint=args.iprint,
        )
        mpo_seconds = perf_counter() - start
        dimensions = mpo_bond_dimensions(mpo)
        artifact = save_block2_mpo(mpo, mpo_path)
        manifest.update(
            {
                "status": "mpo_ready",
                "mpo": artifact,
                "mpo_bond_dimensions": dimensions,
                "maximum_mpo_bond_dimension": (
                    None if not dimensions else max(dimensions)
                ),
            }
        )
        manifest["timings_seconds"]["mpo_construction"] = mpo_seconds
        manifest["memory_gib"].update(
            {
                "current_after_mpo": current_rss_gib(),
                "peak_after_mpo": peak_rss_gib(),
            }
        )
        save_json(manifest_path, manifest)
        print(
            f"MPO constructed in {mpo_seconds:.3f} s; {rss_text()}",
            flush=True,
        )
        print(
            "Maximum MPO bond dimension: "
            f"{manifest['maximum_mpo_bond_dimension']}",
            flush=True,
        )
        print(
            f"Saved MPO: {mpo_path} ({artifact['bytes'] / 1024**2:.1f} MiB)",
            flush=True,
        )
        print(f"Final manifest: {manifest_path}", flush=True)
    except Exception as exc:
        manifest["status"] = "mpo_failed"
        manifest["mpo_error"] = f"{type(exc).__name__}: {exc}"
        manifest["memory_gib"].update(
            {
                "current_at_mpo_failure": current_rss_gib(),
                "peak_at_mpo_failure": peak_rss_gib(),
            }
        )
        save_json(manifest_path, manifest)
        raise
    finally:
        driver.finalize()
        if scratch_context is not None:
            scratch_context.cleanup()


if __name__ == "__main__":
    main()
