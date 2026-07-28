"""Warm-started PyBlock2 DMRG benchmarks for OpenFermion qubit Hamiltonians."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from openfermion import QubitOperator


def save_block2_mps(mps, destination: str | Path) -> dict:
    """Save a complete Block2 MPS restart directory."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    mps.save_data()
    info_path = destination / "mps_info.bin"
    mps.info.save_data(str(info_path))

    copied_files = [info_path.name]
    source_files = []
    for site in range(mps.n_sites + 1):
        source_files.extend(
            (
                mps.info.get_filename(False, site),
                mps.info.get_filename(True, site),
            )
        )
    source_files.extend(
        mps.get_filename(site) for site in range(-1, mps.n_sites)
    )
    for source_name in source_files:
        source = Path(source_name)
        if source.is_file():
            target = destination / source.name
            shutil.copy2(source, target)
            copied_files.append(target.name)

    manifest = {
        "format": "block2_mps_restart_directory",
        "tag": str(mps.info.tag),
        "n_sites": int(mps.n_sites),
        "files": sorted(set(copied_files)),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "path": str(destination),
        "manifest": str(manifest_path),
        "file_count": len(manifest["files"]),
    }


def save_block2_mpo(mpo, destination: str | Path) -> dict:
    """Save a constructed Block2 MPO using its native serialization."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mpo.save_data(str(destination))
    return {
        "path": str(destination),
        "format": "block2_native_mpo",
        "tag": str(mpo.tag),
        "n_sites": int(mpo.n_sites),
        "bytes": destination.stat().st_size,
    }


def _require_block2():
    try:
        from pyblock2.algebra.core import MPS, SubTensor, Tensor
        from pyblock2.algebra.io import MPSTools
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError as exc:
        raise ImportError(
            "Warm-started qubit DMRG requires pyblock2."
        ) from exc
    return MPS, SubTensor, Tensor, MPSTools, DMRGDriver, SymmetryTypes


def _dense_mps_arrays(state, n_qubits: int, cutoff: float):
    """Factor a dense MSB-ordered qubit state into left-to-right MPS arrays."""
    state = np.asarray(state).reshape(-1)
    if state.size != 1 << n_qubits:
        raise ValueError(
            f"state has length {state.size}; expected {1 << n_qubits}"
        )
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"warm-start state has norm {norm}, expected 1")

    work = state.reshape(1, -1)
    arrays = []
    left_dim = 1
    for site in range(n_qubits - 1):
        matrix = work.reshape(left_dim * 2, -1)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        if cutoff > 0:
            keep = singular_values > cutoff
            if not np.any(keep):
                keep[np.argmax(singular_values)] = True
            u = u[:, keep]
            singular_values = singular_values[keep]
            vh = vh[keep]
        right_dim = len(singular_values)
        array = u.reshape(left_dim, 2, right_dim)
        arrays.append(array[0] if site == 0 else array)
        work = singular_values[:, None] * vh
        left_dim = right_dim
    arrays.append(work.reshape(left_dim, 2))
    return arrays


def dense_state_to_block2_pauli_mps(
    driver,
    state,
    n_qubits: int,
    *,
    tag: str,
    cutoff: float = 1e-20,
):
    """Import an OpenFermion/quimb-ordered dense state into Pauli-mode Block2.

    Block2's Pauli local basis is ordered oppositely to the conventional
    ``|0>, |1>`` basis (its primitive Z matrix is ``diag(-1, +1)``).  Flipping
    each physical index makes the imported state consistent with standard
    OpenFermion Pauli operators.
    """
    MPS, SubTensor, Tensor, MPSTools, _, _ = _require_block2()
    arrays = _dense_mps_arrays(state, n_qubits, cutoff)
    arrays = [
        np.flip(array, axis=0 if site == 0 else 1)
        for site, array in enumerate(arrays)
    ]

    quantum_label = driver.vacuum
    tensors = [
        Tensor(
            blocks=[
                SubTensor(
                    q_labels=(quantum_label,) * array.ndim,
                    reduced=np.asarray(array),
                )
            ]
        )
        for array in arrays
    ]
    py_mps = MPS(tensors=tensors)
    basis = [Counter({quantum_label: 2}) for _ in range(n_qubits)]
    block2_mps = MPSTools.to_block2(
        py_mps,
        basis,
        center=0,
        tag=tag,
        left_vacuum=driver.left_vacuum,
    )
    return driver.adjust_mps(block2_mps, dot=2)[0]


def _product_state_pauli_mps(
    basis_index: int,
    coefficient: complex,
    n_qubits: int,
    quantum_label,
):
    """Return one conventional computational determinant as a pyblock MPS."""
    MPS, SubTensor, Tensor, _, _, _ = _require_block2()
    bits = [
        (int(basis_index) >> (n_qubits - 1 - site)) & 1
        for site in range(n_qubits)
    ]
    tensors = []
    for site, bit in enumerate(bits):
        # Block2 Pauli mode orders its local basis as conventional |1>, |0>.
        physical_index = 1 - bit
        if site == 0:
            array = np.zeros((2, 1), dtype=complex)
            array[physical_index, 0] = coefficient
        elif site == n_qubits - 1:
            array = np.zeros((1, 2), dtype=complex)
            array[0, physical_index] = 1.0
        else:
            array = np.zeros((1, 2, 1), dtype=complex)
            array[0, physical_index, 0] = 1.0
        tensors.append(
            Tensor(
                blocks=[
                    SubTensor(
                        q_labels=(quantum_label,) * array.ndim,
                        reduced=array,
                    )
                ]
            )
        )
    return MPS(tensors=tensors)


def sparse_state_to_block2_pauli_mps(
    driver,
    basis_indices,
    coefficients,
    n_qubits: int,
    *,
    tag: str,
    batch_size: int = 32,
    compression_cutoff: float = 1e-13,
):
    """Import a selected-CI state without constructing a dense-state SVD.

    Every supplied determinant is retained. Determinants are added in
    descending coefficient magnitude and the growing direct-sum MPS is
    compressed after bounded batches. Compression removes numerically null
    Schmidt directions; it is not a coefficient/determinant selection.
    """
    MPS, _, _, MPSTools, _, _ = _require_block2()
    indices = np.asarray(basis_indices, dtype=np.int64).reshape(-1)
    coeffs = np.asarray(coefficients, dtype=complex).reshape(-1)
    if indices.size != coeffs.size or indices.size == 0:
        raise ValueError("basis_indices and coefficients must have equal size")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("sparse warm-start basis indices must be unique")
    if np.any(indices < 0) or np.any(indices >= 1 << n_qubits):
        raise ValueError("sparse warm-start basis index is out of range")
    norm = np.linalg.norm(coeffs)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"sparse warm-start state has norm {norm}, expected 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    order = np.argsort(-np.abs(coeffs), kind="stable")
    quantum_label = driver.vacuum
    py_mps = None
    compression_errors = []
    for count, position in enumerate(order, start=1):
        determinant = _product_state_pauli_mps(
            int(indices[position]),
            coeffs[position],
            n_qubits,
            quantum_label,
        )
        py_mps = determinant if py_mps is None else py_mps + determinant
        if count % batch_size == 0 or count == len(order):
            compression_errors.append(
                float(
                    py_mps.compress(
                        k=-1,
                        cutoff=float(compression_cutoff),
                        left=True,
                    )
                )
            )

    bond_dimensions = [
        int(sum(counter.values())) for counter in py_mps.get_bond_dims()
    ]
    constructed_norm = float(np.sqrt(np.real_if_close(py_mps | py_mps)))
    if not np.isclose(constructed_norm, 1.0, atol=1e-10):
        raise RuntimeError(
            "sparse MPS construction changed the state norm: "
            f"{constructed_norm}"
        )
    basis = [Counter({quantum_label: 2}) for _ in range(n_qubits)]
    block2_mps = MPSTools.to_block2(
        py_mps,
        basis,
        center=0,
        tag=tag,
        left_vacuum=driver.left_vacuum,
    )
    block2_mps = driver.adjust_mps(block2_mps, dot=2)[0]
    metadata = {
        "method": "coefficient_ordered_determinant_sum",
        "determinants": int(len(indices)),
        "batch_size": int(batch_size),
        "compression_cutoff": float(compression_cutoff),
        "largest_reported_compression_error": max(
            compression_errors, default=0.0
        ),
        "max_constructed_mps_bond": max(bond_dimensions, default=1),
        "constructed_state_norm": constructed_norm,
    }
    return block2_mps, metadata


def _needs_complex_driver(hamiltonian: QubitOperator, state) -> bool:
    if state is not None and np.linalg.norm(np.asarray(state).imag) > 1e-12:
        return True
    for term, coefficient in hamiltonian.terms.items():
        num_y = sum(pauli == "Y" for _, pauli in term)
        mapped_coefficient = complex(coefficient) * ((-1j) ** num_y)
        if abs(mapped_coefficient.imag) > 1e-12:
            return True
    return False


def _build_pauli_mpo(driver, hamiltonian: QubitOperator, *, iprint: int = 0):
    """Build the standard-Pauli MPO in Block2's reversed local basis."""
    builder = driver.expr_builder()
    for term, coefficient in hamiltonian.terms.items():
        if not term:
            builder.add_const(coefficient)
            continue
        operators = "".join(pauli for _, pauli in term)
        indices = [int(qubit) for qubit, _ in term]
        num_y = operators.count("Y")
        mapped_coefficient = complex(coefficient) * ((-1j) ** num_y)
        mapped_coefficient = np.real_if_close(mapped_coefficient).item()
        builder.add_term(operators, indices, mapped_coefficient)
    return driver.get_mpo(builder.finalize(), iprint=iprint)


def _infer_mpo_bond_dimension(mpo):
    try:
        from .mpo import infer_largest_mpo_bond_dimension

        return infer_largest_mpo_bond_dimension(mpo, verbose=False)
    except Exception:
        return None


def run_block2_qubit_dmrg_curve(
    *,
    label: str,
    hamiltonian: QubitOperator,
    exact_state=None,
    sparse_state=None,
    exact_energy: float,
    warm_start_energy: float | None = None,
    n_qubits: int,
    bond_dims,
    dmrg_sweeps: int = 100,
    dmrg_tolerance: float = 1.6e-3,
    mps_cutoff: float = 1e-20,
    initial_state: str = "cisd",
    sparse_batch_size: int = 32,
    sparse_compression_cutoff: float = 1e-13,
    full_curve: bool = False,
    n_threads: int = 4,
    stack_mem_gb: float = 0.5,
    davidson_threshold: float = 1e-10,
    verbose: bool = False,
    scratch: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """Benchmark Pauli-mode Block2 DMRG with a selected-CI warm start.

    A fresh copy of the imported warm-start MPS is used at every tested bond
    dimension, matching the current quimb benchmark's per-bond-dimension warm
    start.  Random initialization remains available for control runs.
    """
    _, _, _, _, DMRGDriver, SymmetryTypes = _require_block2()
    if initial_state not in {"cisd", "exact_fci", "random"}:
        raise ValueError(
            "initial_state must be 'cisd', 'exact_fci', or 'random'"
        )
    if dmrg_sweeps < 1:
        raise ValueError("dmrg_sweeps must be positive")

    sparse_coefficients = None if sparse_state is None else sparse_state[1]
    state_for_dtype = (
        exact_state if exact_state is not None else sparse_coefficients
    )
    use_complex = _needs_complex_driver(hamiltonian, state_for_dtype)
    symmetry_type = SymmetryTypes.SGB
    if use_complex:
        symmetry_type |= SymmetryTypes.CPX

    scratch_obj = None
    if scratch is None:
        scratch_obj = tempfile.TemporaryDirectory(
            prefix="block2_qubit_benchmark_"
        )
        scratch_path = Path(scratch_obj.name)
    else:
        scratch_path = Path(scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)

    driver = DMRGDriver(
        scratch=str(scratch_path),
        symm_type=symmetry_type,
        n_threads=int(n_threads),
        n_mkl_threads=1,
        stack_mem=int(stack_mem_gb * 1024**3),
    )
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)

    try:
        mpo_start = perf_counter()
        mpo = _build_pauli_mpo(
            driver, hamiltonian, iprint=2 if verbose else 0
        )
        mpo_seconds = perf_counter() - mpo_start
        mpo_bond_dimension = _infer_mpo_bond_dimension(mpo)
        artifacts = {}
        if artifact_dir is not None:
            artifact_dir = Path(artifact_dir)
            artifacts["mpo"] = save_block2_mpo(
                mpo,
                artifact_dir / f"{label}_hamiltonian_mpo.block2.bin",
            )
            print(
                f"{label}: saved Hamiltonian MPO to "
                f"{artifacts['mpo']['path']}",
                flush=True,
            )

        warm_mps = None
        imported_energy = None
        construction = None
        if initial_state != "random":
            if sparse_state is not None:
                warm_mps, construction = sparse_state_to_block2_pauli_mps(
                    driver,
                    sparse_state[0],
                    sparse_state[1],
                    n_qubits,
                    tag=f"WARM-{label}",
                    batch_size=sparse_batch_size,
                    compression_cutoff=sparse_compression_cutoff,
                )
            elif exact_state is not None:
                warm_mps = dense_state_to_block2_pauli_mps(
                    driver,
                    exact_state,
                    n_qubits,
                    tag=f"WARM-{label}",
                    cutoff=mps_cutoff,
                )
                construction = {"method": "dense_sequential_svd"}
            else:
                raise ValueError("a warm-start state was requested but absent")
            # Avoid ``get_identity_mpo`` here: some pyblock2 versions return a
            # real identity MPO even for a CPX driver.
            imported_energy = driver.expectation(warm_mps, mpo, warm_mps)
            imported_energy = float(np.real_if_close(imported_energy))
            expected_warm_energy = (
                exact_energy
                if warm_start_energy is None
                else float(warm_start_energy)
            )
            if abs(imported_energy - expected_warm_energy) > 1e-7:
                raise RuntimeError(
                    "Block2 warm-start energy check failed: "
                    f"{imported_energy} versus {expected_warm_energy}"
                )
            if artifact_dir is not None:
                artifacts["warm_start_mps"] = save_block2_mps(
                    warm_mps,
                    artifact_dir
                    / f"{label}_{initial_state}_warm_start_mps.block2",
                )
                print(
                    f"{label}: saved {initial_state} warm-start MPS to "
                    f"{artifacts['warm_start_mps']['path']}",
                    flush=True,
                )

        rows = []
        first_converged_bd = None
        thresholds = [float(davidson_threshold)] * dmrg_sweeps
        for bond_dim in bond_dims:
            bond_dim = int(bond_dim)
            tag = f"KET-{label}-BD{bond_dim}"
            if warm_mps is not None:
                ket = driver.copy_mps(warm_mps, tag=tag)
                noises = [0.0] * dmrg_sweeps
            else:
                ket = driver.get_random_mps(
                    tag=tag, bond_dim=bond_dim, nroots=1
                )
                initial_noises = [1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6]
                noises = (
                    initial_noises[:dmrg_sweeps]
                    + [0.0] * max(0, dmrg_sweeps - len(initial_noises))
                )

            start = perf_counter()
            energy = driver.dmrg(
                mpo,
                ket,
                n_sweeps=dmrg_sweeps,
                bond_dims=[bond_dim] * dmrg_sweeps,
                noises=noises,
                thrds=thresholds,
                dav_max_iter=50,
                iprint=1 if verbose else 0,
            )
            seconds = perf_counter() - start
            energy = float(np.real_if_close(energy))
            error = abs(energy - exact_energy)
            converged = error <= dmrg_tolerance
            if converged and first_converged_bd is None:
                first_converged_bd = bond_dim

            row = {
                "frame": label,
                "backend": "block2_pauli",
                "bond_dim": bond_dim,
                "energy": energy,
                "abs_energy_error": error,
                "within_dmrg_tolerance": converged,
                "dmrg_seconds": seconds,
                "max_result_mps_bond": bond_dim,
            }
            rows.append(row)
            print(
                f"{label:26s} bond_dim={bond_dim:3d} "
                f"E={energy:.12f} |dE|={error:.3e} "
                f"seconds={seconds:.1f}",
                flush=True,
            )
            if converged and not full_curve:
                print(
                    f"{label}: reached chemical accuracy at "
                    f"bond_dim={bond_dim}; stopping this frame.",
                    flush=True,
                )
                break

        summary = {
            "frame": label,
            "backend": "block2_pauli",
            "first_converged_bond_dim": first_converged_bd,
            "converged_within_grid": first_converged_bd is not None,
            "mpo_bond_dimension": mpo_bond_dimension,
            "mpo_build_seconds": mpo_seconds,
            "warm_start": initial_state != "random",
            "warm_start_kind": initial_state,
            "imported_warm_start_energy": imported_energy,
            "expected_warm_start_energy": warm_start_energy,
            "warm_start_construction": construction,
            "saved_tensor_networks": artifacts,
            "complex_driver": use_complex,
            "davidson_threshold": davidson_threshold,
        }
        return rows, summary
    finally:
        if scratch_obj is not None:
            scratch_obj.cleanup()
