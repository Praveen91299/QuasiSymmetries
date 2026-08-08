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

from .mps_unitary import (
    _standard_mps_arrays,
    compose_mps_unitaries,
    project_mps_onto_prefix_configurations,
)
from .bs.utils import PauliTermStream, as_pauli_term_stream


def _rss_message() -> str:
    """Best-effort resident-memory report for diagnosing native allocations."""
    try:
        import psutil

        rss_gib = psutil.Process().memory_info().rss / 1024**3
        return f" RSS={rss_gib:.3f} GiB"
    except Exception:
        return ""


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


def save_qubit_mps_arrays(tensors, destination: str | Path) -> dict:
    """Save a portable open-boundary qubit MPS as a NumPy archive."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(tensor) for tensor in tensors]
    np.savez_compressed(
        destination,
        **{f"tensor_{site:05d}": array for site, array in enumerate(arrays)},
    )
    return {
        "path": str(destination),
        "format": "numpy_open_boundary_qubit_mps",
        "n_sites": len(arrays),
        "max_bond_dimension": max(
            (array.shape[-1] for array in arrays[:-1]), default=1
        ),
        "bytes": destination.stat().st_size,
    }


def load_qubit_mps_arrays(source: str | Path) -> list[np.ndarray]:
    """Load an MPS written by :func:`save_qubit_mps_arrays`."""
    with np.load(Path(source), allow_pickle=False) as archive:
        names = sorted(
            archive.files,
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        return [np.asarray(archive[name]) for name in names]


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


def block2_pauli_mps_to_arrays(driver, mps, *, tag: str) -> list[np.ndarray]:
    """Export a Block2 Pauli MPS to conventional-basis NumPy tensors.

    Parameters
    ----------
    driver
        Live ``DMRGDriver`` that owns the source MPS.
    mps
        Block2 Pauli MPS to export; it is copied and not modified.
    tag
        Unique scratch tag used for the temporary one-site-form copy.

    Returns
    -------
    tensors
        List of open-boundary arrays with shape
        ``(left_bond, 2, right_bond)`` in conventional ``|0>, |1>`` order.
    """
    _, _, _, MPSTools, _, _ = _require_block2()
    copied = driver.copy_mps(mps, tag=tag)
    copied = driver.adjust_mps(copied, dot=1)[0]
    py_mps = MPSTools.from_block2(copied)
    arrays = []
    for site in range(copied.n_sites):
        blocks = py_mps.tensors[site].blocks
        if len(blocks) != 1:
            raise ValueError(
                "Pauli-mode reference MPS must have one tensor block per site."
            )
        array = np.asarray(blocks[0].reduced).copy()
        if site == 0:
            array = array.reshape(1, 2, array.shape[-1])
        elif site == copied.n_sites - 1:
            array = array.reshape(array.shape[0], 2, 1)
        elif array.ndim != 3:
            raise ValueError(
                f"Unexpected exported MPS tensor rank at site {site}."
            )
        # Block2 Pauli mode stores |1>, |0>; portable arrays use |0>, |1>.
        arrays.append(array[:, ::-1, :])
    return arrays


def qubit_mps_arrays_to_block2_pauli_mps(
    driver,
    tensors,
    *,
    tag: str,
):
    """Import portable conventional-basis MPS tensors into Block2 Pauli mode.

    Parameters
    ----------
    driver
        Initialized ``DMRGDriver`` whose system uses ``pauli_mode=True``.
        Its real/complex symmetry type must be compatible with ``tensors``.
    tensors
        Open-boundary qubit MPS tensors in conventional local basis order
        ``|0>, |1>``. Tensor shapes are ``(left_bond, 2, right_bond)``;
        rank-two boundary tensors are accepted.
    tag
        Unique Block2 scratch tag assigned to the imported MPS.

    Returns
    -------
    block2_mps
        A two-site-form Block2 Pauli MPS representing the supplied tensors.
        No SVD compression or bond-dimension truncation is performed.
    """
    MPS, SubTensor, Tensor, _, _, _ = _require_block2()
    arrays = _standard_mps_arrays(tensors)
    quantum_label = driver.vacuum
    py_tensors = []
    for site, array in enumerate(arrays):
        # Block2 Pauli mode uses local order |1>, |0>.
        converted = np.asarray(array[:, ::-1, :])
        if site == 0:
            reduced = converted[0, :, :]
            labels = (quantum_label, quantum_label)
        elif site == len(arrays) - 1:
            reduced = converted[:, :, 0]
            labels = (quantum_label, quantum_label)
        else:
            reduced = converted
            labels = (quantum_label, quantum_label, quantum_label)
        py_tensors.append(
            Tensor(
                blocks=[
                    SubTensor(q_labels=labels, reduced=reduced.copy())
                ]
            )
        )
    py_mps = MPS(tensors=py_tensors)
    return pyblock_pauli_mps_to_block2(
        driver, py_mps, len(arrays), tag=tag
    )


def project_block2_pauli_mps_onto_prefix_configurations(
    driver,
    mps,
    retained_configurations,
    *,
    n_prefix_sites: int,
    tag: str = "PROJECTED",
    expected_retained_probability: float | None = None,
    probability_tolerance: float = 1e-10,
):
    """Construct a normalized Block2 MPS projected onto prefix bit strings.

    Parameters
    ----------
    driver
        Live ``DMRGDriver`` that owns ``mps`` and was initialized with
        ``pauli_mode=True``.
    mps
        Source Block2 Pauli MPS. The source is copied for export and is not
        modified.
    retained_configurations
        Iterable of unique binary strings to retain on the leading symmetry
        sites.
    n_prefix_sites
        Number of leading sites represented by each retained string. All later
        sites are left unrestricted.
    tag
        Unique Block2 scratch tag for the returned MPS and temporary export.
    expected_retained_probability
        Optional independently calculated retained probability. When supplied,
        the exact projected norm is checked against it.
    probability_tolerance
        Absolute tolerance for the optional probability consistency check.

    Returns
    -------
    projected_mps, metadata
        ``projected_mps`` is the normalized Block2 Pauli MPS after coherent
        projection. ``metadata`` contains the retained/omitted probabilities,
        trie dimensions, bond dimensions, and whether probability validation
        was performed.

    Notes
    -----
    This function imposes no MPS bond cap and performs no SVD compression. The
    prefix trie is absorbed exactly into the original MPS bonds, after which
    the resulting tensors are imported into Block2.
    """
    if probability_tolerance < 0:
        raise ValueError("probability_tolerance must be nonnegative")
    arrays = block2_pauli_mps_to_arrays(
        driver, mps, tag=f"{tag}-EXPORT"
    )
    projected_arrays, metadata = project_mps_onto_prefix_configurations(
        arrays,
        retained_configurations,
        n_prefix_sites=n_prefix_sites,
        normalize=True,
    )
    retained_probability = metadata["retained_probability"]
    validation_performed = expected_retained_probability is not None
    if validation_performed and not np.isclose(
        retained_probability,
        float(expected_retained_probability),
        rtol=0.0,
        atol=float(probability_tolerance),
    ):
        raise RuntimeError(
            "prefix-search and projected-MPS retained probabilities differ: "
            f"{expected_retained_probability} versus {retained_probability}"
        )
    projected_mps = qubit_mps_arrays_to_block2_pauli_mps(
        driver, projected_arrays, tag=tag
    )
    metadata.update(
        {
            "expected_retained_probability": (
                None
                if expected_retained_probability is None
                else float(expected_retained_probability)
            ),
            "probability_validation_tolerance": float(
                probability_tolerance
            ),
            "probability_validation_performed": validation_performed,
        }
    )
    return projected_mps, metadata


def evaluate_block2_prefix_projection_energy(
    driver,
    mps,
    mpo,
    retained_configurations,
    *,
    n_prefix_sites: int,
    tag: str = "PROJECTED",
    original_energy: float | None = None,
    expected_retained_probability: float | None = None,
    probability_tolerance: float = 1e-10,
    iprint: int = 0,
):
    """Project a Block2 MPS and measure its actual energy change exactly.

    Parameters
    ----------
    driver
        Live ``DMRGDriver`` that owns ``mps`` and ``mpo``. It must remain
        initialized for the duration of this call.
    mps
        Original Block2 Pauli MPS. It is exported but is not modified.
    mpo
        Hamiltonian Block2 MPO in the same qubit frame and local basis as
        ``mps``.
    retained_configurations
        Iterable of binary strings retained on the first
        ``n_prefix_sites`` qubits.
    n_prefix_sites
        Number of leading symmetry-qubit sites encoded by each retained
        string. Sites after this prefix are left unrestricted.
    tag
        Unique Block2 scratch tag for the projected MPS and temporary export.
    original_energy
        Optional previously calculated ``<mps|mpo|mps>``. Supplying it avoids
        one expectation contraction. If omitted, the function calculates it.
    expected_retained_probability
        Optional probability obtained independently from the prefix search.
        If supplied, it is checked against the exact projected-MPS norm.
    probability_tolerance
        Absolute tolerance for the optional retained-probability check.
    iprint
        Verbosity passed to Block2 expectation contractions.

    Returns
    -------
    projected_mps, result
        ``projected_mps`` is the normalized Block2 MPS after applying the
        coherent prefix projector. ``result`` reports projection probability,
        omitted probability, original and projected energies, their absolute
        difference, trie dimensions, and input/projected MPS bond dimensions.

    Notes
    -----
    Projection uses an exact prefix-trie tensor network. No SVD, numerical
    cutoff, or maximum bond dimension is applied, so the reported energy change
    is the projection error for the supplied MPS and MPO, up to floating-point
    contraction error.
    """
    projected_mps, projection = (
        project_block2_pauli_mps_onto_prefix_configurations(
            driver,
            mps,
            retained_configurations,
            n_prefix_sites=n_prefix_sites,
            tag=tag,
            expected_retained_probability=expected_retained_probability,
            probability_tolerance=probability_tolerance,
        )
    )
    if original_energy is None:
        original_energy = driver.expectation(
            mps, mpo, mps, iprint=iprint
        )
    projected_energy = driver.expectation(
        projected_mps, mpo, projected_mps, iprint=iprint
    )
    original_energy = float(np.real_if_close(original_energy))
    projected_energy = float(np.real_if_close(projected_energy))
    result = {
        **projection,
        "original_energy": original_energy,
        "projected_energy": projected_energy,
        "actual_projection_energy_error": abs(
            projected_energy - original_energy
        ),
    }
    return projected_mps, result


def evaluate_qubit_mps_arrays_prefix_projection_energy(
    hamiltonian,
    tensors,
    retained_configurations,
    *,
    n_prefix_sites: int,
    expected_retained_probability: float | None = None,
    probability_tolerance: float = 1e-10,
    mpo_builder: str = "blocked_sum",
    mpo_cutoff: float = 1e-10,
    sum_mpo_mod: int = 20,
    n_threads: int = 4,
    stack_mem_gb: float = 0.5,
    scratch: str | Path | None = None,
    tag: str = "PROJECTED-VERIFY",
    iprint: int = 0,
) -> tuple[list[np.ndarray], dict]:
    """Measure prefix-projection energy error from a portable qubit MPS.

    Parameters
    ----------
    hamiltonian
        ``PauliTermStream`` or OpenFermion ``QubitOperator`` represented in the
        same qubit frame as ``tensors``.
    tensors
        Original normalized qubit MPS arrays in conventional ``|0>, |1>``
        basis order and ``(left_bond, 2, right_bond)`` shape.
    retained_configurations
        Iterable of binary strings retained on the first
        ``n_prefix_sites`` MPS sites.
    n_prefix_sites
        Number of leading symmetry sites represented by each retained string.
    expected_retained_probability
        Optional independently calculated retained probability to validate
        against the exact projected norm.
    probability_tolerance
        Absolute tolerance for retained-probability validation.
    mpo_builder
        PyBlock2 Pauli MPO construction route: ``"blocked_sum"`` or
        ``"expression"``.
    mpo_cutoff
        Numerical cutoff used only while constructing the Hamiltonian MPO.
        It does not truncate the projected MPS.
    sum_mpo_mod
        Block size for the blocked-sum MPO construction route.
    n_threads
        Number of Block2 computational threads.
    stack_mem_gb
        Block2 stack-memory allocation in GiB.
    scratch
        Optional Block2 scratch directory. A temporary directory is used when
        omitted.
    tag
        Base scratch tag for imported original and projected MPS objects.
    iprint
        Block2 MPO/expectation verbosity.

    Returns
    -------
    projected_tensors, result
        ``projected_tensors`` is the normalized, exactly projected portable
        MPS. ``result`` contains the retained probability, original/projected
        energies, actual projection error, MPO build time, and bond/trie
        diagnostics.

    Notes
    -----
    This convenience wrapper creates a temporary Block2 driver and rebuilds
    the Hamiltonian MPO. When a live driver and MPO are already available,
    :func:`evaluate_block2_prefix_projection_energy` is more efficient. No SVD
    or bond-dimension cap is applied to the projected MPS in either route.
    """
    _, _, _, _, DMRGDriver, SymmetryTypes = _require_block2()
    arrays = _standard_mps_arrays(tensors)
    n_qubits = len(arrays)
    stream = as_pauli_term_stream(hamiltonian, n_qubits)
    _validate_hermitian_pauli_coefficients(stream)
    projected_arrays, projection = project_mps_onto_prefix_configurations(
        arrays,
        retained_configurations,
        n_prefix_sites=n_prefix_sites,
        normalize=True,
    )
    if expected_retained_probability is not None and not np.isclose(
        projection["retained_probability"],
        float(expected_retained_probability),
        rtol=0.0,
        atol=float(probability_tolerance),
    ):
        raise RuntimeError(
            "prefix-search and projected-MPS retained probabilities differ: "
            f"{expected_retained_probability} versus "
            f"{projection['retained_probability']}"
        )

    use_complex = _needs_complex_driver(stream, None) or any(
        np.linalg.norm(array.imag) > 1e-12
        for array in arrays
        if np.iscomplexobj(array)
    )
    symmetry_type = SymmetryTypes.SGB
    if use_complex:
        symmetry_type |= SymmetryTypes.CPX
    scratch_obj = None
    if scratch is None:
        scratch_obj = tempfile.TemporaryDirectory(
            prefix="block2_projection_verify_"
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
        min_mpo_mem=True,
        compressed_mps_storage=True,
    )
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)
    try:
        mpo_start = perf_counter()
        if mpo_builder == "blocked_sum":
            mpo = _build_pauli_mpo_blocked_sum(
                driver,
                stream,
                n_qubits,
                cutoff=mpo_cutoff,
                sum_mpo_mod=sum_mpo_mod,
                iprint=iprint,
            )
        elif mpo_builder == "expression":
            mpo = _build_pauli_mpo_expression(
                driver, stream, iprint=iprint
            )
        else:
            raise ValueError(
                "mpo_builder must be 'blocked_sum' or 'expression'"
            )
        mpo_seconds = perf_counter() - mpo_start
        original_mps = qubit_mps_arrays_to_block2_pauli_mps(
            driver, arrays, tag=f"{tag}-ORIGINAL"
        )
        projected_mps = qubit_mps_arrays_to_block2_pauli_mps(
            driver, projected_arrays, tag=f"{tag}-PROJECTED"
        )
        original_energy = float(
            np.real_if_close(
                driver.expectation(
                    original_mps, mpo, original_mps, iprint=iprint
                )
            )
        )
        projected_energy = float(
            np.real_if_close(
                driver.expectation(
                    projected_mps, mpo, projected_mps, iprint=iprint
                )
            )
        )
        result = {
            **projection,
            "original_energy": original_energy,
            "projected_energy": projected_energy,
            "actual_projection_energy_error": abs(
                projected_energy - original_energy
            ),
            "expected_retained_probability": (
                None
                if expected_retained_probability is None
                else float(expected_retained_probability)
            ),
            "probability_validation_tolerance": float(
                probability_tolerance
            ),
            "probability_validation_performed": (
                expected_retained_probability is not None
            ),
            "mpo_builder": mpo_builder,
            "mpo_cutoff": float(mpo_cutoff),
            "mpo_build_seconds": mpo_seconds,
            "mpo_bond_dimension": _infer_mpo_bond_dimension(mpo),
            "complex_driver": use_complex,
        }
        return projected_arrays, result
    finally:
        driver.finalize()
        if scratch_obj is not None:
            scratch_obj.cleanup()


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
    """Import an OpenFermion-ordered dense state into Pauli-mode Block2.

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
    dtype,
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
            array = np.zeros((2, 1), dtype=dtype)
            array[physical_index, 0] = coefficient
        elif site == n_qubits - 1:
            array = np.zeros((1, 2), dtype=dtype)
            array[0, physical_index] = 1.0
        else:
            array = np.zeros((1, 2, 1), dtype=dtype)
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


def _pyblock_mps_norm(py_mps) -> float:
    """Return the Hermitian norm, including for complex pyblock tensors."""
    py_mps.canonicalize(0)
    return float(
        np.sqrt(
            sum(
                np.vdot(block.reduced, block.reduced).real
                for block in py_mps.tensors[0].blocks
            )
        )
    )


def _compress_pyblock_mps_with_svd_fallback(
    py_mps,
    *,
    k: int,
    cutoff: float,
    left: bool,
) -> tuple[float, int]:
    """Compress a pyblock MPS with a robust fallback for failed NumPy SVDs.

    Parameters
    ----------
    py_mps
        Mutable ``pyblock2.algebra.core.MPS`` to compress in place.
    k
        Maximum retained bond dimension passed to ``MPS.compress``; ``-1``
        retains every singular direction above ``cutoff``.
    cutoff
        Singular-value threshold passed unchanged to ``MPS.compress``.
    left
        Sweep direction flag passed unchanged to ``MPS.compress``.

    Returns
    -------
    compression_error, fallback_count
        Error reported by pyblock2 and the number of local matrix SVDs for
        which NumPy failed and the rescaled LAPACK ``gesvd`` fallback was used.

    Notes
    -----
    Pyblock2's Python tensor layer calls ``numpy.linalg.svd`` internally. The
    default divide-and-conquer LAPACK driver can occasionally fail on the very
    ill-conditioned direct-sum tensors produced by a full CISD expansion. The
    fallback changes only the numerical factorization algorithm: it does not
    change ``k``, ``cutoff``, determinants, or coefficients.
    """
    original_svd = np.linalg.svd
    fallback_count = 0

    def robust_svd(matrix, *args, **kwargs):
        nonlocal fallback_count
        try:
            return original_svd(matrix, *args, **kwargs)
        except np.linalg.LinAlgError:
            from scipy.linalg import svd as scipy_svd

            array = np.asarray(matrix)
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "non-finite entries encountered during CISD MPS compression"
                )
            scale = float(np.max(np.abs(array))) if array.size else 0.0
            if scale == 0.0:
                return original_svd(array, *args, **kwargs)
            fallback_count += 1
            full_matrices = kwargs.get(
                "full_matrices", args[0] if args else True
            )
            compute_uv = kwargs.get("compute_uv", True)
            result = scipy_svd(
                array / scale,
                full_matrices=bool(full_matrices),
                compute_uv=bool(compute_uv),
                check_finite=True,
                lapack_driver="gesvd",
            )
            if compute_uv:
                u, singular_values, vh = result
                return u, singular_values * scale, vh
            return result * scale

    # The pyblock tensor implementation imports the shared NumPy module and
    # resolves np.linalg.svd at call time. Keep the replacement confined to
    # this single-threaded compression call and restore it even on failure.
    np.linalg.svd = robust_svd
    try:
        error = float(
            py_mps.compress(k=int(k), cutoff=float(cutoff), left=bool(left))
        )
    finally:
        np.linalg.svd = original_svd
    return error, fallback_count


def sparse_state_to_pyblock_pauli_mps(
    driver,
    basis_indices,
    coefficients,
    n_qubits: int,
    *,
    batch_size: int = 32,
    compression_cutoff: float = 1e-13,
):
    """Construct a selected-CI pyblock MPS without a dense-state SVD.

    Every supplied determinant is retained. Determinants are added in
    descending coefficient magnitude and the growing direct-sum MPS is
    compressed after bounded batches. Compression removes numerically null
    Schmidt directions; it is not a coefficient/determinant selection.
    """
    _require_block2()
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
    if np.linalg.norm(coeffs.imag) <= 1e-12:
        coeffs = coeffs.real
        tensor_dtype = float
    else:
        tensor_dtype = complex
    quantum_label = driver.vacuum
    py_mps = None
    compression_errors = []
    svd_fallback_count = 0
    for count, position in enumerate(order, start=1):
        determinant = _product_state_pauli_mps(
            int(indices[position]),
            coeffs[position],
            n_qubits,
            quantum_label,
            tensor_dtype,
        )
        py_mps = determinant if py_mps is None else py_mps + determinant
        if count % batch_size == 0 or count == len(order):
            compression_error, fallbacks = (
                _compress_pyblock_mps_with_svd_fallback(
                    py_mps,
                    k=-1,
                    cutoff=float(compression_cutoff),
                    left=True,
                )
            )
            compression_errors.append(compression_error)
            svd_fallback_count += fallbacks

    bond_dimensions = [
        int(sum(counter.values())) for counter in py_mps.get_bond_dims()
    ]
    constructed_norm = _pyblock_mps_norm(py_mps)
    if not np.isclose(constructed_norm, 1.0, atol=1e-10):
        raise RuntimeError(
            "sparse MPS construction changed the state norm: "
            f"{constructed_norm}"
        )
    metadata = {
        "method": "coefficient_ordered_determinant_sum",
        "determinants": int(len(indices)),
        "batch_size": int(batch_size),
        "compression_cutoff": float(compression_cutoff),
        "largest_reported_compression_error": max(
            compression_errors, default=0.0
        ),
        "svd_fallback_count": int(svd_fallback_count),
        "svd_fallback": "rescaled_scipy_gesvd_after_numpy_failure",
        "max_constructed_mps_bond": max(bond_dimensions, default=1),
        "constructed_state_norm": constructed_norm,
    }
    return py_mps, metadata


def pyblock_pauli_mps_to_block2(driver, py_mps, n_qubits: int, *, tag: str):
    """Convert an already prepared pyblock MPS to Block2 Pauli mode."""
    _, _, _, MPSTools, _, _ = _require_block2()
    quantum_label = driver.vacuum
    basis = [Counter({quantum_label: 2}) for _ in range(n_qubits)]
    block2_mps = MPSTools.to_block2(
        py_mps,
        basis,
        center=0,
        tag=tag,
        left_vacuum=driver.left_vacuum,
    )
    return driver.adjust_mps(block2_mps, dot=2)[0]


def sparse_state_to_block2_pauli_mps(
    driver,
    basis_indices,
    coefficients,
    n_qubits: int,
    *,
    tag: str,
    batch_size: int = 32,
    compression_cutoff: float = 1e-13,
    unitaries=(),
    transform_max_bond: int | None = None,
    transform_cutoff: float = 1e-13,
):
    """Build a sparse-CI MPS and apply optional composable unitary objects."""
    py_mps, metadata = sparse_state_to_pyblock_pauli_mps(
        driver,
        basis_indices,
        coefficients,
        n_qubits,
        batch_size=batch_size,
        compression_cutoff=compression_cutoff,
    )
    composed = None
    if unitaries:
        composed = compose_mps_unitaries(unitaries, n_qubits=n_qubits)
        parsed_gates = composed.get_parsed_gates()
        final_permutation = composed.get_permutation()
    else:
        parsed_gates = ()
        final_permutation = None

    if parsed_gates or final_permutation is not None:
        transform_metadata = transform_pyblock_pauli_mps(
            py_mps,
            parsed_gates=parsed_gates,
            permutation=final_permutation,
            max_bond=transform_max_bond,
            cutoff=transform_cutoff,
        )
        if composed is not None:
            transform_metadata["unitary_composition"] = {
                "component_types": list(composed.component_types),
                "number_of_components": len(composed.component_types),
                "number_of_composed_gates": len(composed.parsed_gates),
                "composed_permutation": list(composed.permutation),
            }
        metadata["tensor_circuit"] = transform_metadata
    block2_mps = pyblock_pauli_mps_to_block2(
        driver, py_mps, n_qubits, tag=tag
    )
    return block2_mps, metadata


def _mps_site_array(py_mps, site: int) -> np.ndarray:
    blocks = py_mps.tensors[site].blocks
    if len(blocks) != 1:
        raise ValueError("Pauli-mode pyblock MPS must have one block per site")
    array = np.asarray(blocks[0].reduced)
    if site == 0:
        return array.reshape(1, 2, array.shape[-1])
    if site == len(py_mps.tensors) - 1:
        return array.reshape(array.shape[0], 2, 1)
    return array


def _set_mps_site_array(py_mps, site: int, array: np.ndarray) -> None:
    if site == 0:
        array = array[0]
    elif site == len(py_mps.tensors) - 1:
        array = array[:, :, 0]
    py_mps.tensors[site].blocks[0].reduced = np.asarray(array)


def _internal_gate(gate: np.ndarray) -> np.ndarray:
    """Convert conventional |00>,|01>,|10>,|11> order to Block2 Pauli order."""
    permutation = np.array([3, 2, 1, 0])
    gate = np.asarray(gate)
    return gate[np.ix_(permutation, permutation)]


def _apply_one_site_gate(py_mps, site: int, gate: np.ndarray) -> None:
    array = _mps_site_array(py_mps, site)
    internal = np.asarray(gate)[np.ix_([1, 0], [1, 0])]
    array = np.einsum("ab,lbr->lar", internal, array, optimize=True)
    _set_mps_site_array(py_mps, site, array)


def _apply_two_site_gate(
    py_mps,
    left: int,
    gate: np.ndarray,
    *,
    max_bond: int | None,
    cutoff: float,
) -> tuple[float, int]:
    """Apply a nearest-neighbour gate and split it with a capped SVD."""
    if left < 0 or left + 1 >= len(py_mps.tensors):
        raise IndexError(f"invalid two-site gate location {left}")
    # Put the orthogonality center on the acted pair. Only in this gauge are
    # the singular values below actual Schmidt values, so a bond cap has a
    # physical, gauge-independent meaning.
    py_mps.canonicalize(left)
    a = _mps_site_array(py_mps, left)
    b = _mps_site_array(py_mps, left + 1)
    theta = np.tensordot(a, b, axes=(-1, 0))
    gate4 = _internal_gate(gate).reshape(2, 2, 2, 2)
    theta = np.einsum("abij,lijr->labr", gate4, theta, optimize=True)
    left_dim, _, _, right_dim = theta.shape
    matrix = theta.reshape(left_dim * 2, 2 * right_dim)
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = len(singular_values)
    if cutoff > 0:
        keep = max(1, int(np.count_nonzero(singular_values > cutoff)))
    if max_bond is not None:
        keep = min(keep, int(max_bond))
    discarded = float(np.linalg.norm(singular_values[keep:]))
    u = u[:, :keep]
    singular_values = singular_values[:keep]
    vh = vh[:keep]
    _set_mps_site_array(py_mps, left, u.reshape(left_dim, 2, keep))
    _set_mps_site_array(
        py_mps,
        left + 1,
        (singular_values[:, None] * vh).reshape(keep, 2, right_dim),
    )
    return discarded, keep


def _cnot_gate(control_left: bool) -> np.ndarray:
    gate = np.zeros((4, 4))
    for left_bit in range(2):
        for right_bit in range(2):
            source = 2 * left_bit + right_bit
            if control_left:
                target_left = left_bit
                target_right = right_bit ^ left_bit
            else:
                target_left = left_bit ^ right_bit
                target_right = right_bit
            gate[2 * target_left + target_right, source] = 1.0
    return gate


def transform_pyblock_pauli_mps(
    py_mps,
    *,
    parsed_gates=(),
    permutation=None,
    max_bond: int | None,
    cutoff: float,
) -> dict:
    """Apply a composed basis-gate circuit directly to a pyblock MPS."""
    if max_bond is not None and max_bond < 1:
        raise ValueError("transform_max_bond must be positive")
    n_qubits = len(py_mps.tensors)
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    fermionic_swap = swap.copy()
    fermionic_swap[3, 3] = -1
    truncation_errors: list[float] = []
    largest_bond = 1
    gate_counts = Counter()

    def apply_two(left, gate, kind):
        nonlocal largest_bond
        error, bond = _apply_two_site_gate(
            py_mps, left, gate, max_bond=max_bond, cutoff=cutoff
        )
        truncation_errors.append(error)
        largest_bond = max(largest_bond, bond)
        gate_counts[kind] += 1

    def apply_nonlocal_two(qubit_a, qubit_b, gate, kind):
        if qubit_a == qubit_b:
            raise ValueError(f"{kind} requires two distinct qubits")
        if qubit_a < qubit_b:
            swaps = list(range(qubit_b - 1, qubit_a, -1))
            for left in swaps:
                apply_two(left, swap, "routing_swap")
            apply_two(qubit_a, gate, kind)
            for left in reversed(swaps):
                apply_two(left, swap, "routing_swap")
        else:
            swaps = list(range(qubit_b, qubit_a - 1))
            for left in swaps:
                apply_two(left, swap, "routing_swap")
            # The physical left/right order is opposite to the gate's operand
            # order, so exchange both its input and output tensor legs.
            reversed_gate = swap @ np.asarray(gate) @ swap
            apply_two(qubit_a - 1, reversed_gate, kind)
            for left in reversed(swaps):
                apply_two(left, swap, "routing_swap")

    one_qubit_gates = {
        # Keep numerically real circuits in Block2's real scalar family.
        # Declaring H/X as complex here caused MPSTools to create a CPX MPS
        # even when every tensor entry was real, which is incompatible with a
        # real MPO and MovingEnvironment.
        "X": np.array([[0, 1], [1, 0]], dtype=float),
        "H": np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2),
        "S": np.diag([1, 1j]),
        "Sdg": np.diag([1, -1j]),
    }
    for parsed_gate in parsed_gates:
        name = str(parsed_gate[0])
        if name in one_qubit_gates:
            _apply_one_site_gate(
                py_mps, int(parsed_gate[1]), one_qubit_gates[name]
            )
            gate_counts[name] += 1
        elif name == "PHASE":
            phase_gate = np.diag([1.0, np.exp(parsed_gate[2])])
            _apply_one_site_gate(
                py_mps,
                int(parsed_gate[1]),
                np.real_if_close(phase_gate),
            )
            gate_counts[name] += 1
        elif name == "CNOT":
            control, target = int(parsed_gate[1]), int(parsed_gate[2])
            apply_nonlocal_two(
                control, target, _cnot_gate(True), "CNOT"
            )
        elif name == "SWAP":
            apply_nonlocal_two(
                int(parsed_gate[1]), int(parsed_gate[2]), swap, "SWAP"
            )
        elif name == "FSWAP":
            apply_nonlocal_two(
                int(parsed_gate[1]),
                int(parsed_gate[2]),
                fermionic_swap,
                "FSWAP",
            )
        elif name == "FGIVENS":
            theta = float(parsed_gate[3])
            cosine, sine = np.cos(theta), np.sin(theta)
            givens_gate = np.array(
                [
                    [1, 0, 0, 0],
                    [0, cosine, -sine, 0],
                    [0, sine, cosine, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
            apply_nonlocal_two(
                int(parsed_gate[1]),
                int(parsed_gate[2]),
                givens_gate,
                "FGIVENS",
            )
        else:
            raise ValueError(f"unsupported MPS basis gate {parsed_gate!r}")

    if permutation is not None:
        permutation = np.asarray(permutation, dtype=int)
        if sorted(permutation.tolist()) != list(range(n_qubits)):
            raise ValueError("invalid final qubit permutation")
        desired = np.argsort(permutation).tolist()
        current = list(range(n_qubits))
        for destination, label in enumerate(desired):
            source = current.index(label)
            for left in range(source - 1, destination - 1, -1):
                apply_two(left, swap, "permutation_swap")
                current[left], current[left + 1] = (
                    current[left + 1],
                    current[left],
                )

    norm_before_normalization = _pyblock_mps_norm(py_mps)
    if norm_before_normalization == 0:
        raise RuntimeError("tensor-circuit truncation produced a zero MPS")
    first = _mps_site_array(py_mps, 0) / norm_before_normalization
    _set_mps_site_array(py_mps, 0, first)
    final_bonds = [
        int(sum(counter.values())) for counter in py_mps.get_bond_dims()
    ]
    return {
        "method": "pyblock2_local_gate_svd",
        "max_bond_cap": max_bond,
        "svd_cutoff": float(cutoff),
        "gate_counts": dict(gate_counts),
        "number_of_svd_splits": len(truncation_errors),
        "root_sum_squared_discarded_singular_values": float(
            np.linalg.norm(truncation_errors)
        ),
        "largest_single_split_discarded_norm": max(
            truncation_errors, default=0.0
        ),
        "norm_before_final_normalization": norm_before_normalization,
        "max_final_mps_bond": max(final_bonds, default=1),
        "max_intermediate_split_bond": largest_bond,
    }


def _mask_to_block2_word(mask, n_qubits):
    """Return Block2 operator text, sites, and number of physical Y factors."""
    x, z = mask
    operators = []
    indices = []
    num_y = 0
    for qubit in range(n_qubits):
        xb = (x >> qubit) & 1
        zb = (z >> qubit) & 1
        if not (xb or zb):
            continue
        indices.append(qubit)
        if xb and zb:
            operators.append("Y")
            num_y += 1
        elif xb:
            operators.append("X")
        else:
            operators.append("Z")
    return "".join(operators), indices, num_y


def _needs_complex_driver(hamiltonian, state) -> bool:
    if state is not None and np.linalg.norm(np.asarray(state).imag) > 1e-12:
        return True
    stream = as_pauli_term_stream(hamiltonian)
    for item in stream.terms:
        num_y = bin(int(item.mask[0] & item.mask[1])).count("1")
        mapped_coefficient = item.signed_coefficient * ((-1j) ** num_y)
        if abs(mapped_coefficient.imag) > 1e-12:
            return True
    return False


def _unitaries_need_complex_driver(unitaries, n_qubits: int) -> bool:
    """Return whether an MPS circuit can produce complex-valued tensors.

    Driver selection must account for the transformed warm start as well as
    the Hamiltonian and input coefficients. A real CISD state acted on by an
    ``S``/``Sdg`` Clifford is represented by a CPX Block2 MPS even when the
    transformed Hermitian Pauli Hamiltonian has a real-mode MPO.
    """
    if not unitaries:
        return False
    composed = compose_mps_unitaries(unitaries, n_qubits=n_qubits)
    for gate in composed.get_parsed_gates():
        name = str(gate[0])
        if name in {"S", "Sdg"}:
            return True
        if name == "PHASE":
            phase = np.exp(complex(gate[2]))
            if abs(phase.imag) > 1e-12:
                return True
    return False


def _validate_hermitian_pauli_coefficients(
    hamiltonian, *, tolerance: float = 1e-10
) -> None:
    """A Hermitian Pauli expansion must have real coefficients."""
    largest_imaginary = max(
        (
            abs(item.signed_coefficient.imag)
            for item in as_pauli_term_stream(hamiltonian).terms
        ),
        default=0.0,
    )
    if largest_imaginary > tolerance:
        raise ValueError(
            "DMRG Hamiltonian is not Hermitian in the Pauli basis: largest "
            f"imaginary coefficient is {largest_imaginary:.3e}. Hermitize "
            "the operator as (H + H†) / 2 before constructing its MPO."
        )


def _build_pauli_mpo_expression(
    driver, hamiltonian, *, iprint: int = 0
):
    """Build the standard-Pauli MPO in Block2's reversed local basis."""
    builder = driver.expr_builder()
    stream = as_pauli_term_stream(hamiltonian)
    for item in stream.terms:
        if item.mask == (0, 0):
            builder.add_const(float(item.signed_coefficient.real))
            continue
        operators, indices, num_y = _mask_to_block2_word(
            item.mask, stream.n_qubits
        )
        mapped_coefficient = item.signed_coefficient * ((-1j) ** num_y)
        mapped_coefficient = np.real_if_close(mapped_coefficient).item()
        builder.add_term(operators, indices, mapped_coefficient)
    return driver.get_mpo(builder.finalize(), iprint=iprint)


def _build_pauli_mpo_blocked_sum(
    driver,
    hamiltonian,
    n_qubits: int,
    *,
    cutoff: float = 1e-10,
    sum_mpo_mod: int = 20,
    iprint: int = 0,
):
    """Use Block2's blocked sum-of-MPO construction for general Pauli words.

    ``DMRGDriver.get_mpo_any_pauli`` asserts that every word has an even
    number of Y operators because it only implements a real-valued shortcut.
    Clifford conjugation can produce perfectly valid Hermitian words with an
    odd number of Y operators. Build the equivalent symbolic expression
    directly so a CPX Block2 driver can represent those terms, while retaining
    the same memory-bounded blocked-sum MPO algorithm.
    """
    from pyblock2.driver.core import MPOAlgorithmTypes

    builder = driver.expr_builder()
    stream = as_pauli_term_stream(hamiltonian, n_qubits)
    for item in stream.terms:
        if item.mask == (0, 0):
            builder.add_const(float(item.signed_coefficient.real))
            continue
        operators, indices, num_y = _mask_to_block2_word(
            item.mask, stream.n_qubits
        )
        # Block2 Pauli mode stores iY as its real primitive operator.
        mapped_coefficient = item.signed_coefficient * ((-1j) ** num_y)
        mapped_coefficient = np.real_if_close(mapped_coefficient).item()
        builder.add_term(operators, indices, mapped_coefficient)
    return driver.get_mpo(
        builder.finalize(adjust_order=False),
        cutoff=cutoff,
        algo_type=MPOAlgorithmTypes.FastBlockedSumBipartite,
        sum_mpo_mod=int(sum_mpo_mod),
        iprint=iprint,
    )


def build_block2_qubit_mpo(
    driver,
    hamiltonian,
    *,
    n_qubits: int | None = None,
    builder: str = "blocked_sum",
    cutoff: float = 1e-10,
    sum_mpo_mod: int = 20,
    iprint: int = 0,
):
    """Construct a pyblock2 MPO from a packed Pauli Hamiltonian.

    Parameters
    ----------
    driver
        Initialized pyblock2 ``DMRGDriver`` in Pauli mode. Its scalar type
        must be capable of representing the supplied Hamiltonian.
    hamiltonian
        Hermitian ``PauliTermStream`` or OpenFermion ``QubitOperator``.
    n_qubits
        Required number of MPO sites. If omitted, it is taken from the Pauli
        stream. When supplied, it must match the stream's qubit count.
    builder
        ``"blocked_sum"`` for the memory-reduced blocked sum-of-terms route,
        or ``"expression"`` for pyblock2's standard expression route.
    cutoff
        Numerical MPO-compression cutoff used by ``"blocked_sum"``. It is
        ignored by ``"expression"``.
    sum_mpo_mod
        Number of partial MPOs combined per blocked-sum stage.
    iprint
        pyblock2 MPO-construction verbosity.

    Returns
    -------
    mpo
        Constructed pyblock2 MPO owned by ``driver``. The caller must keep the
        driver alive while using or serializing the MPO and must eventually
        call ``driver.finalize()``.
    """
    stream = as_pauli_term_stream(hamiltonian, n_qubits)
    _validate_hermitian_pauli_coefficients(stream)
    if builder == "blocked_sum":
        return _build_pauli_mpo_blocked_sum(
            driver,
            stream,
            stream.n_qubits,
            cutoff=float(cutoff),
            sum_mpo_mod=int(sum_mpo_mod),
            iprint=int(iprint),
        )
    if builder == "expression":
        return _build_pauli_mpo_expression(
            driver,
            stream,
            iprint=int(iprint),
        )
    raise ValueError("builder must be 'blocked_sum' or 'expression'")


# Retained for small tests and explicit comparisons with Block2's native
# expression builder.
_build_pauli_mpo = _build_pauli_mpo_expression


def _infer_mpo_bond_dimension(mpo):
    try:
        from .mpo import infer_largest_mpo_bond_dimension

        return infer_largest_mpo_bond_dimension(mpo, verbose=False)
    except Exception:
        return None


class Block2SweepTimer:
    """Collect wall-clock durations from Block2's native DMRG callbacks."""

    def __init__(self):
        self._sweep_start = None
        self.sweep_seconds = []

    def __call__(self, stage, _verbosity):
        now = perf_counter()
        if stage == "DMRG::sweep.start":
            self._sweep_start = now
        elif stage == "DMRG::sweep.end" and self._sweep_start is not None:
            self.sweep_seconds.append(now - self._sweep_start)
            self._sweep_start = None


def block2_dmrg_sweep_status(
    driver,
    *,
    requested_sweeps: int,
    energy_tolerance: float,
    noises,
    sweep_seconds=None,
) -> dict:
    """Summarize whether Block2 met its sweep-energy stopping criterion."""
    sweep_rows = list(driver._dmrg.energies)
    sweep_energies = []
    for row in sweep_rows:
        values = list(row)
        if values:
            sweep_energies.append(
                float(np.real_if_close(values[0]))
            )
    sweeps_completed = len(sweep_energies)
    last_energy_change = (
        abs(sweep_energies[-1] - sweep_energies[-2])
        if sweeps_completed >= 2
        else None
    )
    final_sweep_index = max(0, sweeps_completed - 1)
    final_noise = (
        float(noises[min(final_sweep_index, len(noises) - 1)])
        if noises
        else 0.0
    )
    energy_criterion_met = (
        last_energy_change is not None
        and last_energy_change < float(energy_tolerance)
        and final_noise == 0.0
    )
    reached_max_sweeps = sweeps_completed >= int(requested_sweeps)
    # Block2's normal early-exit condition is the same energy/noise criterion.
    # Treat an early return as converged as a defensive fallback in case the
    # backend does not retain both final sweep energies in a future version.
    sweep_converged = bool(
        energy_criterion_met
        or (sweeps_completed > 0 and not reached_max_sweeps)
    )
    return {
        "sweep_converged": sweep_converged,
        "reached_max_sweeps": reached_max_sweeps,
        "sweep_limit_reached_without_convergence": bool(
            reached_max_sweeps and not sweep_converged
        ),
        "sweeps_requested": int(requested_sweeps),
        "sweeps_completed": sweeps_completed,
        "sweep_energy_tolerance": float(energy_tolerance),
        "last_sweep_energy_change": last_energy_change,
        "final_sweep_noise": final_noise,
        "sweep_energies": sweep_energies,
        "per_sweep_seconds": (
            [] if sweep_seconds is None else list(sweep_seconds)
        ),
    }


def run_block2_qubit_reference_dmrg(
    *,
    label: str,
    hamiltonian,
    n_qubits: int,
    bond_dim: int,
    sparse_state=None,
    initial_state: str = "cisd",
    dmrg_sweeps: int = 100,
    sweep_tolerance: float = 1e-8,
    mps_cutoff: float = 1e-13,
    mpo_cutoff: float = 1e-10,
    mpo_builder: str = "blocked_sum",
    sum_mpo_mod: int = 20,
    sparse_batch_size: int = 32,
    unitaries=(),
    transform_max_bond: int | None = None,
    transform_cutoff: float = 1e-13,
    davidson_threshold: float = 1e-10,
    noises=(1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6),
    n_threads: int = 4,
    n_mkl_threads: int = 1,
    stack_mem_gb: float = 0.5,
    verbose: bool = False,
    scratch: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Optimize and export a high-accuracy qubit reference MPS.

    Parameters
    ----------
    label
        Human-readable calculation label used in logs and artifact names.
    hamiltonian
        Hermitian ``PauliTermStream`` or OpenFermion ``QubitOperator``.
    n_qubits
        Number of qubit/MPS sites.
    bond_dim
        Fixed maximum MPS bond dimension used for every DMRG sweep.
    sparse_state
        Optional ``(basis_indices, coefficients)`` selected-CI state required
        when ``initial_state="cisd"``.
    initial_state
        ``"cisd"`` for the sparse-state warm start or ``"random"``.
    dmrg_sweeps, sweep_tolerance, davidson_threshold, noises
        Sweep count, sweep-energy tolerance, local eigensolver threshold, and
        per-sweep noise schedule.
    mps_cutoff, sparse_batch_size
        Numerical cutoff and determinant batch size for sparse-CI MPS import.
    mpo_cutoff, mpo_builder, sum_mpo_mod
        Hamiltonian MPO construction cutoff, route, and blocked-sum block size.
    unitaries
        Ordered unitary objects applied to the CISD warm-start MPS.
    transform_max_bond, transform_cutoff
        Optional bond cap and SVD cutoff used only while applying ``unitaries``
        to the warm start.
    n_threads, n_mkl_threads, stack_mem_gb
        Block2 operator/threading count, BLAS/LAPACK thread count, and Block2
        stack-memory allocation in GiB.
    verbose
        Enable Block2 progress output.
    scratch
        Optional persistent Block2 scratch directory; otherwise temporary.
    artifact_dir
        Optional directory in which to save the MPO and final MPS artifacts.

    Returns
    -------
    tensors, summary
        ``tensors`` is the optimized conventional-basis portable MPS. ``summary``
        contains the variational energy, sweep convergence data and timings,
        MPO information, warm-start construction metadata, and artifact paths.

    Notes
    -----
    This is deliberately not a bond-dimension scan. DMRG starts and remains at
    one large ``bond_dim`` while the sweeps converge. The final variational
    energy replaces the FCI reference energy and the returned portable MPS
    replaces the explicit FCI vector in entropy and Fiedler calculations.
    """
    _, _, _, _, DMRGDriver, SymmetryTypes = _require_block2()
    bond_dim = int(bond_dim)
    if bond_dim < 1:
        raise ValueError("reference bond_dim must be positive")
    if initial_state not in {"cisd", "random"}:
        raise ValueError("reference initial_state must be 'cisd' or 'random'")
    if initial_state == "cisd" and sparse_state is None:
        raise ValueError("CISD reference initialization requires sparse_state")
    if dmrg_sweeps < 1:
        raise ValueError("dmrg_sweeps must be positive")
    if int(n_threads) < 1 or int(n_mkl_threads) < 1:
        raise ValueError("n_threads and n_mkl_threads must be positive")
    noises = tuple(float(value) for value in noises)
    if any(value < 0 for value in noises):
        raise ValueError("reference DMRG noises must be nonnegative")
    _validate_hermitian_pauli_coefficients(hamiltonian)

    state_coefficients = None if sparse_state is None else sparse_state[1]
    use_complex = (
        _needs_complex_driver(hamiltonian, state_coefficients)
        or _unitaries_need_complex_driver(unitaries, n_qubits)
    )
    symmetry_type = SymmetryTypes.SGB
    if use_complex:
        symmetry_type |= SymmetryTypes.CPX
    scratch_obj = None
    if scratch is None:
        scratch_obj = tempfile.TemporaryDirectory(prefix="block2_reference_")
        scratch_path = Path(scratch_obj.name)
    else:
        scratch_path = Path(scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)

    driver = DMRGDriver(
        scratch=str(scratch_path),
        symm_type=symmetry_type,
        n_threads=int(n_threads),
        n_mkl_threads=int(n_mkl_threads),
        stack_mem=int(stack_mem_gb * 1024**3),
        min_mpo_mem=True,
        compressed_mps_storage=True,
    )
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)
    try:
        mpo_start = perf_counter()
        if mpo_builder == "blocked_sum":
            mpo = _build_pauli_mpo_blocked_sum(
                driver,
                hamiltonian,
                n_qubits,
                cutoff=mpo_cutoff,
                sum_mpo_mod=sum_mpo_mod,
                iprint=2 if verbose else 0,
            )
        elif mpo_builder == "expression":
            mpo = _build_pauli_mpo_expression(
                driver, hamiltonian, iprint=2 if verbose else 0
            )
        else:
            raise ValueError("unknown reference MPO builder")
        mpo_seconds = perf_counter() - mpo_start
        artifacts = {}
        if artifact_dir is not None:
            artifact_dir = Path(artifact_dir)
            artifacts["mpo"] = save_block2_mpo(
                mpo, artifact_dir / f"{label}_reference_mpo.block2.bin"
            )

        construction = None
        if initial_state == "cisd":
            ket, construction = sparse_state_to_block2_pauli_mps(
                driver,
                sparse_state[0],
                sparse_state[1],
                n_qubits,
                tag=f"REF-{label}",
                batch_size=sparse_batch_size,
                compression_cutoff=mps_cutoff,
                unitaries=unitaries,
                transform_max_bond=(
                    bond_dim
                    if transform_max_bond is None
                    else int(transform_max_bond)
                ),
                transform_cutoff=transform_cutoff,
            )
        else:
            ket = driver.get_random_mps(
                tag=f"REF-{label}", bond_dim=bond_dim, nroots=1
            )

        thresholds = [float(davidson_threshold)] * dmrg_sweeps
        stage_noises = list(noises[:dmrg_sweeps])
        stage_noises += [0.0] * (dmrg_sweeps - len(stage_noises))
        timer = Block2SweepTimer()
        driver.set_callback(timer)
        start = perf_counter()
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=dmrg_sweeps,
            tol=sweep_tolerance,
            bond_dims=[bond_dim] * dmrg_sweeps,
            noises=stage_noises,
            thrds=thresholds,
            dav_max_iter=50,
            iprint=1 if verbose else 0,
        )
        seconds = perf_counter() - start
        energy = float(np.real_if_close(energy))
        status = block2_dmrg_sweep_status(
            driver,
            requested_sweeps=dmrg_sweeps,
            energy_tolerance=sweep_tolerance,
            noises=stage_noises,
            sweep_seconds=timer.sweep_seconds,
        )
        print(
            f"{label} reference bond_dim={bond_dim:4d} "
            f"E={energy:.12f} seconds={seconds:.1f}",
            flush=True,
        )
        if status["sweep_limit_reached_without_convergence"]:
            print(
                f"WARNING: {label} reference bond_dim={bond_dim} "
                "reached its sweep limit without sweep convergence.",
                flush=True,
            )

        tensors = block2_pauli_mps_to_arrays(
            driver, ket, tag=f"EXPORT-{label}"
        )
        if artifact_dir is not None:
            artifacts["mps"] = save_block2_mps(
                ket, artifact_dir / f"{label}_reference_mps.block2"
            )
            artifacts["portable_mps"] = save_qubit_mps_arrays(
                tensors, artifact_dir / f"{label}_reference_mps.npz"
            )
        summary = {
            "method": "pyblock2_qubit_reference_dmrg",
            "hamiltonian_input_representation": (
                "pauli_mask_coefficients"
                if isinstance(hamiltonian, PauliTermStream)
                else "openfermion_qubit_operator"
            ),
            "energy": energy,
            "energy_label": "variational_dmrg_reference",
            "bond_dim": bond_dim,
            "bond_dimension_convergence_tested": False,
            "accuracy_note": (
                "Sweep convergence is tested at one fixed bond dimension; "
                "residual finite-bond variational error is not estimated."
            ),
            "dmrg_seconds": seconds,
            **status,
            "mpo_bond_dimension": _infer_mpo_bond_dimension(mpo),
            "mpo_build_seconds": mpo_seconds,
            "mpo_builder": mpo_builder,
            "mpo_cutoff": mpo_cutoff,
            "initial_state": initial_state,
            "initial_state_construction": construction,
            "warm_start_unitary_components": [
                type(unitary).__name__ for unitary in unitaries
            ],
            "saved_tensor_networks": artifacts,
            "complex_driver": use_complex,
            "n_threads": int(n_threads),
            "n_mkl_threads": int(n_mkl_threads),
        }
        return tensors, summary
    finally:
        driver.finalize()
        if scratch_obj is not None:
            scratch_obj.cleanup()


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
    sweep_tolerance: float = 1e-6,
    mps_cutoff: float = 1e-20,
    mpo_cutoff: float = 1e-10,
    mpo_builder: str = "blocked_sum",
    sum_mpo_mod: int = 20,
    initial_state: str = "cisd",
    sparse_batch_size: int = 32,
    sparse_compression_cutoff: float = 1e-13,
    unitaries=(),
    transform_max_bond: int | None = None,
    transform_cutoff: float = 1e-13,
    validate_warm_start_energy: bool = True,
    full_curve: bool = False,
    n_threads: int = 4,
    n_mkl_threads: int = 1,
    stack_mem_gb: float = 0.5,
    davidson_threshold: float = 1e-10,
    warm_start_noises=(),
    verbose: bool = False,
    scratch: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    bond_result_callback=None,
) -> tuple[list[dict], dict]:
    """Benchmark Pauli-mode Block2 DMRG with a selected-CI warm start.

    Parameters
    ----------
    bond_result_callback
        Optional callable invoked with a copy of each completed per-bond row.
        It can persist incremental checkpoints before the full curve returns.

    A fresh copy of the imported warm-start MPS is used at every tested bond
    dimension. Random initialization remains available for control runs.

    Returns
    -------
    rows, summary
        Per-bond DMRG measurements and a curve-level convergence/MPO summary.
    """
    _, _, _, _, DMRGDriver, SymmetryTypes = _require_block2()
    if initial_state not in {"cisd", "exact_fci", "random"}:
        raise ValueError(
            "initial_state must be 'cisd', 'exact_fci', or 'random'"
        )
    if dmrg_sweeps < 1:
        raise ValueError("dmrg_sweeps must be positive")
    if int(n_threads) < 1 or int(n_mkl_threads) < 1:
        raise ValueError("n_threads and n_mkl_threads must be positive")
    warm_start_noises = tuple(float(value) for value in warm_start_noises)
    if any(value < 0 for value in warm_start_noises):
        raise ValueError("warm_start_noises must be nonnegative")
    if mpo_builder not in {"blocked_sum", "expression"}:
        raise ValueError(
            "mpo_builder must be 'blocked_sum' or 'expression'"
        )
    _validate_hermitian_pauli_coefficients(hamiltonian)

    sparse_coefficients = None if sparse_state is None else sparse_state[1]
    state_for_dtype = (
        exact_state if exact_state is not None else sparse_coefficients
    )
    complex_hamiltonian_or_state = _needs_complex_driver(
        hamiltonian, state_for_dtype
    )
    complex_unitary_circuit = _unitaries_need_complex_driver(
        unitaries, n_qubits
    )
    use_complex = complex_hamiltonian_or_state or complex_unitary_circuit
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

    mpo_start = perf_counter()

    driver = DMRGDriver(
        scratch=str(scratch_path),
        symm_type=symmetry_type,
        n_threads=int(n_threads),
        n_mkl_threads=int(n_mkl_threads),
        stack_mem=int(stack_mem_gb * 1024**3),
        min_mpo_mem=True,
        compressed_mps_storage=True,
    )
    print(
        f"{label}: creating Block2 driver with stack_mem="
        f"{stack_mem_gb:.3f} GiB.{_rss_message()}",
        flush=True,
    )
    driver.initialize_system(n_sites=n_qubits, pauli_mode=True)

    try:
        print(
            f"{label}: completing "
            f"{len(as_pauli_term_stream(hamiltonian, n_qubits).terms)}-term Pauli MPO "
            f"with {mpo_builder} builder."
            f"{_rss_message()}",
            flush=True,
        )
        if mpo_builder == "blocked_sum":
            mpo_start = perf_counter()
            mpo = _build_pauli_mpo_blocked_sum(
                driver,
                hamiltonian,
                n_qubits,
                cutoff=mpo_cutoff,
                sum_mpo_mod=sum_mpo_mod,
                iprint=2 if verbose else 0,
            )
        else:
            mpo_start = perf_counter()
            mpo = _build_pauli_mpo_expression(
                driver, hamiltonian, iprint=2 if verbose else 0
            )
        mpo_seconds = perf_counter() - mpo_start
        print(
            f"{label}: Pauli MPO built in {mpo_seconds:.1f} s."
            f"{_rss_message()}",
            flush=True,
        )
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
                print(
                    f"{label}: constructing warm-start MPS from "
                    f"{len(sparse_state[0])} determinants."
                    f"{_rss_message()}",
                    flush=True,
                )
                warm_mps, construction = sparse_state_to_block2_pauli_mps(
                    driver,
                    sparse_state[0],
                    sparse_state[1],
                    n_qubits,
                    tag=f"WARM-{label}",
                    batch_size=sparse_batch_size,
                    compression_cutoff=sparse_compression_cutoff,
                    unitaries=unitaries,
                    transform_max_bond=transform_max_bond,
                    transform_cutoff=transform_cutoff,
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
            print(
                f"{label}: warm-start MPS constructed and contracted."
                f"{_rss_message()}",
                flush=True,
            )
            imported_energy = float(np.real_if_close(imported_energy))
            expected_warm_energy = (
                exact_energy
                if warm_start_energy is None
                else float(warm_start_energy)
            )
            warm_energy_delta = imported_energy - expected_warm_energy
            tensor_circuit = (
                construction.get("tensor_circuit", {})
                if construction is not None
                else {}
            )
            circuit_discarded = float(
                tensor_circuit.get(
                    "root_sum_squared_discarded_singular_values", 0.0
                )
            )
            effective_energy_validation = (
                validate_warm_start_energy and circuit_discarded <= 1e-12
            )
            print(
                f"{label}: imported warm-start E={imported_energy:.12f}, "
                f"delta from untruncated reference={warm_energy_delta:+.3e}.",
                flush=True,
            )
            if (
                effective_energy_validation
                and abs(imported_energy - expected_warm_energy) > 1e-7
            ):
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
                noises = list(warm_start_noises[:dmrg_sweeps])
                noises += [0.0] * (dmrg_sweeps - len(noises))
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
            sweep_timer = Block2SweepTimer()
            driver.set_callback(sweep_timer)
            energy = driver.dmrg(
                mpo,
                ket,
                n_sweeps=dmrg_sweeps,
                tol=sweep_tolerance,
                bond_dims=[bond_dim] * dmrg_sweeps,
                noises=noises,
                thrds=thresholds,
                dav_max_iter=50,
                iprint=1 if verbose else 0,
            )
            seconds = perf_counter() - start
            sweep_status = block2_dmrg_sweep_status(
                driver,
                requested_sweeps=dmrg_sweeps,
                energy_tolerance=sweep_tolerance,
                noises=noises,
                sweep_seconds=sweep_timer.sweep_seconds,
            )
            energy = float(np.real_if_close(energy))
            error = abs(energy - exact_energy)
            converged = error <= dmrg_tolerance
            if converged and first_converged_bd is None:
                first_converged_bd = bond_dim
                if artifact_dir is not None:
                    artifacts["first_chemically_accurate_mps"] = (
                        save_block2_mps(
                            ket,
                            artifact_dir
                            / (
                                f"{label}_first_chemical_accuracy_"
                                f"bd{bond_dim}_mps.block2"
                            ),
                        )
                    )

            row = {
                "frame": label,
                "backend": "block2_pauli",
                "bond_dim": bond_dim,
                "energy": energy,
                "abs_energy_error": error,
                "within_dmrg_tolerance": converged,
                "dmrg_seconds": seconds,
                "max_result_mps_bond": bond_dim,
                **sweep_status,
            }
            rows.append(row)
            if bond_result_callback is not None:
                bond_result_callback(dict(row))
            print(
                f"{label:26s} bond_dim={bond_dim:3d} "
                f"E={energy:.12f} |dE|={error:.3e} "
                f"seconds={seconds:.1f}",
                flush=True,
            )
            if sweep_status["sweep_limit_reached_without_convergence"]:
                delta = sweep_status["last_sweep_energy_change"]
                delta_text = "unavailable" if delta is None else f"{delta:.3e}"
                print(
                    f"WARNING: {label} bond_dim={bond_dim} exhausted "
                    f"{dmrg_sweeps} sweeps without sweep-energy convergence; "
                    f"last |delta E_sweep|={delta_text}, "
                    f"tolerance={sweep_tolerance:.3e}.",
                    flush=True,
                )
            if converged and not full_curve:
                print(
                    f"{label}: reached chemical accuracy at "
                    f"bond_dim={bond_dim}; stopping this frame.",
                    flush=True,
                )
                break

        first_converged_row = next(
            (
                row
                for row in rows
                if row["within_dmrg_tolerance"]
            ),
            None,
        )
        summary = {
            "frame": label,
            "backend": "block2_pauli",
            "hamiltonian_input_representation": (
                "pauli_mask_coefficients"
                if isinstance(hamiltonian, PauliTermStream)
                else "openfermion_qubit_operator"
            ),
            "first_converged_bond_dim": first_converged_bd,
            "converged_within_grid": first_converged_bd is not None,
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
            "mpo_bond_dimension": mpo_bond_dimension,
            "mpo_build_seconds": mpo_seconds,
            "mpo_builder": mpo_builder,
            "mpo_cutoff": mpo_cutoff,
            "sum_mpo_mod": sum_mpo_mod,
            "warm_start": initial_state != "random",
            "warm_start_kind": initial_state,
            "imported_warm_start_energy": imported_energy,
            "expected_warm_start_energy": warm_start_energy,
            "warm_start_energy_difference": (
                None
                if imported_energy is None or warm_start_energy is None
                else imported_energy - warm_start_energy
            ),
            "warm_start_energy_validation_requested": (
                validate_warm_start_energy
            ),
            "warm_start_energy_validation_performed": (
                effective_energy_validation
                if initial_state != "random"
                else False
            ),
            "warm_start_construction": construction,
            "saved_tensor_networks": artifacts,
            "complex_driver": use_complex,
            "complex_driver_from_hamiltonian_or_input_state": (
                complex_hamiltonian_or_state
            ),
            "complex_driver_from_unitary_circuit": complex_unitary_circuit,
            "n_threads": int(n_threads),
            "n_mkl_threads": int(n_mkl_threads),
            "davidson_threshold": davidson_threshold,
            "warm_start_noises": list(warm_start_noises),
            "sweep_energy_tolerance": sweep_tolerance,
            "bond_dims_without_sweep_convergence": [
                row["bond_dim"]
                for row in rows
                if row["sweep_limit_reached_without_convergence"]
            ],
        }
        return rows, summary
    finally:
        # Block2 stores its stack arena in a process-global frame. Explicitly
        # release it before starting the next benchmark frame.
        driver.finalize()
        if scratch_obj is not None:
            scratch_obj.cleanup()
