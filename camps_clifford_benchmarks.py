"""Run CAMPS Clifford-disentangling benchmarks for the 13 saved systems.

This script is intentionally a thin bridge between the external CAMPS
implementation in FOCUS and the local benchmark machinery used by the JUL08
Clifford/Fiedler scripts.

Workflow
--------
For each saved Hamiltonian, the script:

1. converts the chosen saved reference state, by default the FCI state, to a
   CAMPS-compatible MPS;
2. runs CAMPS entropy minimization on that MPS to select local Clifford gates;
3. converts the OpenFermion ``QubitOperator`` Hamiltonian to the CAMPS Pauli
   array format and applies the selected Clifford sequence;
4. converts the transformed Hamiltonian back to an OpenFermion
   ``QubitOperator``;
5. inverse-transforms each final-frame ``Z_i`` by the selected CAMPS Clifford
   sequence to obtain the equivalent original-frame Pauli generators;
6. benchmarks the CAMPS Hamiltonian and its Fiedler-reordered version using the
   same local quimb MPO/DMRG settings as ``july08_clifford_benchmarks.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep native linear algebra conservative.  The 4Q CAMPS path calls many small
# Torch/SciPy/BLAS SVDs; on macOS/Python-3.13 this is much more stable with
# one native thread.  FOCUS' own example also sets torch.set_num_threads(1).
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np

# The cloned TenNet/Python-3.13 stack can make numba fail while caching quimb's
# import-time kernels.  This is harmless to disable and keeps the benchmark
# script runnable in the CAMPS environment.
os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-camps")
import quimb.tensor as qtn
from openfermion import (
    QubitOperator,
    count_qubits,
    get_ground_state,
    get_sparse_operator,
    jordan_wigner,
)

from quasisymmetries.benchmark import BenchmarkData
from quasisymmetries.clifford_symmetry_optimized import (
    permute_qubits_in_qubit_operator,
)
from quasisymmetries.fiedler import do_fiedler_reordering
from quasisymmetries.metrics import get_entropies_at_cuts


HAM_DIR = Path("./saved/hamiltonians")
DEFAULT_OUTPUT_DIR = Path("./saved/results/JUL17_CAMPS")
LOG_BASE = np.e

SYSTEMS = [
    "H2O_corr",]
#     "H4chain_corr",
#     "H4chain_diss",
#     "H4rect_corr",
#     "H4rect_diss",
#     "LIH_eqm",
#     "LIH_corr",
#     "H2O_eqm",
#     "H2O_corr",
#     "H2O_diss",
#     "N2frozen_eqm",
#     "N2frozen_corr",
#     "N2frozen_diss",
# ]


@dataclass
class CampsModules:
    dtype_config: Any
    batch_svd_config: Any
    Clifford: Any
    Hamiltonian: Any
    random_clifford: Any
    minimize_entropy_multiSweep: Any
    update_ham_multiSweep: Any
    mapping_inverse: Any
    pauli_transform: Any
    pauli_to_vector: Any
    vector_to_pauli: Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return as_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return as_jsonable(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, QubitOperator):
        return str(value)
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(val) for val in value]
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        json.dump(as_jsonable(data), file_obj, indent=2)
        file_obj.write("\n")


def sym_strings(symmetries: list[QubitOperator]) -> list[str]:
    return [str(sym) for sym in symmetries]


def write_entropies(file_obj: Any, label: str, entropies: list[float]) -> None:
    print(label, file=file_obj)
    for i, ent in enumerate(entropies):
        print(f"  {i + 1}|{i + 2}: {ent}", file=file_obj)


def get_coeffs_and_ops(of_op: QubitOperator, n_qubits: int):
    identity = np.array([[1, 0], [0, 1]], dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {"X": x, "Y": y, "Z": z}

    coeffs = []
    ops_list = []
    for term, coeff in of_op.terms.items():
        ops = [identity.copy() for _ in range(n_qubits)]
        for qubit, pauli in term:
            ops[qubit] = pauli_map[pauli]
        coeffs.append(coeff)
        ops_list.append(ops)
    return coeffs, ops_list


def MPO_from_QubitOperator(
    H: QubitOperator,
    max_bond=None,
    mpo_cutoff=1e-10,
    verbose=True,
    compression_freq=20,
):
    """Quimb-only QubitOperator -> MPO helper, matching JUL08 settings."""
    n_qubits = count_qubits(H)
    zero2 = np.zeros((2, 2), dtype=float)
    mpo = qtn.MPO_product_operator([zero2] * n_qubits)

    coeffs, ops = get_coeffs_and_ops(H, n_qubits)
    for i, (coeff, op) in enumerate(zip(coeffs, ops)):
        mpo += coeff * qtn.MPO_product_operator(op)
        if mpo_cutoff is not None and i % compression_freq == 0:
            mpo.compress(max_bond=max_bond, cutoff=mpo_cutoff)

    if mpo_cutoff is not None:
        mpo.compress(max_bond=max_bond, cutoff=mpo_cutoff)

    if verbose:
        print(f"Bond dimensions of MPO: {mpo.bond_sizes()}")
    return mpo


def find_dmrg_conv_bd_quimb(
    Hq: QubitOperator,
    n_qubits: int,
    exact_energy: float,
    bd_list=None,
    tol=1.6e-3,
    n_sweeps=10,
    reps=1,
    verbose=False,
    compress_cutoff=1e-10,
    sweep_tol=1e-6,
    noise=1e-3,
    bsz=2,
    guess_mps=None,
    seed=None,
    return_data=False,
):
    """Quimb DMRG convergence helper, matching JUL08 settings."""
    mpo = MPO_from_QubitOperator(
        Hq,
        max_bond=None,
        mpo_cutoff=compress_cutoff,
        verbose=verbose,
        compression_freq=20,
    )
    verbosity = 2 if verbose else 0

    if seed is not None:
        np.random.seed(seed)

    if bd_list is None:
        bd_list = (
            [i for i in range(1, 11, 1)]
            + [i for i in range(12, 21, 2)]
            + [i for i in range(30, 101, 10)]
        )

    last_energy = None
    for bd in bd_list:
        if verbose:
            print(f"Starting max bd = {bd}")
        for _ in range(reps):
            if guess_mps is None:
                guess_mps = qtn.MPS_rand_state(n_qubits, 1)
            else:
                guess_mps += noise * qtn.MPS_rand_state(n_qubits, bond_dim=1)
                guess_mps.normalize()

            dmrg = qtn.DMRG(mpo, bd, bsz=bsz, cutoffs=compress_cutoff, p0=guess_mps)
            dmrg.opts["local_eig_tol"] = 1e-3
            dmrg.opts["pempsriodic_compress_ham_eps"] = compress_cutoff
            dmrg.opts["periodic_compress_norm_eps"] = compress_cutoff
            dmrg.solve(
                tol=sweep_tol,
                bond_dims=bd,
                max_sweeps=n_sweeps,
                sweep_sequence="RL",
                verbosity=verbosity,
                suppress_warnings=False,
                cutoffs=compress_cutoff,
            )
            last_energy = dmrg.energy
            if abs(dmrg.energy - exact_energy) <= tol:
                print(f"DMRG converged at bond dimension: {bd}")
                if return_data:
                    print("Returning MPO...")
                    return bd, dmrg.energy, {"mpo": mpo}
                return bd, dmrg.energy

    print(f"DMRG not converged at bd = {bd_list[-1]}")
    if return_data:
        print("Returning MPO...")
        return bd_list[-1], last_energy, {"mpo": mpo}
    return bd_list[-1], last_energy


def load_camps_modules(pyfocus_path: str | None = None) -> CampsModules:
    if pyfocus_path is not None:
        sys.path.insert(0, str(Path(pyfocus_path).expanduser().resolve()))

    try:
        from pyfocus.camps.mps.disentangled import (
            minimize_entropy_multiSweep,
            update_ham_multiSweep,
        )
        from pyfocus.camps.utils.clifford import random_clifford
        from pyfocus.camps.utils.config import batch_svd_config, dtype_config
        from pyfocus.camps.utils.pauli_alg import (
            mapping_inverse,
            pauli_to_vector,
            pauli_transform,
            vector_to_pauli,
        )
        from pyfocus.camps.utils.typing import Clifford, Hamiltonian
    except Exception as exc:
        raise ImportError(
            "Could not import CAMPS from pyfocus. Install FOCUS/pyfocus, or pass "
            "--pyfocus-path pointing at the FOCUS repository root. The underlying "
            f"import failure was: {type(exc).__name__}: {exc}"
        ) from exc

    return CampsModules(
        dtype_config=dtype_config,
        batch_svd_config=batch_svd_config,
        Clifford=Clifford,
        Hamiltonian=Hamiltonian,
        random_clifford=random_clifford,
        minimize_entropy_multiSweep=minimize_entropy_multiSweep,
        update_ham_multiSweep=update_ham_multiSweep,
        mapping_inverse=mapping_inverse,
        pauli_transform=pauli_transform,
        pauli_to_vector=pauli_to_vector,
        vector_to_pauli=vector_to_pauli,
    )


def configure_camps(args: argparse.Namespace, camps: CampsModules) -> None:
    import torch

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You requested --device cuda, but this PyTorch build has no CUDA "
                "support. Use --device cpu in the camps environment on this Mac."
            )

    dtype_kwargs = {
        "use_float64": True,
        "use_complex": True,
        "device": args.device,
        "driver": args.svd_driver,
        "batch": args.svd_batch,
        "direct_limit": args.svd_direct_limit,
        "scratch_dir": args.scratch_dir,
    }
    camps.dtype_config.apply(**dtype_kwargs)
    batch_svd_kwargs = dict(dtype_kwargs)
    # CAMPS' batched SVD path stores singular values. These should be real even
    # when the MPS/gates are complex; the upstream example config does this too.
    # If this is left complex, the Rényi/vN entropy selection can hit PyTorch
    # complex comparison/argmin failures on CPU.
    batch_svd_kwargs["use_complex"] = False
    camps.batch_svd_config.apply(**batch_svd_kwargs)

    if args.batch_svd_batch is not None or args.batch_svd_direct_limit is not None:
        camps.batch_svd_config.apply(
            use_float64=True,
            use_complex=False,
            device=args.device,
            driver=args.svd_driver,
            batch=args.svd_batch if args.batch_svd_batch is None else args.batch_svd_batch,
            direct_limit=(
                args.svd_direct_limit
                if args.batch_svd_direct_limit is None
                else args.batch_svd_direct_limit
            ),
            scratch_dir=args.scratch_dir,
        )


def make_camps_n_sites(n_qubits: int, n_dim: int) -> int:
    if n_dim == 2:
        return n_qubits
    if n_dim == 4:
        if n_qubits % 2:
            raise ValueError("CAMPS n_dim=4 requires an even number of qubits")
        return n_qubits // 2
    raise ValueError("Only CAMPS n_dim=2 and n_dim=4 are supported here")


def load_candidate_clifford(args: argparse.Namespace, camps: CampsModules):
    if args.camps_clifford_npz is not None:
        data = np.load(args.camps_clifford_npz, allow_pickle=True)
        gates = np.asarray(data["gates"])
        mapping = np.asarray(data["mapping"], dtype=np.int64)
        phases = np.asarray(
            data["phases"] if "phases" in data else np.zeros_like(mapping),
            dtype=np.int64,
        )
        endian = str(data["endian"].item()) if "endian" in data else args.endian
        return camps.Clifford(
            gates=gates,
            mapping=mapping,
            phases=phases,
            endian=endian,
        )

    return camps.random_clifford(
        nums=args.candidate_gates,
        n_qubits=args.n_dim,
        seed=args.seed,
        add_I=True,
        endian=args.endian,
    )


def qubit_operator_to_camps_hamiltonian(
    operator: QubitOperator,
    n_qubits: int,
    Hamiltonian,
):
    rows: list[np.ndarray] = []
    coeffs: list[complex] = []
    for term, coeff in operator.terms.items():
        row = np.full(n_qubits, b"I", dtype="S1")
        for qubit, pauli in term:
            row[qubit] = pauli.encode("ascii")
        rows.append(row)
        coeffs.append(complex(coeff))

    if not rows:
        rows = [np.full(n_qubits, b"I", dtype="S1")]
        coeffs = [0.0]

    return Hamiltonian(array=np.stack(rows), coeff=np.asarray(coeffs, dtype=complex))


def row_to_pauli_string(row: np.ndarray) -> str:
    arr = np.asarray(row)
    if arr.dtype.kind == "S":
        return b"".join(arr.astype("S1").reshape(-1).tolist()).decode("ascii")
    return "".join(str(x) for x in arr.reshape(-1).tolist())


def camps_hamiltonian_to_qubit_operator(hamiltonian: Any) -> QubitOperator:
    array, coeff = camps_hamiltonian_parts(hamiltonian)
    return camps_parts_to_qubit_operator(array, coeff)


def camps_hamiltonian_parts(hamiltonian: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(array, coeff)`` for either CAMPS dict or attribute objects."""
    if hasattr(hamiltonian, "array"):
        return np.asarray(hamiltonian.array), np.asarray(hamiltonian.coeff)
    return np.asarray(hamiltonian["array"]), np.asarray(hamiltonian["coeff"])


def camps_parts_to_qubit_operator(array: np.ndarray, coeff: np.ndarray) -> QubitOperator:
    out = QubitOperator()
    for row, c in zip(array, coeff):
        pauli_string = row_to_pauli_string(row)
        term = tuple(
            (i, pauli)
            for i, pauli in enumerate(pauli_string)
            if pauli != "I"
        )
        out += QubitOperator(term, complex(c))
    return out


def z_rows(n_qubits: int) -> np.ndarray:
    rows = np.full((n_qubits, n_qubits), b"I", dtype="S1")
    for i in range(n_qubits):
        rows[i, i] = b"Z"
    return rows


def pauli_rows_to_qubit_operators(
    rows: np.ndarray,
    coeffs: np.ndarray | None = None,
) -> list[QubitOperator]:
    if coeffs is None:
        coeffs = np.ones(len(rows), dtype=complex)
    out = []
    for row, coeff in zip(rows, coeffs):
        pauli_string = row_to_pauli_string(row)
        term = tuple(
            (i, pauli)
            for i, pauli in enumerate(pauli_string)
            if pauli != "I"
        )
        out.append(QubitOperator(term, complex(coeff)))
    return out


def _local_reorder(n_dim: int) -> list[int]:
    # Reproduce CAMPS' site-to-spin-orbital ordering convention in
    # update_ham_singleSweep.
    if n_dim == 2:
        return [0, 2, 1, 3]
    if n_dim == 4:
        return [0, 4, 1, 5, 2, 6, 3, 7]
    raise ValueError("Only n_dim=2 and n_dim=4 are supported")


def _phase_to_sign(phases: np.ndarray) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.int64)
    if np.any((phases % 2) != 0):
        raise ValueError(
            "Encountered a CAMPS Pauli phase with odd exponent. This bridge "
            "expects real Pauli operators with phases 0 or 2."
        )
    return np.where((phases % 4) == 0, 1.0, -1.0)


def _apply_inverse_local_clifford(
    rows: np.ndarray,
    coeffs: np.ndarray,
    *,
    start: int,
    stop: int,
    mapping_index: int,
    phase_index: int | None = None,
    clifford: Any,
    camps: CampsModules,
    new_order: list[int],
) -> None:
    if phase_index is None:
        phase_index = mapping_index
    mapping = np.asarray(clifford.mapping[mapping_index])[new_order]
    phases = np.asarray(clifford.phases[phase_index])[new_order]
    inv_mapping, inv_phases = camps.mapping_inverse(mapping, phases)

    local_vectors = camps.pauli_to_vector(rows[:, start:stop])
    transformed, output_phases = camps.pauli_transform(
        local_vectors,
        gs_map=inv_mapping,
        ps_map=inv_phases,
    )
    rows[:, start:stop] = camps.vector_to_pauli(transformed)
    coeffs *= _phase_to_sign(output_phases)


def inverse_transform_pauli_rows_by_camps(
    rows: np.ndarray,
    *,
    idx: np.ndarray,
    clifford: Any,
    n_sites: int,
    n_dim: int,
    camps: CampsModules,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the inverse of the selected CAMPS Clifford sequence to Pauli rows.

    CAMPS applies one forward nearest-neighbor sweep and one backward sweep per
    macro sweep.  This routine applies the inverse gates in exact reverse order.
    It is used to compute the original-frame generators corresponding to
    final-frame ``Z_i``.
    """
    rows = np.array(rows, dtype="S1", copy=True)
    coeffs = np.ones(rows.shape[0], dtype=complex)
    idx = np.asarray(idx, dtype=np.int64)
    new_order = _local_reorder(n_dim)
    offset = rows.shape[1] // n_sites
    if offset not in (1, 2):
        raise ValueError(
            f"Unexpected CAMPS site/qubit offset {offset}; check n_sites/n_dim."
        )

    if idx.ndim == 2:
        idx = idx[np.newaxis, ...]
    if idx.shape[1] != 2:
        raise ValueError(f"Expected selected indices with shape (sweep, 2, bonds), got {idx.shape}")

    for macro in range(idx.shape[0] - 1, -1, -1):
        macro_clifford = clifford[macro] if isinstance(clifford, list) else clifford
        # Invert the CAMPS backward sweep first.
        for i in range(0, n_sites - 1):
            _apply_inverse_local_clifford(
                rows,
                coeffs,
                start=i * offset,
                stop=(i + 2) * offset,
                mapping_index=int(idx[macro, 1, i]),
                # CAMPS update_ham_singleSweep currently uses the forward-sweep
                # phase index also during the backward Hamiltonian update.
                # Mirror that behavior so this inverse matches the Hamiltonian
                # actually produced by CAMPS.
                phase_index=int(idx[macro, 0, i]),
                clifford=macro_clifford,
                camps=camps,
                new_order=new_order,
            )
        # Then invert the CAMPS forward sweep.
        for i in range(n_sites - 2, -1, -1):
            _apply_inverse_local_clifford(
                rows,
                coeffs,
                start=i * offset,
                stop=(i + 2) * offset,
                mapping_index=int(idx[macro, 0, i]),
                phase_index=int(idx[macro, 0, i]),
                clifford=macro_clifford,
                camps=camps,
                new_order=new_order,
            )

    return rows, coeffs


def statevector_to_camps_sites(
    state: np.ndarray,
    *,
    n_sites: int,
    phys_dim: int,
) -> list[np.ndarray]:
    """Convert a dense state vector to CAMPS' list-of-site-tensors MPS form."""
    state = np.asarray(state, dtype=complex)
    if state.size != phys_dim**n_sites:
        raise ValueError(
            f"State dimension {state.size} does not equal {phys_dim}^{n_sites}."
        )

    psi = state.reshape((phys_dim,) * n_sites)
    sites: list[np.ndarray] = []
    left_dim = 1
    for site in range(n_sites - 1):
        psi = psi.reshape(left_dim * phys_dim, -1)
        u, s, vh = np.linalg.svd(psi, full_matrices=False)
        bond_dim = len(s)
        sites.append(u.reshape(left_dim, phys_dim, bond_dim))
        psi = np.diag(s) @ vh
        left_dim = bond_dim
    sites.append(psi.reshape(left_dim, phys_dim, 1))
    return sites


def run_camps_for_system(
    *,
    HQ: QubitOperator,
    reference_state: np.ndarray,
    n_qubits: int,
    args: argparse.Namespace,
    camps: CampsModules,
    candidate_clifford: Any,
):
    n_sites = make_camps_n_sites(n_qubits, args.n_dim)
    reference_sites = statevector_to_camps_sites(
        reference_state,
        n_sites=n_sites,
        phys_dim=args.n_dim,
    )
    hamiltonian = qubit_operator_to_camps_hamiltonian(HQ, n_qubits, camps.Hamiltonian)
    ham_array, ham_coeff = camps_hamiltonian_parts(hamiltonian)

    disentangled_sites, idx_torch, entropy_change, random_lst = camps.minimize_entropy_multiSweep(
        sites=reference_sites,
        dmax=args.camps_dmax,
        clifford=candidate_clifford,
        microiter=args.microiter,
        iroot=0,
        alpha=args.alpha,
        n_dim=args.n_dim,
        use_random_gates=args.use_random_gates,
        random_nums=args.random_nums,
        random_endian=args.endian,
        given_clifford=None,
        save_mode="normal",
    )
    if hasattr(idx_torch, "detach"):
        idx = idx_torch.detach().cpu().numpy().astype(np.int64)
    else:
        idx = np.asarray(idx_torch, dtype=np.int64)
    active_clifford = random_lst if args.use_random_gates else candidate_clifford

    transformed_array, signs = camps.update_ham_multiSweep(
        ham_array,
        idx,
        active_clifford,
        n_sites=n_sites,
        in_place=False,
        n_dim=args.n_dim,
    )
    H_camps = camps_parts_to_qubit_operator(
        transformed_array,
        ham_coeff * np.asarray(signs),
    )
    sym_rows, sym_coeffs = inverse_transform_pauli_rows_by_camps(
        z_rows(n_qubits),
        idx=idx,
        clifford=active_clifford,
        n_sites=n_sites,
        n_dim=args.n_dim,
        camps=camps,
    )
    equivalent_symmetries = pauli_rows_to_qubit_operators(sym_rows, sym_coeffs)

    return {
        "disentangled_sites": disentangled_sites,
        "H_camps": H_camps,
        "entropy_change": entropy_change,
        "selected_idx": idx,
        "n_sites": n_sites,
        "equivalent_symmetries": equivalent_symmetries,
    }


def exact_ground_state_for_qubit_hamiltonian(
    Hq: QubitOperator,
    n_qubits: int,
) -> tuple[float, np.ndarray]:
    sparse = get_sparse_operator(Hq, n_qubits)
    energy, state = get_ground_state(sparse)
    return float(np.real_if_close(energy)), state


def fiedler_working_data(
    H_camps: QubitOperator,
    fci_camps: np.ndarray,
    n_qubits: int,
):
    ent_reord, H_reord, state_reord, fiedler_info = do_fiedler_reordering(
        H_camps,
        fci_camps,
        n_qubits=n_qubits,
        verbose=False,
        log_base=LOG_BASE,
    )
    return {
        "H": H_reord,
        "fci_state": state_reord,
        "entropies": ent_reord,
        "info": fiedler_info,
    }


def make_benchmark_dataset(
    tag: str,
    symmetries: list[QubitOperator],
    cut_entropies: list[float],
    dmrg_bd: int | str,
) -> BenchmarkData:
    return BenchmarkData(
        tag=tag,
        symmetries=symmetries,
        cut_entropies=list(cut_entropies),
        dmrg_bd=0 if dmrg_bd == "" else int(dmrg_bd),
        clifford_synthesis_basis="CAMPS",
        clifford_generator_mapping="inverse_Z",
    )


def run_local_benchmarks(
    *,
    system: str,
    label: str,
    Hq: QubitOperator,
    fci_state: np.ndarray,
    fci_energy: float,
    n_qubits: int,
    skip_dmrg: bool,
    skip_mpo: bool,
    dmrg_rows: list[dict[str, Any]],
    mpo_rows: list[dict[str, Any]],
):
    if skip_dmrg:
        dmrg_bd = ""
        dmrg_energy = ""
    else:
        guess_mps = qtn.MatrixProductState.from_dense(fci_state, cutoff=1e-20)
        dmrg_bd, dmrg_energy, _dmrg_data = find_dmrg_conv_bd_quimb(
            Hq,
            n_qubits,
            fci_energy,
            tol=1.6e-3,
            n_sweeps=100,
            reps=1,
            verbose=False,
            compress_cutoff=1e-20,
            sweep_tol=1e-6,
            noise=1e0,
            bsz=2,
            guess_mps=guess_mps,
            seed=0,
            return_data=True,
        )

    dmrg_row = {
        "system": system,
        "benchmark": label,
        "n_qubits": n_qubits,
        "dmrg_bd": dmrg_bd,
        "dmrg_energy": dmrg_energy,
    }
    dmrg_rows.append(dmrg_row)

    if skip_mpo:
        mpo_bd = ""
    else:
        mpo = MPO_from_QubitOperator(
            Hq,
            None,
            mpo_cutoff=1e-20,
            compression_freq=20,
            verbose=False,
        )
        mpo_bd = max(mpo.bond_sizes())

    mpo_row = {
        "system": system,
        "benchmark": label,
        "n_qubits": n_qubits,
        "mpo_bd": mpo_bd,
    }
    mpo_rows.append(mpo_row)
    return dmrg_row, mpo_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=SYSTEMS)
    parser.add_argument("--ham-dir", type=Path, default=HAM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pyfocus-path", default=None)
    parser.add_argument("--reference-state", default="fci", choices=["fci", "cisd"])
    parser.add_argument("--skip-dmrg", action="store_true")
    parser.add_argument("--skip-mpo", action="store_true")
    parser.add_argument("--skip-fiedler", action="store_true")

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--scratch-dir", default="./scratch_camps")
    parser.add_argument("--svd-driver", default="gesvd")
    parser.add_argument("--svd-batch", type=int, default=-1)
    parser.add_argument("--svd-direct-limit", type=int, default=300)
    parser.add_argument("--batch-svd-batch", type=int, default=None)
    parser.add_argument("--batch-svd-direct-limit", type=int, default=None)

    parser.add_argument("--n-dim", type=int, default=2, choices=[2, 4])
    parser.add_argument("--use-orb", action="store_true")
    parser.add_argument("--microiter", type=int, default=5)
    parser.add_argument("--camps-dmax", type=int, default=100)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Rényi entropy parameter for CAMPS gate selection. Use alpha != 1; "
            "CAMPS' batched candidate routine does not support alpha=1."
        ),
    )
    parser.add_argument("--candidate-gates", type=int, default=200)
    parser.add_argument("--random-nums", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--endian", default="big", choices=["big", "little"])
    parser.add_argument("--use-random-gates", action="store_true")
    parser.add_argument(
        "--camps-clifford-npz",
        type=Path,
        default=None,
        help="Optional npz with CAMPS Clifford arrays: gates, mapping, phases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.alpha == 1:
        raise ValueError(
            "CAMPS' batched Clifford selection path does not support alpha=1 "
            "because its von Neumann entropy helper is not vectorized over "
            "candidate gates. Use a Rényi value such as --alpha 0.5."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_file = args.output_dir / "camps_clifford_benchmarks.txt"
    json_file = args.output_dir / "camps_clifford_benchmarks.json"
    dmrg_csv = args.output_dir / "camps_dmrg_bond_dimensions.csv"
    mpo_csv = args.output_dir / "camps_mpo_bond_dimensions.csv"

    camps = load_camps_modules(args.pyfocus_path)
    configure_camps(args, camps)
    candidate_clifford = load_candidate_clifford(args, camps)

    dmrg_rows: list[dict[str, Any]] = []
    mpo_rows: list[dict[str, Any]] = []
    all_results: dict[str, Any] = {
        "method": "CAMPS",
        "systems": {},
        "settings": {
            "n_dim": args.n_dim,
            "use_orb": args.use_orb,
            "microiter": args.microiter,
            "camps_dmax": args.camps_dmax,
            "candidate_gates": args.candidate_gates,
            "seed": args.seed,
            "reference_state": args.reference_state,
            "fiedler_reference_state": "exact ground state of CAMPS-transformed Hamiltonian",
            "dmrg_protocol": "JUL08 quimb settings",
        },
    }

    with text_file.open("w") as text:
        text.reconfigure(line_buffering=True)
        print("CAMPS Clifford/Fiedler benchmarks", file=text)
        print(f"systems = {args.systems}", file=text)
        print(f"n_dim = {args.n_dim}", file=text)
        print(f"use_orb = {args.use_orb}", file=text)
        print(f"CAMPS reference state = {args.reference_state}", file=text)
        print("Fiedler reference = exact ground state after CAMPS", file=text)

        for system in args.systems:
            print(f"\nStarting {system}")
            print("\n" + "=" * 80, file=text)
            print(system, file=text)

            with (args.ham_dir / f"{system}.pkl").open("rb") as file_obj:
                H, fci_e, fci_gs, cisd_e, cisd_gs = pickle.load(file_obj)
            HQ = jordan_wigner(H)
            n_qubits = count_qubits(HQ)
            reference_state = fci_gs if args.reference_state == "fci" else cisd_gs

            camps_result = run_camps_for_system(
                HQ=HQ,
                reference_state=reference_state,
                n_qubits=n_qubits,
                args=args,
                camps=camps,
                candidate_clifford=candidate_clifford,
            )

            H_camps = camps_result["H_camps"]
            e_camps_exact, fci_camps = exact_ground_state_for_qubit_hamiltonian(
                H_camps,
                n_qubits,
            )
            if abs(e_camps_exact - fci_e) > 1e-6:
                print(
                    "WARNING: exact energy of CAMPS Hamiltonian differs from saved FCI "
                    f"energy by {e_camps_exact - fci_e}",
                    file=text,
                )

            ent_camps = get_entropies_at_cuts(fci_camps, n_qubits, log_base=LOG_BASE)
            equivalent_symmetries = camps_result["equivalent_symmetries"]

            print("\nEquivalent original-frame Pauli generators from CAMPS:", file=text)
            for i, sym in enumerate(equivalent_symmetries, start=1):
                print(f"  S_{i}: {sym}", file=text)
            write_entropies(text, "\nFCI entanglement after CAMPS:", ent_camps)

            dmrg_row, mpo_row = run_local_benchmarks(
                system=system,
                label="CAMPS",
                Hq=H_camps,
                fci_state=fci_camps,
                fci_energy=fci_e,
                n_qubits=n_qubits,
                skip_dmrg=args.skip_dmrg,
                skip_mpo=args.skip_mpo,
                dmrg_rows=dmrg_rows,
                mpo_rows=mpo_rows,
            )
            print(f"DMRG CAMPS: {dmrg_row}", file=text)
            print(f"MPO CAMPS: {mpo_row}", file=text)

            fiedler_result = None
            if not args.skip_fiedler:
                fiedler_result = fiedler_working_data(H_camps, fci_camps, n_qubits)
                print("\nCAMPS + Fiedler", file=text)
                print("Fiedler ordering:", fiedler_result["info"]["ordering"], file=text)
                write_entropies(
                    text,
                    "FCI entanglement after CAMPS + Fiedler:",
                    fiedler_result["entropies"],
                )

                dmrg_row_f, mpo_row_f = run_local_benchmarks(
                    system=system,
                    label="CAMPS + Fiedler",
                    Hq=fiedler_result["H"],
                    fci_state=fiedler_result["fci_state"],
                    fci_energy=fci_e,
                    n_qubits=n_qubits,
                    skip_dmrg=args.skip_dmrg,
                    skip_mpo=args.skip_mpo,
                    dmrg_rows=dmrg_rows,
                    mpo_rows=mpo_rows,
                )
                print(f"DMRG CAMPS + Fiedler: {dmrg_row_f}", file=text)
                print(f"MPO CAMPS + Fiedler: {mpo_row_f}", file=text)

            datasets = [
                make_benchmark_dataset(
                    tag="CAMPS",
                    symmetries=equivalent_symmetries,
                    cut_entropies=ent_camps,
                    dmrg_bd=dmrg_row["dmrg_bd"],
                )
            ]
            if fiedler_result is not None:
                fiedler_inverse_perm = [
                    int(q) for q in np.argsort(fiedler_result["info"]["ordering"])
                ]
                datasets.append(
                    make_benchmark_dataset(
                        tag="CAMPS + Fiedler",
                        symmetries=[
                            permute_qubits_in_qubit_operator(
                                sym,
                                fiedler_inverse_perm,
                            )
                            for sym in equivalent_symmetries
                        ],
                        cut_entropies=fiedler_result["entropies"],
                        dmrg_bd=dmrg_row_f["dmrg_bd"],
                    )
                )
            BenchmarkData.save_datasets(
                datasets,
                args.output_dir / f"camps_{system}_datasets",
            )

            all_results["systems"][system] = {
                "n_qubits": n_qubits,
                "fci_energy": fci_e,
                "cisd_energy": cisd_e,
                "camps_exact_energy_check": e_camps_exact,
                "camps_n_sites": camps_result["n_sites"],
                "selected_idx": camps_result["selected_idx"],
                "entropy_change": camps_result["entropy_change"],
                "equivalent_symmetries": sym_strings(equivalent_symmetries),
                "entropies_after_camps": ent_camps,
                "dmrg": dmrg_row,
                "mpo": mpo_row,
                "fiedler": None if fiedler_result is None else {
                    "ordering": fiedler_result["info"]["ordering"],
                    "info": fiedler_result["info"],
                    "entropies": fiedler_result["entropies"],
                },
            }

            write_csv(dmrg_csv, dmrg_rows)
            write_csv(mpo_csv, mpo_rows)
            save_json(json_file, all_results)

    write_csv(dmrg_csv, dmrg_rows)
    write_csv(mpo_csv, mpo_rows)
    save_json(json_file, all_results)
    print(f"Wrote text output to {text_file}")
    print(f"Wrote JSON output to {json_file}")
    print(f"Wrote DMRG CSV to {dmrg_csv}")
    print(f"Wrote MPO CSV to {mpo_csv}")


if __name__ == "__main__":
    main()
