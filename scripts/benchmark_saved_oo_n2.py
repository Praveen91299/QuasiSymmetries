"""Verify a saved orbital optimization and benchmark Clifford-frame DMRG.

The default input is the N2 run requested in ``oo_n2_20260714``.  The script:

1. rebuilds the FCI non-commutator (NC) cost at the identity and at the saved
   spatial-orbital rotation;
2. expands the spatial parity generators to Jordan--Wigner Z strings;
3. uses :class:`quasisymmetries.Clifford` to map the ordered generators to
   ``+Z0, +Z1, ...`` for both the canonical and optimized Hamiltonians; and
4. runs identical DMRG bond-dimension curves in the two Clifford frames,
   warm-started by the complete CISD state by default.

Run from ``QuasiSymmetries``:

    python scripts/benchmark_saved_oo_n2.py

To verify the costs and prepare/validate the Clifford Hamiltonians without
running the expensive DMRG stage:

    python scripts/benchmark_saved_oo_n2.py --skip-dmrg
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
from openfermion import (
    MolecularData,
    QubitOperator,
    get_fermion_operator,
    hermitian_conjugated,
    jordan_wigner,
)
from pyscf import ci, fci, lib, scf
from pyscf.fci import cistring

from quasisymmetries.clifford_symmetry_optimized import Clifford
from quasisymmetries.mps_unitary import OrbitalRotationUnitary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OO_PROJECT_ROOT = WORKSPACE_ROOT / "quasisymmetry"
DEFAULT_OO_JSON = (
    WORKSPACE_ROOT
    / "oo_n2_20260714"
    / "OO_N2_1.1000_D2h.chk_20260714_174556_e2256b.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "saved" / "results" / "OO_N2_20260714_174556_e2256b"
)

def load_rotation_from_oo_data(oo_data: dict, norb: int) -> np.ndarray:
    """Load and evaluate the optional orbital-optimization helper.

    Parameters
    ----------
    oo_data
        Saved orbital-optimization JSON mapping containing its rotation
        parameterization.
    norb
        Number of spatial orbitals expected by the saved rotation.

    Returns
    -------
    rotation
        Spatial-orbital rotation matrix reconstructed by the external
        ``quasisymmetry`` project.

    Notes
    -----
    The external project is needed only by this script's orbital-optimization
    workflow. Importing shared DMRG helpers from this module therefore must not
    require that sibling repository to be present.
    """
    if str(OO_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(OO_PROJECT_ROOT))
    try:
        from src.orbital_rotation import rotation_from_oo_data
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "the saved orbital-optimization workflow requires the sibling "
            f"repository at {OO_PROJECT_ROOT}"
        ) from exc
    return np.asarray(rotation_from_oo_data(oo_data, norb))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oo-json", type=Path, default=DEFAULT_OO_JSON)
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
        help=(
            "use Block2's memory-bounded blocked sum-of-MPO builder "
            "(default) or Block2's high-memory global expression builder"
        ),
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
    parser.add_argument(
        "--stack-mem-gb",
        type=float,
        default=0.25,
        help=(
            "Block2 stack arena in GiB; MPO tensors are disk-backed to keep "
            "the default safe on memory-constrained machines"
        ),
    )
    parser.add_argument("--davidson-threshold", type=float, default=1e-10)
    parser.add_argument(
        "--full-curve",
        action="store_true",
        help=(
            "continue through every requested bond dimension after reaching "
            "chemical accuracy; by default each frame stops at its first "
            "converged bond dimension"
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
        "--transform-mps-max-bond",
        type=int,
        default=None,
        help=(
            "bond cap while applying orbital-Givens and Clifford gates; "
            "defaults to the largest value in --bond-dims"
        ),
    )
    parser.add_argument(
        "--transform-mps-cutoff",
        type=float,
        default=1e-13,
        help="singular-value cutoff after every two-site circuit gate",
    )
    parser.add_argument(
        "--no-save-tensor-networks",
        dest="save_tensor_networks",
        action="store_false",
        help="do not save the constructed warm-start MPS and Hamiltonian MPO",
    )
    parser.set_defaults(save_tensor_networks=True)
    parser.add_argument("--skip-dmrg", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--cost-atol",
        type=float,
        default=1e-9,
        help="absolute tolerance for matching the costs stored in the OO JSON",
    )
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(jsonable(payload), file_obj, indent=2, allow_nan=False)
        file_obj.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_saved_path(raw_path: str | Path, oo_json: Path) -> Path:
    """Resolve paths saved relative to either project, cwd, or the JSON."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)

    candidates = (
        oo_json.parent / path,
        Path.cwd() / path,
        OO_PROJECT_ROOT / path,
        WORKSPACE_ROOT / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "\n".join(f"  {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {raw_path!r}; tried:\n{tried}")


def solve_fci(input_path: Path) -> tuple[float, np.ndarray]:
    from chemistry import fcidump_data

    data = fcidump_data(str(input_path))
    solver = fci.direct_spin1.FCI()
    solver.max_cycle = 10000
    solver.conv_tol = 1e-10
    energy, ci_vector = solver.kernel(
        data["H1"],
        data["H2"],
        data["NORB"],
        data["NELEC"],
        ecore=data["ECORE"],
    )
    if not solver.converged:
        raise RuntimeError("PySCF FCI did not converge")
    return float(energy), np.asarray(ci_vector, dtype=complex)


def solve_cisd(input_path: Path) -> tuple[float, np.ndarray]:
    """Solve full-space RHF-CISD and return it in the FCI determinant layout."""
    molecule = lib.chkfile.load_mol(str(input_path))
    mean_field = scf.RHF(molecule)
    mean_field.update_from_chk(str(input_path))
    solver = ci.CISD(mean_field)
    correlation_energy, cisd_vector = solver.kernel()
    if not solver.converged:
        raise RuntimeError("PySCF CISD did not converge")
    fcivec = solver.to_fcivec(cisd_vector)
    return (
        float(mean_field.e_tot + correlation_energy),
        np.asarray(fcivec, dtype=complex),
    )


def parity_linear_operators(parity_matrix, norb: int, nelec):
    """Reproduce the parity operators used by optimize_symmetries.py."""
    try:
        import ffsim
    except ImportError as exc:
        raise ImportError(
            "ffsim is required to reproduce the saved orbital-optimization cost"
        ) from exc

    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    if parity_matrix.shape[1] not in (norb, 2 * norb):
        raise ValueError("parity matrix must have norb or 2*norb columns")

    alpha = []
    beta = []
    for orbital in range(norb):
        alpha_op = ffsim.FermionOperator(
            {(ffsim.cre_a(orbital), ffsim.des_a(orbital)): -2, (): 1}
        )
        beta_op = ffsim.FermionOperator(
            {(ffsim.cre_b(orbital), ffsim.des_b(orbital)): -2, (): 1}
        )
        alpha.append(ffsim.linear_operator(alpha_op, norb, nelec))
        beta.append(ffsim.linear_operator(beta_op, norb, nelec))

    operators = []
    for row in parity_matrix:
        factors = []
        if row.size == norb:
            for orbital in np.flatnonzero(row % 2):
                factors.extend((alpha[int(orbital)], beta[int(orbital)]))
        else:
            for spin_orbital in np.flatnonzero(row % 2):
                orbital, spin = divmod(int(spin_orbital), 2)
                factors.append(alpha[orbital] if spin == 0 else beta[orbital])
        if not factors:
            raise ValueError("identity parity rows are not valid generators")
        product = factors[0]
        for factor in factors[1:]:
            product = product @ factor
        operators.append(product)
    return operators


def nc_cost(moldata, symmetries, reference_state, orbital_rotation) -> float:
    """Evaluate the exact objective used for the saved ``cost_function=NC``."""
    import ffsim

    rotated_state = ffsim.apply_orbital_rotation(
        reference_state,
        orbital_rotation,
        moldata.norb,
        moldata.nelec,
    )
    hamiltonian = ffsim.linear_operator(
        moldata.hamiltonian.rotated(orbital_rotation),
        norb=moldata.norb,
        nelec=moldata.nelec,
    )
    h_state = hamiltonian @ rotated_state
    total = 0.0
    for symmetry in symmetries:
        commutator_state = (
            hamiltonian @ (symmetry @ rotated_state)
            - symmetry @ h_state
        )
        total += float(np.vdot(commutator_state, commutator_state).real)
    return total


def z_symmetries_from_parity_matrix(parity_matrix, norb: int):
    """Expand spatial parities to interleaved-spin Jordan--Wigner Z strings."""
    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    if parity_matrix.shape[1] == norb:
        expanded = np.repeat(parity_matrix, 2, axis=1)
    elif parity_matrix.shape[1] == 2 * norb:
        expanded = parity_matrix
    else:
        raise ValueError("parity matrix must have norb or 2*norb columns")

    symmetries = []
    for row in expanded:
        support = np.flatnonzero(row % 2)
        if support.size == 0:
            raise ValueError("identity parity rows are not valid generators")
        symmetries.append(
            QubitOperator(tuple((int(index), "Z") for index in support), 1.0)
        )
    return symmetries


def molecular_hamiltonian_to_jw(molecular_hamiltonian, nelec):
    """Convert ffsim chemistry tensors to OpenFermion's interleaved-spin JW form."""
    norb = molecular_hamiltonian.one_body_tensor.shape[0]
    molecule = MolecularData(
        geometry="FCIDUMP",
        basis="unknown",
        multiplicity=abs(int(nelec[0]) - int(nelec[1])) + 1,
        charge=0,
    )
    molecule.n_orbitals = norb
    molecule.n_qubits = 2 * norb
    molecule.n_electrons = int(sum(nelec))
    molecule.nuclear_repulsion = float(np.real(molecular_hamiltonian.constant))
    molecule.one_body_integrals = np.asarray(
        molecular_hamiltonian.one_body_tensor, dtype=float
    )
    # ffsim: (p,s,q,r); OpenFermion MolecularData: (p,q,r,s).
    molecule.two_body_integrals = np.transpose(
        np.asarray(molecular_hamiltonian.two_body_tensor, dtype=float),
        (0, 2, 3, 1),
    )
    fermion_hamiltonian = get_fermion_operator(
        molecule.get_molecular_hamiltonian()
    )
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress(abs_tol=1e-12)
    return qubit_hamiltonian


def hermitize_qubit_hamiltonian(
    hamiltonian: QubitOperator, *, cutoff: float = 1e-12
) -> tuple[QubitOperator, float]:
    """Remove numerical anti-Hermitian residue from a Pauli expansion."""
    adjoint = hermitian_conjugated(hamiltonian)
    antihermitian = hamiltonian - adjoint
    largest_residual = max(
        (abs(value) for value in antihermitian.terms.values()), default=0.0
    )
    hermitian = 0.5 * (hamiltonian + adjoint)
    hermitian.compress(abs_tol=cutoff)
    return hermitian, float(largest_residual)


def expand_ci_state(
    ci_vector: np.ndarray, norb: int, nelec, cutoff: float = 1e-14
) -> np.ndarray:
    """Embed a PySCF/ffsim CI vector into OpenFermion's full JW state space."""
    n_alpha, n_beta = map(int, nelec)
    alpha_strings = np.asarray(
        cistring.make_strings(range(norb), n_alpha), dtype=np.int64
    )
    beta_strings = np.asarray(
        cistring.make_strings(range(norb), n_beta), dtype=np.int64
    )
    ci_vector = np.asarray(ci_vector).reshape(
        len(alpha_strings), len(beta_strings)
    )
    state = np.zeros(1 << (2 * norb), dtype=complex)

    for alpha_index, alpha_det in enumerate(alpha_strings):
        alpha_occ = [
            orbital for orbital in range(norb)
            if (int(alpha_det) >> orbital) & 1
        ]
        for beta_index, beta_det in enumerate(beta_strings):
            coefficient = ci_vector[alpha_index, beta_index]
            if abs(coefficient) <= cutoff:
                continue
            beta_occ = [
                orbital for orbital in range(norb)
                if (int(beta_det) >> orbital) & 1
            ]
            inversions = sum(
                beta_orbital < alpha_orbital
                for alpha_orbital in alpha_occ
                for beta_orbital in beta_occ
            )
            phase = -1.0 if inversions % 2 else 1.0
            basis_index = 0
            for orbital in alpha_occ:
                basis_index |= 1 << (2 * norb - 1 - 2 * orbital)
            for orbital in beta_occ:
                basis_index |= 1 << (2 * norb - 2 - 2 * orbital)
            state[basis_index] = phase * coefficient

    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"expanded CI state has norm {norm}, expected 1")
    return state


def sparse_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return every exactly nonzero computational-basis coefficient."""
    state = np.asarray(state).reshape(-1)
    indices = np.flatnonzero(state != 0)
    return indices, state[indices]


def apply_one_qubit_gate(state, matrix, qubit: int, n_qubits: int):
    tensor = np.asarray(state).reshape((2,) * n_qubits)
    front = np.moveaxis(tensor, qubit, 0)
    transformed = np.tensordot(matrix, front, axes=(1, 0))
    return np.moveaxis(transformed, 0, qubit).reshape(-1)


def transform_state_matrix_free(clifford: Clifford, state) -> np.ndarray:
    """Apply the stored Clifford without constructing its 2**n sparse matrix."""
    transformed = np.asarray(state, dtype=complex).reshape(-1)
    n_qubits = clifford.n_qubits
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    h_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    s_gate = np.diag([1, 1j])
    sdg_gate = np.diag([1, -1j])
    one_qubit_gates = {
        "X": x_gate,
        "H": h_gate,
        "S": s_gate,
        "Sdg": sdg_gate,
    }

    for gate in clifford.parsed_gates:
        name = str(gate[0])
        if name in one_qubit_gates:
            transformed = apply_one_qubit_gate(
                transformed, one_qubit_gates[name], int(gate[1]), n_qubits
            )
        elif name == "CNOT":
            control, target = int(gate[1]), int(gate[2])
            tensor = transformed.reshape((2,) * n_qubits)
            front = np.moveaxis(tensor, (control, target), (0, 1)).copy()
            front[1] = front[1, ::-1]
            transformed = np.moveaxis(
                front, (0, 1), (control, target)
            ).reshape(-1)
        else:
            raise ValueError(f"unsupported Clifford gate {gate!r}")

    # permutation[old_qubit] = new_qubit
    old_qubit_for_new_axis = np.argsort(np.asarray(clifford.permutation))
    transformed = np.transpose(
        transformed.reshape((2,) * n_qubits),
        axes=old_qubit_for_new_axis,
    ).reshape(-1)
    return transformed


def build_clifford(symmetries, n_qubits: int) -> Clifford:
    clifford = Clifford.from_symmetries(
        symmetries,
        n_qubits=n_qubits,
        symmetry_qubits_first=True,
        synthesis_basis="Z",
        generator_mapping="positive_z",
    )
    expected = [
        QubitOperator(((index, "Z"),), 1.0)
        for index in range(len(symmetries))
    ]
    actual = list(clifford.transformed_symmetries)
    if actual != expected:
        raise RuntimeError(
            f"Clifford generator mapping failed:\nactual={actual}\nexpected={expected}"
        )
    return clifford


def run_qubit_dmrg_curve(
    *,
    label: str,
    hamiltonian,
    exact_state,
    sparse_warm_state=None,
    exact_energy: float,
    warm_start_energy: float | None = None,
    n_qubits: int,
    args: argparse.Namespace,
    unitaries=(),
    validate_warm_start_energy: bool = True,
) -> tuple[list[dict], dict]:
    """Run a qubit-Hamiltonian curve using only Pauli-mode pyblock2."""
    from quasisymmetries.block2_qubit_benchmark import (
        run_block2_qubit_dmrg_curve,
    )

    transform_max_bond = getattr(args, "transform_mps_max_bond", None)
    transform_cutoff = getattr(args, "transform_mps_cutoff", 1e-13)
    return run_block2_qubit_dmrg_curve(
        label=label,
        hamiltonian=hamiltonian,
        exact_state=exact_state,
        sparse_state=sparse_warm_state,
        exact_energy=exact_energy,
        warm_start_energy=warm_start_energy,
        n_qubits=n_qubits,
        bond_dims=args.bond_dims,
        dmrg_sweeps=args.dmrg_sweeps,
        dmrg_tolerance=args.dmrg_tol,
        sweep_tolerance=args.sweep_tol,
        mps_cutoff=args.mps_cutoff,
        mpo_cutoff=args.mpo_cutoff,
        mpo_builder=args.block2_mpo_builder,
        sum_mpo_mod=args.sum_mpo_mod,
        initial_state=args.initial_state,
        sparse_batch_size=args.sparse_batch_size,
        sparse_compression_cutoff=args.sparse_compression_cutoff,
        unitaries=unitaries,
        transform_max_bond=(
            transform_max_bond
            if transform_max_bond is not None
            else max(args.bond_dims)
        ),
        transform_cutoff=transform_cutoff,
        validate_warm_start_energy=validate_warm_start_energy,
        full_curve=args.full_curve,
        n_threads=args.n_threads,
        n_mkl_threads=getattr(args, "n_mkl_threads", 1),
        stack_mem_gb=args.stack_mem_gb,
        davidson_threshold=args.davidson_threshold,
        verbose=args.verbose,
        artifact_dir=(
            args.output_dir / "tensor_networks"
            if args.save_tensor_networks
            else None
        ),
    )


def main() -> None:
    args = parse_args()
    try:
        from chemistry import load_moldata
    except ImportError as exc:
        raise ImportError(
            "The saved-cost workflow requires ffsim, pyscf, openfermion, and "
            "openfermionpyscf from the sibling quasisymmetry project."
        ) from exc

    oo_json = args.oo_json.expanduser().resolve()
    with oo_json.open(encoding="utf-8") as file_obj:
        oo_data = json.load(file_obj)

    if str(oo_data.get("cost_function", "")).upper() != "NC":
        raise ValueError("this verifier currently expects a saved NC optimization")
    if str(oo_data.get("reference", "")).lower() != "fci":
        raise ValueError("this verifier currently expects reference='fci'")

    molpath = resolve_saved_path(oo_data["molpath"], oo_json)
    parity_path = resolve_saved_path(oo_data["parity"], oo_json)
    parity_matrix = np.atleast_2d(np.loadtxt(parity_path, dtype=int))
    moldata = load_moldata(str(molpath))
    norb = int(moldata.norb)
    n_qubits = 2 * norb
    if parity_matrix.shape[0] >= n_qubits:
        raise ValueError("need fewer independent symmetries than qubits")

    print(f"OO JSON: {oo_json}", flush=True)
    print(f"Hamiltonian: {molpath}", flush=True)
    print(f"Parity matrix: {parity_path}", flush=True)
    print(
        f"System: norb={norb}, n_qubits={n_qubits}, "
        f"nelec={tuple(moldata.nelec)}, n_symmetries={len(parity_matrix)}",
        flush=True,
    )

    fci_energy, fci_ci = solve_fci(molpath)
    reference_state = fci_ci.reshape(-1)
    cisd_energy = None
    cisd_ci = None
    if args.initial_state == "cisd":
        cisd_energy, cisd_ci = solve_cisd(molpath)
        print(
            f"CISD warm-start energy: {cisd_energy:.12f} "
            f"(FCI error {abs(cisd_energy - fci_energy):.3e})",
            flush=True,
        )
    orbital_rotation = load_rotation_from_oo_data(oo_data, norb)
    identity = np.eye(norb)
    linear_symmetries = parity_linear_operators(
        parity_matrix, norb, moldata.nelec
    )

    print("Recomputing saved NC costs...", flush=True)
    cost_before = nc_cost(
        moldata, linear_symmetries, reference_state, identity
    )
    cost_after = nc_cost(
        moldata, linear_symmetries, reference_state, orbital_rotation
    )
    stored_before = float(oo_data["cost_before"])
    stored_after = float(oo_data["cost_after"])
    before_error = abs(cost_before - stored_before)
    after_error = abs(cost_after - stored_after)
    print(
        f"cost before: recomputed={cost_before:.15g} "
        f"stored={stored_before:.15g} diff={before_error:.3e}",
        flush=True,
    )
    print(
        f"cost after:  recomputed={cost_after:.15g} "
        f"stored={stored_after:.15g} diff={after_error:.3e}",
        flush=True,
    )
    if before_error > args.cost_atol or after_error > args.cost_atol:
        raise AssertionError(
            "recomputed OO costs do not match the saved values within "
            f"--cost-atol={args.cost_atol}"
        )

    pauli_symmetries = z_symmetries_from_parity_matrix(
        parity_matrix, norb
    )
    clifford = build_clifford(pauli_symmetries, n_qubits)
    canonical_jw = molecular_hamiltonian_to_jw(
        moldata.hamiltonian, moldata.nelec
    )
    optimized_jw = molecular_hamiltonian_to_jw(
        moldata.hamiltonian.rotated(orbital_rotation), moldata.nelec
    )
    canonical_jw, canonical_antihermitian_residual = (
        hermitize_qubit_hamiltonian(canonical_jw)
    )
    optimized_jw, optimized_antihermitian_residual = (
        hermitize_qubit_hamiltonian(optimized_jw)
    )
    print(
        "Largest removed anti-Hermitian Pauli coefficient: "
        f"canonical={canonical_antihermitian_residual:.3e}, "
        f"orbital_rotated={optimized_antihermitian_residual:.3e}",
        flush=True,
    )
    canonical_clifford = clifford.transform(canonical_jw)
    optimized_clifford = clifford.transform(optimized_jw)
    canonical_terms_before = len(canonical_jw.terms)
    optimized_terms_before = len(optimized_jw.terms)
    canonical_terms_after = len(canonical_clifford.terms)
    optimized_terms_after = len(optimized_clifford.terms)
    # The untransformed operators are no longer used. Their Python Pauli-term
    # dictionaries otherwise remain live during Block2 MPO construction.
    del canonical_jw, optimized_jw
    gc.collect()

    warm_ci = cisd_ci if args.initial_state == "cisd" else fci_ci
    warm_start_energy = (
        cisd_energy if args.initial_state == "cisd" else fci_energy
    )
    # For CISD, retain every nonzero CI coefficient. No determinant-amplitude
    # cutoff is applied before MPS construction.
    state_cutoff = 0.0 if args.initial_state == "cisd" else 1e-14
    canonical_state = expand_ci_state(
        warm_ci, norb, moldata.nelec, cutoff=state_cutoff
    )
    canonical_sparse_state = sparse_state(canonical_state)
    # Both curves start from this same canonical-orbital selected-CI state.
    # The optimized curve creates its rotated state by applying a capped
    # adjacent-Givens tensor circuit directly to the pyblock MPS.
    optimized_sparse_state = canonical_sparse_state
    print(
        "Warm-start source determinants before tensor circuits: "
        f"{len(canonical_sparse_state[0])}; the canonical and optimized "
        "curves use the same determinant-sum MPS.",
        flush=True,
    )
    canonical_dmrg_state = None
    optimized_dmrg_state = None
    del canonical_state, warm_ci
    gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "benchmark.json"
    curve_path = args.output_dir / "dmrg_curve.csv"
    result = {
        "input": {
            "oo_json": oo_json,
            "molpath": molpath,
            "parity": parity_path,
        },
        "system": {
            "norb": norb,
            "n_qubits": n_qubits,
            "nelec": tuple(int(value) for value in moldata.nelec),
            "fci_energy": fci_energy,
            "cisd_energy": cisd_energy,
        },
        "cost_verification": {
            "cost_function": "NC",
            "recomputed_before": cost_before,
            "stored_before": stored_before,
            "absolute_error_before": before_error,
            "recomputed_after": cost_after,
            "stored_after": stored_after,
            "absolute_error_after": after_error,
            "cost_atol": args.cost_atol,
            "passed": True,
        },
        "clifford": {
            "synthesis_basis": clifford.synthesis_basis,
            "generator_mapping": clifford.generator_mapping,
            "factor_descriptions": list(clifford.factor_descriptions),
            "permutation": list(clifford.permutation),
            "input_symmetries": [str(sym) for sym in pauli_symmetries],
            "transformed_symmetries": [
                str(sym) for sym in clifford.transformed_symmetries
            ],
            "canonical_terms_before": canonical_terms_before,
            "canonical_terms_after": canonical_terms_after,
            "optimized_terms_before": optimized_terms_before,
            "optimized_terms_after": optimized_terms_after,
            "canonical_antihermitian_residual_removed": (
                canonical_antihermitian_residual
            ),
            "optimized_antihermitian_residual_removed": (
                optimized_antihermitian_residual
            ),
        },
        "dmrg_settings": {
            "skipped": args.skip_dmrg,
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
            "qubit_backend": args.qubit_backend,
            "n_threads": args.n_threads,
            "stack_mem_gb": args.stack_mem_gb,
            "davidson_threshold": args.davidson_threshold,
            "sparse_batch_size": args.sparse_batch_size,
            "sparse_compression_cutoff": args.sparse_compression_cutoff,
            "transform_mps_max_bond": (
                args.transform_mps_max_bond
                if args.transform_mps_max_bond is not None
                else max(args.bond_dims)
            ),
            "transform_mps_cutoff": args.transform_mps_cutoff,
            "save_tensor_networks": args.save_tensor_networks,
        },
        "warm_start": {
            "kind": args.initial_state,
            "energy": warm_start_energy,
            "canonical_source_nonzero_determinants": len(
                canonical_sparse_state[0]
            ),
            "optimized_source_nonzero_determinants": len(
                optimized_sparse_state[0]
            ),
            "determinant_coefficient_truncation": False,
            "optimized_state_construction": (
                "canonical selected-CI pyblock MPS followed by adjacent "
                "orbital Givens gates with capped two-site SVDs"
            ),
            "mps_numerical_rank_cutoff": (
                args.sparse_compression_cutoff
                if args.qubit_backend == "block2"
                else args.mps_cutoff
            ),
        },
        "dmrg_summary": {},
        "comparison": {},
        "dmrg_curve": [],
    }
    save_json(result_path, result)

    if not args.skip_dmrg:
        all_rows = []
        orbital_unitary = OrbitalRotationUnitary(
            orbital_rotation,
            tolerance=max(float(args.transform_mps_cutoff), 1e-12),
        )
        for (
            label,
            hamiltonian,
            state,
            sparse_warm_state,
            warm_unitaries,
            validate_warm_energy,
        ) in (
            (
                "canonical_orbitals",
                canonical_clifford,
                canonical_dmrg_state,
                canonical_sparse_state,
                (clifford,),
                True,
            ),
            (
                "optimized_orbitals",
                optimized_clifford,
                optimized_dmrg_state,
                optimized_sparse_state,
                (orbital_unitary, clifford),
                False,
            ),
        ):
            rows, summary = run_qubit_dmrg_curve(
                label=label,
                hamiltonian=hamiltonian,
                exact_state=state,
                sparse_warm_state=sparse_warm_state,
                exact_energy=fci_energy,
                warm_start_energy=warm_start_energy,
                n_qubits=n_qubits,
                args=args,
                unitaries=warm_unitaries,
                validate_warm_start_energy=validate_warm_energy,
            )
            all_rows.extend(rows)
            result["dmrg_curve"] = all_rows
            result["dmrg_summary"][label] = summary
            write_csv(curve_path, all_rows)
            save_json(result_path, result)

        canonical_bd = result["dmrg_summary"]["canonical_orbitals"][
            "first_converged_bond_dim"
        ]
        optimized_bd = result["dmrg_summary"]["optimized_orbitals"][
            "first_converged_bond_dim"
        ]
        result["comparison"] = {
            "canonical_first_converged_bond_dim": canonical_bd,
            "optimized_first_converged_bond_dim": optimized_bd,
            "bond_dimension_reduction": (
                canonical_bd - optimized_bd
                if canonical_bd is not None and optimized_bd is not None
                else None
            ),
            "optimized_over_canonical_ratio": (
                optimized_bd / canonical_bd
                if canonical_bd is not None
                and optimized_bd is not None
                and canonical_bd != 0
                else None
            ),
        }
        save_json(result_path, result)

    print(f"Wrote results to {result_path}", flush=True)
    if not args.skip_dmrg:
        print(f"Wrote DMRG curve to {curve_path}", flush=True)


if __name__ == "__main__":
    main()
