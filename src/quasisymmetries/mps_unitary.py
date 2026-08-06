"""Composable unitary descriptions for local-gate MPS transformations."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

ParsedMPSGate = tuple


@runtime_checkable
class MPSUnitary(Protocol):
    """Interface for a unitary that can be compiled for the MPS executor."""

    n_qubits: int

    def get_parsed_gates(self) -> Sequence[ParsedMPSGate]:
        """Return basis gates in application order."""

    def get_permutation(self) -> Sequence[int]:
        """Return ``permutation[old_qubit] = new_qubit``."""


@dataclass(frozen=True)
class ComposedMPSUnitary:
    """A gate sequence followed by one final qubit permutation."""

    n_qubits: int
    parsed_gates: tuple[ParsedMPSGate, ...]
    permutation: tuple[int, ...]
    component_types: tuple[str, ...] = ()

    def get_parsed_gates(self) -> tuple[ParsedMPSGate, ...]:
        return self.parsed_gates

    def get_permutation(self) -> tuple[int, ...]:
        return self.permutation


@dataclass(frozen=True)
class PermutationUnitary:
    """A pure qubit permutation, usable as an item in a unitary list."""

    permutation: tuple[int, ...]

    def __init__(self, permutation: Sequence[int]):
        values = tuple(int(value) for value in permutation)
        if sorted(values) != list(range(len(values))):
            raise ValueError("permutation must contain every qubit exactly once")
        object.__setattr__(self, "permutation", values)

    @property
    def n_qubits(self) -> int:
        return len(self.permutation)

    def get_parsed_gates(self) -> tuple:
        return ()

    def get_permutation(self) -> tuple[int, ...]:
        return self.permutation


def decompose_real_orbital_rotation(
    rotation: np.ndarray, tol: float = 1e-12
) -> list[tuple[int, int, complex]]:
    """Decompose a real orthogonal matrix into adjacent Givens rotations."""
    current = np.asarray(np.real_if_close(rotation), dtype=float).copy()
    n_orbitals = current.shape[0]
    if current.shape != (n_orbitals, n_orbitals):
        raise ValueError("orbital rotation must be square")
    if not np.allclose(current.T @ current, np.eye(n_orbitals), atol=tol):
        raise ValueError("orbital rotation is not orthogonal")

    order = []
    for column in range(n_orbitals - 1):
        for row in range(1, n_orbitals):
            if row > column:
                order.append((row, column, 2 * column - row + n_orbitals))
    order.sort(key=lambda item: item[2])

    rotations: list[tuple[int, int, complex]] = []
    for row, column, _ in order:
        if abs(current[row, column]) <= tol:
            continue
        theta = np.arctan2(current[row, column], current[row - 1, column])
        cosine, sine = np.cos(theta), np.sin(theta)
        givens = np.eye(n_orbitals)
        givens[row - 1, row - 1] = cosine
        givens[row, row] = cosine
        givens[row - 1, row] = sine
        givens[row, row - 1] = -sine
        current = givens @ current
        if abs(theta % np.pi) > tol:
            rotations.append((row - 1, row, complex(-theta)))

    for orbital in range(n_orbitals):
        if np.isclose(current[orbital, orbital], -1.0, atol=tol):
            rotations.append((orbital, orbital, 1j * np.pi))
    rotations.reverse()
    return rotations


class OrbitalRotationUnitary:
    """Spin-restricted spatial-orbital rotation compiled to qubit gates."""

    def __init__(self, rotation: np.ndarray, *, tolerance: float = 1e-12):
        matrix = np.asarray(np.real_if_close(rotation), dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("orbital rotation must be a square matrix")
        if not np.allclose(
            matrix.T @ matrix, np.eye(matrix.shape[0]), atol=tolerance
        ):
            raise ValueError("orbital rotation must be orthogonal")
        self.rotation = matrix.copy()
        self.tolerance = float(tolerance)
        self.n_orbitals = int(matrix.shape[0])
        self.n_qubits = 2 * self.n_orbitals

    def get_parsed_gates(self) -> tuple[ParsedMPSGate, ...]:
        gates: list[ParsedMPSGate] = []
        rotations = decompose_real_orbital_rotation(
            self.rotation, self.tolerance
        )
        for first, second, theta in rotations:
            if first == second:
                gates.extend(
                    (
                        ("PHASE", 2 * first, theta),
                        ("PHASE", 2 * first + 1, theta),
                    )
                )
                continue
            if second != first + 1:
                raise RuntimeError("orbital Givens decomposition was not local")
            middle = 2 * first + 1
            gates.extend(
                (
                    ("FSWAP", middle, middle + 1),
                    ("FGIVENS", 2 * first, 2 * first + 1, theta.real),
                    ("FGIVENS", 2 * first + 2, 2 * first + 3, theta.real),
                    ("FSWAP", middle, middle + 1),
                )
            )
        return tuple(gates)

    def get_permutation(self) -> tuple[int, ...]:
        return tuple(range(self.n_qubits))


_ONE_QUBIT_GATES = {"X", "H", "S", "Sdg", "PHASE"}
_TWO_QUBIT_GATES = {"CNOT", "SWAP", "FSWAP", "FGIVENS"}


def _relabel_gate(
    gate: ParsedMPSGate, old_position_for_new: Sequence[int]
) -> ParsedMPSGate:
    name = str(gate[0])
    if name in _ONE_QUBIT_GATES:
        return (name, int(old_position_for_new[int(gate[1])]), *gate[2:])
    if name in _TWO_QUBIT_GATES:
        return (
            name,
            int(old_position_for_new[int(gate[1])]),
            int(old_position_for_new[int(gate[2])]),
            *gate[3:],
        )
    raise ValueError(f"unsupported MPS basis gate {gate!r}")


def _unitary_permutation(unitary: MPSUnitary) -> tuple[int, ...]:
    getter = getattr(unitary, "get_permutation", None)
    if getter is None:
        return tuple(range(int(unitary.n_qubits)))
    values = tuple(int(value) for value in getter())
    if sorted(values) != list(range(int(unitary.n_qubits))):
        raise ValueError(
            f"{type(unitary).__name__} returned an invalid permutation"
        )
    return values


def compose_mps_unitaries(
    unitaries: Sequence[MPSUnitary], *, n_qubits: int | None = None
) -> ComposedMPSUnitary:
    """Compose unitary objects listed from first-applied to last-applied.

    Each component is represented as ``P C``: its parsed gates ``C`` are
    applied first and its permutation ``P`` second. Earlier permutations are
    commuted through later circuits by relabeling the later gate operands.
    """
    components = tuple(unitaries)
    if n_qubits is None:
        if not components:
            raise ValueError("n_qubits is required for an empty unitary list")
        n_qubits = int(components[0].n_qubits)
    n_qubits = int(n_qubits)
    if n_qubits < 0:
        raise ValueError("n_qubits must be nonnegative")

    combined_gates: list[ParsedMPSGate] = []
    combined_permutation = tuple(range(n_qubits))
    component_types = []
    for unitary in components:
        if int(unitary.n_qubits) != n_qubits:
            raise ValueError(
                f"{type(unitary).__name__} acts on {unitary.n_qubits} "
                f"qubits, expected {n_qubits}"
            )
        gate_getter = getattr(unitary, "get_parsed_gates", None)
        if gate_getter is None:
            raise TypeError(
                f"{type(unitary).__name__} has no get_parsed_gates()"
            )
        # For U_new U_acc = P_new C_new P_acc C_acc, commute P_acc
        # leftward: C_new P_acc = P_acc (P_acc^-1 C_new P_acc).
        inverse_accumulated = np.argsort(combined_permutation)
        combined_gates.extend(
            _relabel_gate(gate, inverse_accumulated)
            for gate in gate_getter()
        )
        component_permutation = _unitary_permutation(unitary)
        combined_permutation = tuple(
            component_permutation[combined_permutation[old]]
            for old in range(n_qubits)
        )
        component_types.append(type(unitary).__name__)

    return ComposedMPSUnitary(
        n_qubits=n_qubits,
        parsed_gates=tuple(combined_gates),
        permutation=combined_permutation,
        component_types=tuple(component_types),
    )


def _standard_mps_arrays(tensors) -> list[np.ndarray]:
    arrays = [np.asarray(tensor).copy() for tensor in tensors]
    if not arrays:
        raise ValueError("An MPS must contain at least one tensor.")
    for site, array in enumerate(arrays):
        if array.ndim == 2 and site == 0:
            array = array.reshape(1, array.shape[0], array.shape[1])
        elif array.ndim == 2 and site == len(arrays) - 1:
            array = array.reshape(array.shape[0], array.shape[1], 1)
        elif array.ndim != 3:
            raise ValueError(f"MPS tensor {site} must have rank three.")
        if array.shape[1] != 2:
            raise ValueError("Only qubit MPS tensors are supported.")
        if site and arrays[site - 1].shape[2] != array.shape[0]:
            raise ValueError(f"MPS bond mismatch before site {site}.")
        arrays[site] = array
    return arrays


def _canonicalize_mps_arrays(tensors, center: int) -> None:
    for site in range(center):
        left, physical, right = tensors[site].shape
        q, r = np.linalg.qr(tensors[site].reshape(left * physical, right))
        bond = q.shape[1]
        tensors[site] = q.reshape(left, physical, bond)
        tensors[site + 1] = np.einsum(
            "ab,bsr->asr", r, tensors[site + 1], optimize=True
        )
    for site in range(len(tensors) - 1, center, -1):
        left, physical, right = tensors[site].shape
        q, r = np.linalg.qr(
            tensors[site].reshape(left, physical * right).T
        )
        bond = q.shape[1]
        tensors[site] = q.T.reshape(bond, physical, right)
        tensors[site - 1] = np.einsum(
            "lsb,ba->lsa", tensors[site - 1], r.T, optimize=True
        )


def _array_mps_norm(tensors) -> float:
    environment = np.ones((1, 1), dtype=complex)
    for tensor in tensors:
        environment = np.einsum(
            "ab,asr,bsq->rq",
            environment,
            tensor,
            tensor.conj(),
            optimize=True,
        )
    return float(np.sqrt(np.real_if_close(environment[0, 0])))


def transform_qubit_mps_arrays(
    tensors,
    *,
    unitaries=(),
    max_bond: int | None = None,
    cutoff: float = 1e-13,
) -> tuple[list[np.ndarray], dict]:
    """Apply composed local-gate unitaries to a portable qubit MPS.

    The tensors use the conventional computational basis and have shapes
    ``(left bond, physical qubit, right bond)``. Two-qubit gates are routed
    with SWAPs and truncated only according to ``max_bond`` and ``cutoff``.
    """
    arrays = _standard_mps_arrays(tensors)
    n_qubits = len(arrays)
    if max_bond is not None and int(max_bond) < 1:
        raise ValueError("max_bond must be positive")
    composed = compose_mps_unitaries(unitaries, n_qubits=n_qubits)
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    fermionic_swap = swap.copy()
    fermionic_swap[3, 3] = -1
    errors = []
    gate_counts: dict[str, int] = {}
    largest_bond = max(
        (tensor.shape[2] for tensor in arrays[:-1]), default=1
    )

    def record(name):
        gate_counts[name] = gate_counts.get(name, 0) + 1

    def apply_one(site, gate, name):
        arrays[site] = np.einsum(
            "ab,lbr->lar", gate, arrays[site], optimize=True
        )
        record(name)

    def apply_adjacent(left_site, gate, name):
        nonlocal largest_bond
        _canonicalize_mps_arrays(arrays, left_site)
        theta = np.tensordot(
            arrays[left_site], arrays[left_site + 1], axes=(-1, 0)
        )
        theta = np.einsum(
            "abij,lijr->labr",
            np.asarray(gate).reshape(2, 2, 2, 2),
            theta,
            optimize=True,
        )
        left_dim, _, _, right_dim = theta.shape
        matrix = theta.reshape(left_dim * 2, 2 * right_dim)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        keep = len(singular_values)
        if cutoff > 0:
            keep = max(1, int(np.count_nonzero(singular_values > cutoff)))
        if max_bond is not None:
            keep = min(keep, int(max_bond))
        errors.append(float(np.linalg.norm(singular_values[keep:])))
        arrays[left_site] = u[:, :keep].reshape(left_dim, 2, keep)
        arrays[left_site + 1] = (
            singular_values[:keep, None] * vh[:keep]
        ).reshape(keep, 2, right_dim)
        largest_bond = max(largest_bond, keep)
        record(name)

    def apply_nonlocal(first, second, gate, name):
        if first == second:
            raise ValueError(f"{name} requires distinct qubits")
        if first < second:
            route = list(range(second - 1, first, -1))
            for left_site in route:
                apply_adjacent(left_site, swap, "routing_swap")
            apply_adjacent(first, gate, name)
            for left_site in reversed(route):
                apply_adjacent(left_site, swap, "routing_swap")
        else:
            route = list(range(second, first - 1))
            for left_site in route:
                apply_adjacent(left_site, swap, "routing_swap")
            apply_adjacent(first - 1, swap @ gate @ swap, name)
            for left_site in reversed(route):
                apply_adjacent(left_site, swap, "routing_swap")

    one_qubit_gates = {
        "X": np.array([[0, 1], [1, 0]], dtype=float),
        "H": np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2),
        "S": np.diag([1, 1j]),
        "Sdg": np.diag([1, -1j]),
    }
    for parsed_gate in composed.parsed_gates:
        name = str(parsed_gate[0])
        if name in one_qubit_gates:
            apply_one(int(parsed_gate[1]), one_qubit_gates[name], name)
        elif name == "PHASE":
            apply_one(
                int(parsed_gate[1]),
                np.diag([1.0, np.exp(complex(parsed_gate[2]))]),
                name,
            )
        elif name == "CNOT":
            control, target = int(parsed_gate[1]), int(parsed_gate[2])
            cnot = np.zeros((4, 4))
            for first_bit in range(2):
                for second_bit in range(2):
                    source = 2 * first_bit + second_bit
                    output = (
                        2 * first_bit + (second_bit ^ first_bit)
                    )
                    cnot[output, source] = 1.0
            apply_nonlocal(control, target, cnot, name)
        elif name == "SWAP":
            apply_nonlocal(
                int(parsed_gate[1]), int(parsed_gate[2]), swap, name
            )
        elif name == "FSWAP":
            apply_nonlocal(
                int(parsed_gate[1]),
                int(parsed_gate[2]),
                fermionic_swap,
                name,
            )
        elif name == "FGIVENS":
            theta = float(parsed_gate[3])
            cosine, sine = np.cos(theta), np.sin(theta)
            gate = np.array(
                [
                    [1, 0, 0, 0],
                    [0, cosine, -sine, 0],
                    [0, sine, cosine, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
            apply_nonlocal(
                int(parsed_gate[1]), int(parsed_gate[2]), gate, name
            )
        else:
            raise ValueError(f"unsupported MPS basis gate {parsed_gate!r}")

    desired = np.argsort(composed.permutation).tolist()
    current = list(range(n_qubits))
    for destination, label in enumerate(desired):
        source = current.index(label)
        for left_site in range(source - 1, destination - 1, -1):
            apply_adjacent(left_site, swap, "permutation_swap")
            current[left_site], current[left_site + 1] = (
                current[left_site + 1],
                current[left_site],
            )

    norm = _array_mps_norm(arrays)
    if norm == 0:
        raise RuntimeError("MPS transformation produced a zero state")
    arrays[0] /= norm
    return arrays, {
        "method": "portable_numpy_mps_local_gate_svd",
        "component_types": list(composed.component_types),
        "number_of_composed_gates": len(composed.parsed_gates),
        "composed_permutation": list(composed.permutation),
        "max_bond_cap": max_bond,
        "svd_cutoff": float(cutoff),
        "gate_counts": gate_counts,
        "root_sum_squared_discarded_singular_values": float(
            np.linalg.norm(errors)
        ),
        "largest_single_split_discarded_norm": max(errors, default=0.0),
        "max_final_mps_bond": max(
            (tensor.shape[2] for tensor in arrays[:-1]), default=1
        ),
        "max_intermediate_split_bond": largest_bond,
    }


def _validate_kept_mps_sites(tensors, keep_sites):
    arrays = _standard_mps_arrays(tensors)
    sites = tuple(int(site) for site in keep_sites)
    if tuple(sorted(set(sites))) != sites:
        raise ValueError("keep_sites must be unique and in increasing order")
    if any(site < 0 or site >= len(arrays) for site in sites):
        raise ValueError("keep_sites contains an out-of-range site")
    return arrays, sites


def mps_reduced_density_matrix(
    tensors,
    keep_sites,
    *,
    normalize: bool = True,
    max_dense_elements: int | None = 1 << 24,
) -> np.ndarray:
    """Trace an MPS down to a dense reduced density matrix.

    Parameters
    ----------
    tensors
        Open-boundary qubit MPS tensors in ``(left, physical, right)`` order.
    keep_sites
        Unique retained site indices in increasing MPS order.
    normalize
        If true, normalize the returned density matrix to unit trace.
    max_dense_elements
        Maximum allowed number of dense output matrix elements. Set to ``None``
        to disable the allocation guard.

    Returns
    -------
    density_matrix
        Dense reduced density matrix ordered according to ``keep_sites``, with
        shape ``(2**len(keep_sites), 2**len(keep_sites))``.

    Notes
    -----
    ``keep_sites`` is listed in increasing MPS-site order and determines the
    qubit order of the returned matrix.  The result has shape
    ``(2**len(keep_sites), 2**len(keep_sites))``.  Use
    :func:`mps_configuration_probabilities` when only computational-basis
    populations are required, since it avoids the quadratic ``4**k`` output.
    """
    arrays, sites = _validate_kept_mps_sites(tensors, keep_sites)
    dimension = 1 << len(sites)
    if (
        max_dense_elements is not None
        and dimension * dimension > int(max_dense_elements)
    ):
        raise MemoryError(
            "the requested reduced density matrix has "
            f"{dimension * dimension} elements; use "
            "mps_configuration_probabilities for its diagonal"
        )
    kept = set(sites)
    # Axes are (ket bond, bra bond, retained ket bits, retained bra bits).
    environment = np.ones((1, 1, 1, 1), dtype=np.complex128)
    retained_dimension = 1
    for site, tensor in enumerate(arrays):
        if site in kept:
            environment = np.einsum(
                "abij,asr,btu->ruisjt",
                environment,
                tensor,
                tensor.conj(),
                optimize=True,
            ).reshape(
                tensor.shape[2],
                tensor.shape[2],
                2 * retained_dimension,
                2 * retained_dimension,
            )
            retained_dimension *= 2
        else:
            environment = np.einsum(
                "abij,asr,bsu->ruij",
                environment,
                tensor,
                tensor.conj(),
                optimize=True,
            )
    density_matrix = environment[0, 0]
    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)
    if normalize:
        trace = complex(np.trace(density_matrix))
        if abs(trace) == 0:
            raise RuntimeError("cannot normalize a zero-norm MPS")
        density_matrix /= trace
    return density_matrix


def mps_configuration_probabilities(
    tensors,
    keep_sites,
    *,
    normalize: bool = True,
    cutoff: float = 0.0,
) -> dict[str, float]:
    """Return bit-string probabilities after tracing out all other sites.

    Parameters
    ----------
    tensors
        Open-boundary qubit MPS tensors in ``(left, physical, right)`` order.
    keep_sites
        Unique retained site indices in increasing MPS order.
    normalize
        If true, divide probabilities by their total.
    cutoff
        Include output bit strings whose probability is at least this value.

    Returns
    -------
    probabilities
        Dictionary mapping bit strings, ordered as ``keep_sites``, to their
        computational-basis marginal probabilities.

    Notes
    -----
    This contracts only the diagonal of the reduced density matrix, requiring
    ``O(2**k)`` output memory instead of ``O(4**k)``.  Bit-string characters
    follow the increasing order in ``keep_sites``; for ``range(k)``, the first
    character is MPS/qubit site 0.
    """
    arrays, sites = _validate_kept_mps_sites(tensors, keep_sites)
    kept = set(sites)
    # Axes are (ket bond, bra bond, retained computational-basis index).
    environment = np.ones((1, 1, 1), dtype=np.complex128)
    retained_dimension = 1
    for site, tensor in enumerate(arrays):
        if site in kept:
            environment = np.einsum(
                "abi,asr,bsu->ruis",
                environment,
                tensor,
                tensor.conj(),
                optimize=True,
            ).reshape(
                tensor.shape[2],
                tensor.shape[2],
                2 * retained_dimension,
            )
            retained_dimension *= 2
        else:
            environment = np.einsum(
                "abi,asr,bsu->rui",
                environment,
                tensor,
                tensor.conj(),
                optimize=True,
            )
    probabilities = np.asarray(
        np.real_if_close(environment[0, 0]), dtype=float
    )
    probabilities[np.abs(probabilities) < 1e-14] = 0.0
    if np.any(probabilities < -1e-10):
        raise RuntimeError("MPS contraction produced negative probabilities")
    probabilities = np.maximum(probabilities, 0.0)
    if normalize:
        total = float(probabilities.sum())
        if total == 0:
            raise RuntimeError("cannot normalize a zero-norm MPS")
        probabilities /= total
    width = len(sites)
    return {
        format(index, f"0{width}b"): float(probability)
        for index, probability in enumerate(probabilities)
        if probability >= float(cutoff)
    }


def mps_prefix_configurations_above_probability(
    tensors,
    n_prefix_sites: int,
    epsilon: float,
    *,
    include_equal: bool = False,
) -> dict[str, float]:
    """Find significant configurations on the first sites without enumeration.

    Parameters
    ----------
    tensors
        Open-boundary qubit MPS tensors in ``(left, physical, right)`` order.
    n_prefix_sites
        Number of leading sites included in each returned bit string.
    epsilon
        Individual probability threshold used to prune prefix branches.
    include_equal
        If true, retain probabilities equal to ``epsilon``; otherwise require
        strict inequality.

    Returns
    -------
    probabilities
        Dictionary of retained prefix bit strings and normalized probabilities.

    Notes
    -----
    The last ``N - n_prefix_sites`` qubits are traced out implicitly.  At each
    depth the contraction gives the total probability of every completion of
    the current prefix.  A branch whose total probability is at most
    ``epsilon`` is discarded, since every full bit string below it is bounded
    by that value.

    The search is exact up to floating-point contraction error.  It avoids
    ``2**K`` work when the marginal distribution is concentrated, although no
    algorithm can avoid exponential output when exponentially many strings
    themselves exceed the requested threshold.
    """
    arrays = _standard_mps_arrays(tensors)
    n_sites = len(arrays)
    n_prefix_sites = int(n_prefix_sites)
    epsilon = float(epsilon)
    if not 0 <= n_prefix_sites <= n_sites:
        raise ValueError("n_prefix_sites must lie between zero and N")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    # right_environments[site] contracts sites site, ..., N - 1, leaving the
    # two left-bond indices at `site` open.
    right_environments = [None] * (n_sites + 1)
    right_environments[n_sites] = np.ones((1, 1), dtype=np.complex128)
    for site in range(n_sites - 1, -1, -1):
        tensor = arrays[site]
        right_environments[site] = np.einsum(
            "asr,bsu,ru->ab",
            tensor,
            tensor.conj(),
            right_environments[site + 1],
            optimize=True,
        )

    norm_squared = float(
        np.real_if_close(right_environments[0][0, 0])
    )
    if norm_squared <= 0:
        raise RuntimeError("cannot analyze a zero-norm MPS")
    raw_threshold = epsilon * norm_squared

    def retained(raw_probability):
        tolerance = 1e-14 * norm_squared
        if raw_probability < 0 and abs(raw_probability) <= tolerance:
            raw_probability = 0.0
        if raw_probability < -tolerance:
            raise RuntimeError(
                "MPS contraction produced a negative branch probability"
            )
        if include_equal:
            return raw_probability >= raw_threshold, raw_probability
        return raw_probability > raw_threshold, raw_probability

    # Each branch stores its bit prefix and the wavefunction vector on the
    # MPS bond immediately to the right of that prefix.
    branches = [("", np.ones(1, dtype=np.complex128))]
    for site in range(n_prefix_sites):
        tensor = arrays[site]
        suffix = right_environments[site + 1]
        children = []
        for prefix, left_vector in branches:
            for bit in (0, 1):
                right_vector = np.einsum(
                    "a,ar->r",
                    left_vector,
                    tensor[:, bit, :],
                    optimize=True,
                )
                raw_probability = float(
                    np.real_if_close(
                        np.einsum(
                            "r,ru,u->",
                            right_vector,
                            suffix,
                            right_vector.conj(),
                            optimize=True,
                        )
                    )
                )
                keep, raw_probability = retained(raw_probability)
                if keep:
                    children.append((prefix + str(bit), right_vector))
        branches = children
        if not branches:
            break

    output = {}
    suffix = right_environments[n_prefix_sites]
    for bitstring, vector in branches:
        raw_probability = float(
            np.real_if_close(
                np.einsum(
                    "r,ru,u->",
                    vector,
                    suffix,
                    vector.conj(),
                    optimize=True,
                )
            )
        )
        keep, raw_probability = retained(raw_probability)
        if keep:
            output[bitstring] = max(0.0, raw_probability) / norm_squared
    return output


def mps_prefix_configurations_for_probability_mass(
    tensors,
    n_prefix_sites: int,
    max_omitted_probability: float,
) -> tuple[dict[str, float], dict]:
    """Retain configurations until the unexplored probability is bounded.

    Parameters
    ----------
    tensors
        Open-boundary qubit MPS tensors in ``(left, physical, right)`` order.
    n_prefix_sites
        Number of leading sites included in every complete configuration.
    max_omitted_probability
        Maximum total normalized probability allowed to remain represented by
        the unexplored prefix frontier.

    Returns
    -------
    configurations, metadata
        ``configurations`` maps retained complete bit strings to probabilities.
        ``metadata`` reports retained and omitted probability, configuration
        and frontier counts, and the number of expanded trie prefixes.

    Notes
    -----
    A maximum-probability prefix queue is expanded using exact MPS suffix
    environments.  Completed bit strings are removed from the unexplored
    probability until its total is no larger than
    ``max_omitted_probability``.  This controls the *sum* of discarded
    probabilities, unlike applying an independent cutoff to every string.
    """
    arrays = _standard_mps_arrays(tensors)
    n_sites = len(arrays)
    n_prefix_sites = int(n_prefix_sites)
    max_omitted_probability = float(max_omitted_probability)
    if not 0 <= n_prefix_sites <= n_sites:
        raise ValueError("n_prefix_sites must lie between zero and N")
    if not 0 <= max_omitted_probability < 1:
        raise ValueError("max_omitted_probability must lie in [0, 1)")

    right_environments = [None] * (n_sites + 1)
    right_environments[n_sites] = np.ones((1, 1), dtype=np.complex128)
    for site in range(n_sites - 1, -1, -1):
        tensor = arrays[site]
        right_environments[site] = np.einsum(
            "asr,bsu,ru->ab",
            tensor,
            tensor.conj(),
            right_environments[site + 1],
            optimize=True,
        )
    norm_squared = float(np.real_if_close(right_environments[0][0, 0]))
    if norm_squared <= 0:
        raise RuntimeError("cannot analyze a zero-norm MPS")

    def branch_probability(vector, depth):
        probability = float(
            np.real_if_close(
                np.einsum(
                    "r,ru,u->",
                    vector,
                    right_environments[depth],
                    vector.conj(),
                    optimize=True,
                )
            )
        ) / norm_squared
        if probability < 0 and abs(probability) <= 1e-13:
            return 0.0
        if probability < 0:
            raise RuntimeError("MPS contraction produced negative probability")
        return probability

    counter = itertools.count()
    root_vector = np.ones(1, dtype=np.complex128)
    queue = [(-1.0, next(counter), "", root_vector)]
    unexplored_probability = 1.0
    retained = {}
    expanded_prefixes = 0

    while queue and unexplored_probability > max_omitted_probability:
        negative_probability, _order, prefix, vector = heapq.heappop(queue)
        probability = -negative_probability
        depth = len(prefix)
        if depth == n_prefix_sites:
            retained[prefix] = probability
            unexplored_probability = max(
                0.0, unexplored_probability - probability
            )
            continue

        expanded_prefixes += 1
        tensor = arrays[depth]
        children_probability = 0.0
        for bit in (0, 1):
            child_vector = np.einsum(
                "a,ar->r",
                vector,
                tensor[:, bit, :],
                optimize=True,
            )
            child_probability = branch_probability(child_vector, depth + 1)
            children_probability += child_probability
            if child_probability > 0:
                heapq.heappush(
                    queue,
                    (
                        -child_probability,
                        next(counter),
                        prefix + str(bit),
                        child_vector,
                    ),
                )
        # Preserve the queue's probability-mass invariant despite small MPS
        # contraction/roundoff discrepancies between parent and children.
        unexplored_probability += children_probability - probability

    retained_probability = float(sum(retained.values()))
    # The queue is the exact unreported probability distribution represented
    # by disjoint prefixes. Summing it is more reliable than subtraction.
    unexplored_probability = float(sum(-item[0] for item in queue))
    return retained, {
        "retained_probability": retained_probability,
        "omitted_probability_bound": unexplored_probability,
        "requested_max_omitted_probability": max_omitted_probability,
        "retained_configurations": len(retained),
        "frontier_prefixes": len(queue),
        "expanded_prefixes": expanded_prefixes,
    }


def project_mps_onto_prefix_configurations(
    tensors,
    configurations,
    *,
    n_prefix_sites: int | None = None,
    normalize: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """Project a qubit MPS onto selected configurations of its first sites.

    Parameters
    ----------
    tensors
        Open-boundary qubit MPS tensors. Each tensor must have shape
        ``(left_bond, 2, right_bond)``; rank-two boundary tensors are also
        accepted by the shared MPS validator.
    configurations
        Iterable of unique binary strings to retain. Every string must have
        length ``n_prefix_sites``. The projector acts as identity on all later
        MPS sites.
    n_prefix_sites
        Number of leading MPS sites described by each configuration. If
        omitted, it is inferred from the first configuration.
    normalize
        If true, divide the projected MPS by its norm. If false, return the
        unnormalized projected state.

    Returns
    -------
    projected_tensors, metadata
        ``projected_tensors`` is an exact tensor-network application of the
        prefix projector: no SVD, cutoff, or bond-dimension cap is used.
        ``metadata`` contains the input norm, unnormalized projected norm,
        retained probability, trie bond dimensions, and resulting MPS bond
        dimensions.

    Notes
    -----
    The retained strings are represented by a deterministic prefix trie. Its
    bond state is multiplied into the original MPS bonds for the first
    ``n_prefix_sites`` sites and merged back to one state at the final prefix
    site. Consequently, this constructs the coherent projector
    ``sum_b |b><b|`` rather than an incoherent mixture.
    """
    arrays = _standard_mps_arrays(tensors)
    raw_configurations = tuple(str(value) for value in configurations)
    if not raw_configurations:
        raise ValueError("configurations must contain at least one bit string")
    if n_prefix_sites is None:
        n_prefix_sites = len(raw_configurations[0])
    n_prefix_sites = int(n_prefix_sites)
    if not 0 <= n_prefix_sites <= len(arrays):
        raise ValueError("n_prefix_sites must lie between zero and N")
    if any(len(bits) != n_prefix_sites for bits in raw_configurations):
        raise ValueError(
            "every retained configuration must have n_prefix_sites bits"
        )
    if any(set(bits) - {"0", "1"} for bits in raw_configurations):
        raise ValueError("configurations must contain only '0' and '1'")
    if len(set(raw_configurations)) != len(raw_configurations):
        raise ValueError("configurations contains duplicate bit strings")
    retained = set(raw_configurations)
    if n_prefix_sites == 0 and retained != {""}:
        raise ValueError("the only zero-site configuration is the empty string")

    input_norm = _array_mps_norm(arrays)
    if input_norm == 0:
        raise RuntimeError("cannot project a zero-norm MPS")

    projected = [array.copy() for array in arrays]
    trie_bonds = [1]
    if n_prefix_sites:
        prefixes = [tuple([""])]
        for depth in range(1, n_prefix_sites):
            prefixes.append(
                tuple(sorted({bits[:depth] for bits in retained}))
            )
        prefixes.append(("",))  # merge all accepted leaves after site K - 1
        trie_bonds = [len(level) for level in prefixes]

        for site in range(n_prefix_sites):
            tensor = arrays[site]
            left_prefixes = prefixes[site]
            right_prefixes = prefixes[site + 1]
            right_lookup = {
                prefix: index for index, prefix in enumerate(right_prefixes)
            }
            left_count = len(left_prefixes)
            right_count = len(right_prefixes)
            combined = np.zeros(
                (
                    tensor.shape[0] * left_count,
                    2,
                    tensor.shape[2] * right_count,
                ),
                dtype=tensor.dtype,
            )
            for left_index, prefix in enumerate(left_prefixes):
                for bit in (0, 1):
                    child = prefix + str(bit)
                    if site + 1 == n_prefix_sites:
                        if child not in retained:
                            continue
                        right_index = 0
                    else:
                        right_index = right_lookup.get(child)
                        if right_index is None:
                            continue
                    combined[
                        left_index::left_count,
                        bit,
                        right_index::right_count,
                    ] = tensor[:, bit, :]
            projected[site] = combined

    projected_norm = _array_mps_norm(projected)
    if projected_norm == 0:
        raise RuntimeError("selected configurations have zero MPS probability")
    retained_probability = (projected_norm / input_norm) ** 2
    if normalize:
        projected[0] /= projected_norm

    return projected, {
        "method": "exact_prefix_trie_projector",
        "n_prefix_sites": n_prefix_sites,
        "retained_configurations": len(retained),
        "input_mps_norm": input_norm,
        "unnormalized_projected_mps_norm": projected_norm,
        "retained_probability": float(retained_probability),
        "omitted_probability": float(max(0.0, 1.0 - retained_probability)),
        "normalized_output": bool(normalize),
        "svd_compression_used": False,
        "bond_dimension_cap": None,
        "trie_bond_dimensions": trie_bonds,
        "input_mps_bond_dimensions": [
            int(array.shape[2]) for array in arrays[:-1]
        ],
        "projected_mps_bond_dimensions": [
            int(array.shape[2]) for array in projected[:-1]
        ],
    }
