"""Symmetry-sector perturbation theory and related basis-state utilities.

The implementation of :func:`symmetry_sector_perturbation_theory` follows the
two-parameter Rayleigh--Schrodinger recursion in Eqs. (25)--(26) of
``SSPT_Aug03.pdf``.  It works entirely with Hamiltonian-induced sparse
computational-basis support; no Hilbert-space enumeration or Hamiltonian matrix
is used.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from openfermion import QubitOperator

from .bs.utils import (
    PauliMask,
    PauliTermStream,
    as_pauli_term_stream,
    combine_mask,
    symplectic_commutes,
    term_to_masks,
    try_add_to_span,
)
from .metrics import find_commuting_paulis
from .state_utils import PauliActionMask, SparseQubitState


Order = Tuple[int, int]
Sector = Tuple[int, ...]


class DegenerateReferenceError(ZeroDivisionError):
    """Raised when a required product state makes the reduced resolvent singular."""


def _validate_orders(orders: Sequence[int]) -> Order:
    if len(orders) != 2:
        raise ValueError("orders must be the pair (m, n).")
    m, n = orders
    if isinstance(m, bool) or isinstance(n, bool):
        raise TypeError("orders must contain non-negative integers.")
    if int(m) != m or int(n) != n or int(m) < 0 or int(n) < 0:
        raise ValueError("orders must contain non-negative integers.")
    return int(m), int(n)


def _infer_initial_n_qubits(initial_state) -> Optional[int]:
    if isinstance(initial_state, SparseQubitState):
        return initial_state.n_qubits
    if isinstance(initial_state, str):
        return len(initial_state)
    if isinstance(initial_state, Sequence) and not isinstance(
        initial_state, (bytes, bytearray)
    ):
        return len(initial_state)
    return None


def _as_reference_index(initial_state, n_qubits: int) -> int:
    """Select a reference determinant from a determinant or sparse guess.

    A multi-determinant ``SparseQubitState`` is a *guess* rather than a valid
    zeroth-order state for the single-reference recursion.  Its largest-weight
    determinant is therefore used as the initial reference.  Lower diagonal
    energy determinants reached by the recursion supersede it automatically.
    """
    if isinstance(initial_state, SparseQubitState):
        if initial_state.n_qubits != n_qubits:
            raise ValueError("initial_state and Hamiltonian qubit counts differ.")
        nonzero = np.flatnonzero(np.abs(initial_state.coeffs) > 0.0)
        if nonzero.size == 0:
            raise ValueError("initial_state cannot be the zero state.")
        largest = nonzero[np.argmax(np.abs(initial_state.coeffs[nonzero]))]
        return int(initial_state.indices[largest])

    if isinstance(initial_state, str):
        if len(initial_state) != n_qubits or any(bit not in "01" for bit in initial_state):
            raise ValueError("initial_state must be an n_qubits-long bitstring.")
        return int(initial_state, 2)

    if isinstance(initial_state, (int, np.integer)) and not isinstance(
        initial_state, (bool, np.bool_)
    ):
        index = int(initial_state)
        if index < 0 or index >= (1 << n_qubits):
            raise ValueError("initial_state basis index is outside the Hilbert space.")
        return index

    try:
        bits = tuple(int(bit) for bit in initial_state)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "initial_state must be a bitstring, bit sequence, basis index, or "
            "SparseQubitState."
        ) from exc
    if len(bits) != n_qubits or any(bit not in (0, 1) for bit in bits):
        raise ValueError("initial_state must contain exactly n_qubits binary entries.")
    return int("".join(str(bit) for bit in bits), 2)


def _symmetry_masks(
    symmetries: Sequence[QubitOperator | PauliMask], n_qubits: int
) -> Tuple[PauliMask, ...]:
    masks = []
    rref_rows = []
    limit = (1 << n_qubits) - 1
    for position, symmetry in enumerate(symmetries):
        if isinstance(symmetry, QubitOperator):
            if len(symmetry.terms) != 1:
                raise ValueError(f"symmetries[{position}] must contain one Pauli word.")
            (term, coefficient), = symmetry.terms.items()
            if abs(complex(coefficient)) == 0.0:
                raise ValueError(f"symmetries[{position}] has zero coefficient.")
            mask = term_to_masks(term, n_qubits)
        else:
            try:
                x_mask, z_mask = symmetry
                mask = int(x_mask), int(z_mask)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Each symmetry must be a single-term QubitOperator or (x, z) mask."
                ) from exc
        if mask == (0, 0):
            raise ValueError(f"symmetries[{position}] cannot be the identity.")
        if mask[0] < 0 or mask[1] < 0 or (mask[0] | mask[1]) & ~limit:
            raise ValueError(f"symmetries[{position}] exceeds n_qubits={n_qubits}.")
        if mask[0] != 0:
            raise ValueError(
                "SSPT requires the symmetries in their Clifford-diagonalized "
                "(I/Z-only) frame. Transform the Hamiltonian, symmetries, and "
                "initial guess with the same Clifford first."
            )
        if not all(symplectic_commutes(mask, previous) for previous in masks):
            raise ValueError("symmetries must mutually commute.")
        new_rows = try_add_to_span(
            combine_mask(mask, n_qubits), rref_rows, 2 * n_qubits
        )
        if new_rows is None:
            raise ValueError("symmetries must be independent.")
        rref_rows = new_rows
        masks.append(mask)
    return tuple(masks)


def _drop_small(state: SparseQubitState, tolerance: float) -> SparseQubitState:
    keep = np.abs(state.coeffs) > tolerance
    return SparseQubitState(
        state.indices[keep], state.coeffs[keep], n_qubits=state.n_qubits
    )


def _add_states(
    *states: SparseQubitState, n_qubits: int, tolerance: float
) -> SparseQubitState:
    nonempty = [state for state in states if state.nnz]
    if not nonempty:
        return SparseQubitState([], [], n_qubits=n_qubits)
    return _drop_small(
        SparseQubitState(
            np.concatenate([state.indices for state in nonempty]),
            np.concatenate([state.coeffs for state in nonempty]),
            n_qubits=n_qubits,
        ),
        tolerance,
    )


def _scale_state(
    state: SparseQubitState, coefficient: complex, tolerance: float
) -> SparseQubitState:
    return _drop_small(
        SparseQubitState(
            state.indices,
            coefficient * state.coeffs,
            n_qubits=state.n_qubits,
        ),
        tolerance,
    )


def _real_if_close(value: complex, tolerance: float = 1e-10):
    value = complex(value)
    if abs(value.imag) <= tolerance * max(1.0, abs(value.real)):
        return float(value.real)
    return value


def _parse_sector(sector, n_symmetries: int) -> int:
    """Pack a public sector label into an integer with symmetry 0 as bit 0."""
    if isinstance(sector, (int, np.integer)) and not isinstance(
        sector, (bool, np.bool_)
    ):
        packed = int(sector)
        if packed < 0 or packed >= (1 << n_symmetries):
            raise ValueError(
                f"Packed sector {packed} does not fit {n_symmetries} symmetries."
            )
        return packed
    if isinstance(sector, str):
        bits = tuple(int(bit) for bit in sector)
    else:
        try:
            bits = tuple(int(bit) for bit in sector)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "A sector must be a packed integer or a binary sequence."
            ) from exc
    if len(bits) != n_symmetries or any(bit not in (0, 1) for bit in bits):
        raise ValueError(
            f"Sector labels must contain {n_symmetries} binary entries."
        )
    return sum(bit << position for position, bit in enumerate(bits))


def _unpack_sector(sector: int, n_symmetries: int) -> Tuple[int, ...]:
    """Return sector eigenvalue bits in the supplied symmetry order."""
    return tuple((int(sector) >> position) & 1 for position in range(n_symmetries))


def _sparse_overlap(bra: SparseQubitState, ket: SparseQubitState) -> complex:
    if bra.n_qubits != ket.n_qubits:
        raise ValueError("Sparse states have different qubit counts.")
    if bra.nnz <= ket.nnz:
        return sum(
            np.conjugate(coefficient) * ket._amp.get(int(index), 0.0j)
            for index, coefficient in zip(bra.indices, bra.coeffs)
        )
    return np.conjugate(_sparse_overlap(ket, bra))


def symmetry_sector_perturbation_theory(
    orders: Sequence[int],
    symmetries: Sequence[QubitOperator | PauliMask],
    hamiltonian: QubitOperator | PauliTermStream,
    initial_state,
    *,
    n_qubits: Optional[int] = None,
    coefficient_tolerance: float = 1e-12,
    amplitude_tolerance: float = 1e-14,
    degeneracy_tolerance: float = 1e-10,
    forbidden_sectors: Optional[Iterable[Sequence[int] | int | str]] = None,
    reallocate_reference: bool = True,
) -> Tuple[
    float,
    float | complex,
    SparseQubitState,
    Dict[Order, float | complex],
    Dict[Order, SparseQubitState],
]:
    """Compute the rectangular SSPT expansion through orders ``(m, n)``.

    ``m`` counts insertions of the off-diagonal, sector-preserving operator
    :math:`V_0`; ``n`` counts insertions of the sector-changing operator
    :math:`V`.  The Hamiltonian is split into ``Z0 + V0 + V`` from its Pauli
    terms and the supplied diagonalized symmetries.  Only determinants produced
    by acting with ``V0`` or ``V`` on already-required sparse corrections are
    generated.

    If a generated determinant has a lower ``Z0`` energy than the current
    reference, it becomes the reference and the complete recursion restarts.
    A distinct required determinant degenerate with the reference makes the
    reduced resolvent singular and raises :class:`DegenerateReferenceError`.
    Terms carry precomputed symmetry-flip vectors, so forbidden-sector filtering
    is performed while transitions are generated rather than by repeatedly
    evaluating symmetry operators on the target determinants.

    Parameters
    ----------
    orders : Sequence[int]
        Pair ``(m, n)`` giving the largest power of ``V0`` and ``V``.
    symmetries : Sequence[QubitOperator | tuple[int, int]]
        I/Z-only, mutually commuting single Pauli words, supplied either as
        ``QubitOperator`` objects or beam-search ``(x_mask, z_mask)`` tuples.
        General Pauli symmetries must first be mapped to Z form by ``Clifford``.
    hamiltonian : QubitOperator | PauliTermStream
        OpenFermion ``QubitOperator`` or repository ``PauliTermStream`` in the
        same frame as ``symmetries`` and ``initial_state``.
    initial_state : str | Sequence[int] | int | SparseQubitState
        Computational-basis bitstring/bit sequence/index, or a
        ``SparseQubitState`` guess.  For a multi-determinant sparse guess, its
        largest-amplitude determinant supplies the initial reference.
    n_qubits : int | None
        Total number of qubits.  When omitted, it is inferred from the inputs.
    coefficient_tolerance : float
        Hamiltonian-coefficient threshold and Hermiticity tolerance.
    amplitude_tolerance : float
        Sparse wavefunction-amplitude threshold.
    degeneracy_tolerance : float
        Absolute energy tolerance used to detect a singular resolvent.
    forbidden_sectors : Iterable[tuple[int, ...] | int | str] | None
        Optional sectors to remove from the recursion.  Each sector is either a
        packed non-negative integer or a binary label in the supplied symmetry
        order, where 0/1 denotes the +1/-1 symmetry eigenvalue.  Every path that
        attempts to enter a forbidden sector is discarded before the resolvent.
    reallocate_reference : bool
        Whether discovery of a lower-``Z0`` determinant reallocates the
        reference and restarts the recursion.  Leave-one-sector-out ranking
        disables this after freezing the reference found by the full run.

    Returns
    -------
    unperturbed_energy, perturbed_energy, wavefunction, energy_traceback,
    wavefunction_traceback
        The first two entries are ``Z0`` reference energy and rectangular-series
        energy.  ``wavefunction`` is the normalized summed series.  Tracebacks
        are dictionaries keyed by ``(p, q)``; the energy dictionary includes
        ``(0, 0): unperturbed_energy`` and the wavefunction dictionary includes
        ``(0, 0): reference_state``.  Each correction state also carries only
        Hamiltonian-connected determinants, so it can be grouped by symmetry
        sector directly from its basis indices.
    """
    max_m, max_n = _validate_orders(orders)
    for name, value in (
        ("coefficient_tolerance", coefficient_tolerance),
        ("amplitude_tolerance", amplitude_tolerance),
        ("degeneracy_tolerance", degeneracy_tolerance),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    inferred_initial = _infer_initial_n_qubits(initial_state)
    if isinstance(hamiltonian, PauliTermStream):
        inferred_hamiltonian = hamiltonian.n_qubits
    elif isinstance(hamiltonian, QubitOperator):
        inferred_hamiltonian = max(
            (qubit + 1 for term in hamiltonian.terms for qubit, _ in term),
            default=0,
        )
    else:
        raise TypeError("hamiltonian must be a QubitOperator or PauliTermStream.")

    inferred_symmetry = 0
    for symmetry in symmetries:
        if isinstance(symmetry, QubitOperator):
            inferred_symmetry = max(
                inferred_symmetry,
                max(
                    (qubit + 1 for term in symmetry.terms for qubit, _ in term),
                    default=0,
                ),
            )
        else:
            try:
                inferred_symmetry = max(
                    inferred_symmetry,
                    int(symmetry[0]).bit_length(),
                    int(symmetry[1]).bit_length(),
                )
            except (TypeError, ValueError, IndexError):
                pass
    if n_qubits is None:
        candidates = [inferred_hamiltonian, inferred_symmetry]
        if inferred_initial is not None:
            candidates.append(inferred_initial)
        n_qubits = max(candidates)
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("n_qubits could not be inferred; supply it explicitly.")
    if inferred_hamiltonian > n_qubits or inferred_symmetry > n_qubits:
        raise ValueError("Hamiltonian or symmetry support exceeds n_qubits.")
    if inferred_initial is not None and inferred_initial != n_qubits:
        raise ValueError("initial_state and n_qubits differ.")

    stream = as_pauli_term_stream(hamiltonian, n_qubits=n_qubits)
    symmetry_masks = _symmetry_masks(symmetries, n_qubits)
    initial_reference = _as_reference_index(initial_state, n_qubits)

    forbidden_sector_ids = {
        _parse_sector(sector, len(symmetry_masks))
        for sector in (() if forbidden_sectors is None else forbidden_sectors)
    }

    symmetry_sign_masks = tuple(
        PauliActionMask.from_pauli_mask(mask, n_qubits).sign_mask
        for mask in symmetry_masks
    )
    sector_cache: Dict[int, int] = {}

    def determinant_sector(index: int) -> int:
        index = int(index)
        cached = sector_cache.get(index)
        if cached is not None:
            return cached
        sector = 0
        for position, sign_mask in enumerate(symmetry_sign_masks):
            parity = bin(index & int(sign_mask)).count("1") & 1
            sector |= parity << position
        sector_cache[index] = sector
        return sector

    reference_sector = determinant_sector(initial_reference)
    if reference_sector in forbidden_sector_ids:
        raise ValueError("The reference sector cannot be forbidden.")

    z0_actions = []
    v0_actions = []
    v_actions = []
    for weighted_term in stream.terms:
        coefficient = complex(weighted_term.signed_coefficient)
        if abs(coefficient) <= coefficient_tolerance:
            continue
        if abs(coefficient.imag) > coefficient_tolerance:
            raise ValueError(
                "SSPT requires a Hermitian Pauli Hamiltonian (real Pauli coefficients)."
            )
        mask = weighted_term.mask
        action = PauliActionMask.from_pauli_mask(
            mask, n_qubits=n_qubits, coeff=coefficient
        )
        sector_flip = 0
        for position, symmetry in enumerate(symmetry_masks):
            if not symplectic_commutes(mask, symmetry):
                sector_flip |= 1 << position
        tagged_action = action, sector_flip
        if mask[0] == 0:
            z0_actions.append(tagged_action)
        elif sector_flip == 0:
            v0_actions.append(tagged_action)
        else:
            v_actions.append(tagged_action)

    def diagonal_energies(indices) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        energies = np.zeros(indices.shape, dtype=np.complex128)
        for action, _sector_flip in z0_actions:
            energies += action.phases(indices)
        if np.any(
            np.abs(energies.imag)
            > coefficient_tolerance * np.maximum(1.0, np.abs(energies.real))
        ):
            raise ValueError("Z0 produced complex computational-basis energies.")
        return energies.real

    def apply_actions(
        state: SparseQubitState,
        actions: Sequence[Tuple[PauliActionMask, int]],
    ) -> SparseQubitState:
        if not actions or state.nnz == 0:
            return SparseQubitState([], [], n_qubits=n_qubits)
        source_sectors = None
        allowed_by_flip = {}
        if forbidden_sector_ids:
            source_sectors = tuple(
                determinant_sector(index) for index in state.indices
            )
        targets = []
        coefficients = []
        for action, sector_flip in actions:
            action_targets = state.indices ^ int(action.flip_mask)
            if forbidden_sector_ids:
                keep = allowed_by_flip.get(sector_flip)
                if keep is None:
                    keep = np.fromiter(
                        (
                            (source_sector ^ sector_flip)
                            not in forbidden_sector_ids
                            for source_sector in source_sectors
                        ),
                        dtype=bool,
                        count=state.nnz,
                    )
                    allowed_by_flip[sector_flip] = keep
                if not np.any(keep):
                    continue
                kept_targets = action_targets[keep]
                kept_coefficients = (
                    action.phases(state.indices[keep]) * state.coeffs[keep]
                )
            else:
                keep = None
                kept_targets = action_targets
                kept_coefficients = action.phases(state.indices) * state.coeffs

            targets.append(kept_targets)
            coefficients.append(kept_coefficients)

        if not targets:
            return SparseQubitState([], [], n_qubits=n_qubits)
        return _drop_small(
            SparseQubitState(
                np.concatenate(targets),
                np.concatenate(coefficients),
                n_qubits=n_qubits,
            ),
            amplitude_tolerance,
        )

    reference_index = initial_reference
    while True:
        reference_energy = float(diagonal_energies([reference_index])[0])
        reference = SparseQubitState(
            [reference_index], [1.0], n_qubits=n_qubits
        )
        energy_traceback: Dict[Order, complex | float] = {
            (0, 0): reference_energy
        }
        wavefunction_traceback: Dict[Order, SparseQubitState] = {
            (0, 0): reference
        }
        restart_index = None

        for total_order in range(1, max_m + max_n + 1):
            p_min = max(0, total_order - max_n)
            p_max = min(max_m, total_order)
            for p in range(p_min, p_max + 1):
                q = total_order - p
                pieces = []
                connected_pieces = []
                if p > 0:
                    piece = apply_actions(
                        wavefunction_traceback[(p - 1, q)], v0_actions
                    )
                    pieces.append(piece)
                    connected_pieces.append(piece)
                if q > 0:
                    piece = apply_actions(
                        wavefunction_traceback[(p, q - 1)], v_actions
                    )
                    pieces.append(piece)
                    connected_pieces.append(piece)

                connected = _add_states(
                    *connected_pieces,
                    n_qubits=n_qubits,
                    tolerance=amplitude_tolerance,
                )
                external_indices = connected.indices[
                    connected.indices != reference_index
                ]
                if external_indices.size:
                    external_energies = diagonal_energies(external_indices)
                    lower = external_energies < (
                        reference_energy - degeneracy_tolerance
                    )
                    if reallocate_reference and np.any(lower):
                        lower_indices = external_indices[lower]
                        lower_energies = external_energies[lower]
                        best = int(np.argmin(lower_energies))
                        restart_index = int(lower_indices[best])
                        new_energy = float(lower_energies[best])
                        print(
                            "SSPT discovered a lower-energy determinant: "
                            f"|{reference_index:0{n_qubits}b}> "
                            f"(E_Z={reference_energy:.12g}) -> "
                            f"|{restart_index:0{n_qubits}b}> "
                            f"(E_Z={new_energy:.12g}); restarting."
                        )
                        break
                    degenerate = np.abs(external_energies - reference_energy) <= (
                        degeneracy_tolerance
                    )
                    if np.any(degenerate):
                        bad_index = int(external_indices[np.flatnonzero(degenerate)[0]])
                        raise DegenerateReferenceError(
                            "The SSPT reduced resolvent diverges: required determinant "
                            f"|{bad_index:0{n_qubits}b}> has Z0 energy "
                            f"{float(diagonal_energies([bad_index])[0]):.12g}, equal "
                            f"within tolerance to reference |{reference_index:0{n_qubits}b}> "
                            f"at {reference_energy:.12g} (order {(p, q)})."
                        )

                # Folded Rayleigh--Schrodinger terms in Eq. (25).
                for a in range(p + 1):
                    for b in range(q + 1):
                        if not (1 <= a + b <= total_order - 1):
                            continue
                        energy_correction = energy_traceback.get((a, b), 0.0)
                        prior = wavefunction_traceback.get((p - a, q - b))
                        if prior is None or prior.nnz == 0 or energy_correction == 0:
                            continue
                        pieces.append(
                            _scale_state(
                                prior, -complex(energy_correction), amplitude_tolerance
                            )
                        )

                rhs = _add_states(
                    *pieces, n_qubits=n_qubits, tolerance=amplitude_tolerance
                )
                # Q0 projection enforces intermediate normalization.
                keep = rhs.indices != reference_index
                rhs = SparseQubitState(
                    rhs.indices[keep], rhs.coeffs[keep], n_qubits=n_qubits
                )
                if rhs.nnz:
                    denominators = reference_energy - diagonal_energies(rhs.indices)
                    singular = np.abs(denominators) <= degeneracy_tolerance
                    if np.any(singular):
                        bad_index = int(rhs.indices[np.flatnonzero(singular)[0]])
                        raise DegenerateReferenceError(
                            "The SSPT reduced resolvent diverges for determinant "
                            f"|{bad_index:0{n_qubits}b}> at order {(p, q)}."
                        )
                    correction = SparseQubitState(
                        rhs.indices,
                        rhs.coeffs / denominators,
                        n_qubits=n_qubits,
                    )
                    correction = _drop_small(correction, amplitude_tolerance)
                else:
                    correction = SparseQubitState([], [], n_qubits=n_qubits)
                wavefunction_traceback[(p, q)] = correction

                energy_pieces = []
                if p > 0:
                    energy_pieces.append(
                        apply_actions(wavefunction_traceback[(p - 1, q)], v0_actions)
                    )
                if q > 0:
                    energy_pieces.append(
                        apply_actions(wavefunction_traceback[(p, q - 1)], v_actions)
                    )
                energy_state = _add_states(
                    *energy_pieces,
                    n_qubits=n_qubits,
                    tolerance=amplitude_tolerance,
                )
                energy_traceback[(p, q)] = _real_if_close(
                    energy_state.to_dict().get(reference_index, 0.0j)
                )
            if restart_index is not None:
                break

        if restart_index is not None:
            reference_index = restart_index
            reference_sector = determinant_sector(reference_index)
            if reference_sector in forbidden_sector_ids:
                raise RuntimeError("A forbidden sector was selected as the reference.")
            continue
        break

    summed_state = _add_states(
        *wavefunction_traceback.values(),
        n_qubits=n_qubits,
        tolerance=amplitude_tolerance,
    )
    if summed_state.norm() == 0.0:
        raise ValueError("The truncated SSPT wavefunction has zero norm.")
    wavefunction = summed_state.normalized()
    perturbed_energy = _real_if_close(sum(energy_traceback.values()))

    # A warning is useful for very small but nonsingular denominators, while an
    # actual zero/near-zero denominator has already raised above.
    all_external = set()
    for state in wavefunction_traceback.values():
        all_external.update(int(index) for index in state.indices)
    all_external.discard(reference_index)
    if all_external:
        gaps = np.abs(
            reference_energy
            - diagonal_energies(np.fromiter(all_external, dtype=np.int64))
        )
        smallest_gap = float(np.min(gaps))
        warning_scale = max(100.0 * degeneracy_tolerance, 1e-12)
        if smallest_gap <= warning_scale:
            warnings.warn(
                "SSPT encountered a near-degenerate resolvent denominator "
                f"of magnitude {smallest_gap:.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )

    return (
        reference_energy,
        perturbed_energy,
        wavefunction,
        energy_traceback,
        wavefunction_traceback,
    )


# Short, conventional alias used in examples and notebooks.
sspt = symmetry_sector_perturbation_theory


def leave_one_sector_out_sspt(
    orders: Sequence[int],
    symmetries: Sequence[QubitOperator | PauliMask],
    hamiltonian: QubitOperator | PauliTermStream,
    initial_state,
    *,
    n_qubits: Optional[int] = None,
    coefficient_tolerance: float = 1e-12,
    amplitude_tolerance: float = 1e-14,
    degeneracy_tolerance: float = 1e-10,
) -> Dict[str, object]:
    """Rank discovered sectors by leave-one-sector-out SSPT importance.

    The full SSPT calculation is performed first, including its normal reference
    reallocation.  Sectors are collected from the support of every order-resolved
    wavefunction correction.  Each non-reference sector is then forbidden in a
    fresh SSPT recursion starting from the full calculation's final reference.
    Thus all paths passing through the removed sector are absent, and the
    reference is identical in every comparison.

    Returns a dictionary containing ``full_result`` (the ordinary five-element
    SSPT return tuple), the reference and involved sector labels, and
    ``rankings`` sorted by decreasing absolute energy change.  Every ranking row
    also contains the signed energy change, normalized-state infidelity, and the
    corresponding five-element ``excluded_result`` for detailed analysis.

    Parameters
    ----------
    orders : Sequence[int]
        Pair ``(m, n)`` of maximum sector-preserving and sector-changing orders.
    symmetries : Sequence[QubitOperator | tuple[int, int]]
        Diagonalized symmetry generators in the same frame as the Hamiltonian.
    hamiltonian : QubitOperator | PauliTermStream
        Pauli Hamiltonian used by SSPT.
    initial_state : str | Sequence[int] | int | SparseQubitState
        Initial determinant or sparse determinant guess.
    n_qubits : int | None
        Explicit qubit count, or ``None`` to infer it.
    coefficient_tolerance : float
        Hamiltonian coefficient/Hermiticity tolerance.
    amplitude_tolerance : float
        Sparse correction-amplitude threshold.
    degeneracy_tolerance : float
        Absolute tolerance for singular resolvent denominators.

    Returns
    -------
    result : dict[str, object]
        ``result["involved_sectors"]`` is ``tuple[tuple[int, ...], ...]``.
        ``result["rankings"]`` is ``list[dict[str, object]]``.  Each ranking
        dictionary contains a sector tuple, its excluded SSPT energy, energy
        change, wavefunction infidelity, and complete excluded SSPT result.
    """
    common_options = dict(
        n_qubits=n_qubits,
        coefficient_tolerance=coefficient_tolerance,
        amplitude_tolerance=amplitude_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
    )
    full_result = symmetry_sector_perturbation_theory(
        orders,
        symmetries,
        hamiltonian,
        initial_state,
        **common_options,
    )
    (
        full_unperturbed_energy,
        full_energy,
        full_wavefunction,
        _full_energy_traceback,
        full_wavefunction_traceback,
    ) = full_result

    reference_state = full_wavefunction_traceback[(0, 0)]
    reference_index = int(reference_state.indices[0])
    resolved_n_qubits = reference_state.n_qubits
    symmetry_masks = _symmetry_masks(symmetries, resolved_n_qubits)
    symmetry_sign_masks = tuple(
        PauliActionMask.from_pauli_mask(mask, resolved_n_qubits).sign_mask
        for mask in symmetry_masks
    )

    def sector_of(index: int) -> int:
        sector = 0
        for position, sign_mask in enumerate(symmetry_sign_masks):
            parity = bin(int(index) & int(sign_mask)).count("1") & 1
            sector |= parity << position
        return sector

    reference_sector_id = sector_of(reference_index)
    involved_sector_ids = {
        sector_of(index)
        for correction in full_wavefunction_traceback.values()
        for index in correction.indices
    }

    rankings = []
    for excluded_sector_id in sorted(involved_sector_ids):
        if excluded_sector_id == reference_sector_id:
            continue
        excluded_sector = _unpack_sector(
            excluded_sector_id, len(symmetry_masks)
        )
        excluded_result = symmetry_sector_perturbation_theory(
            orders,
            symmetries,
            hamiltonian,
            reference_state,
            forbidden_sectors=(excluded_sector_id,),
            reallocate_reference=False,
            **common_options,
        )
        excluded_energy = excluded_result[1]
        excluded_wavefunction = excluded_result[2]
        signed_change = _real_if_close(
            complex(excluded_energy) - complex(full_energy)
        )
        overlap = _sparse_overlap(full_wavefunction, excluded_wavefunction)
        infidelity = max(0.0, float(1.0 - abs(overlap) ** 2))
        rankings.append(
            {
                "sector": excluded_sector,
                "packed_sector": excluded_sector_id,
                "full_energy": full_energy,
                "excluded_energy": excluded_energy,
                "signed_energy_change": signed_change,
                "absolute_energy_change": float(abs(complex(signed_change))),
                "wavefunction_infidelity": infidelity,
                "excluded_result": excluded_result,
            }
        )

    rankings.sort(
        key=lambda row: (
            -row["absolute_energy_change"],
            -row["wavefunction_infidelity"],
            row["packed_sector"],
        )
    )
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank

    return {
        "full_result": full_result,
        "full_unperturbed_energy": full_unperturbed_energy,
        "full_energy": full_energy,
        "reference_sector": _unpack_sector(
            reference_sector_id, len(symmetry_masks)
        ),
        "packed_reference_sector": reference_sector_id,
        "involved_sectors": tuple(
            _unpack_sector(sector, len(symmetry_masks))
            for sector in sorted(involved_sector_ids)
        ),
        "rankings": rankings,
    }


def mps_sector_leave_one_out_energies(
    hamiltonian,
    tensors,
    sectors: Sequence[Sector],
    *,
    n_prefix_sites: Optional[int] = None,
    mpo_builder: str = "blocked_sum",
    mpo_cutoff: float = 1e-10,
    sum_mpo_mod: int = 20,
    n_threads: int = 4,
    stack_mem_gb: float = 0.5,
    scratch=None,
    tag: str = "SECTOR-LOO",
    iprint: int = 0,
) -> Dict[str, object]:
    """Calculate MPS leave-one-sector-out energies by prefix projection.

    Parameters
    ----------
    hamiltonian : QubitOperator | PauliTermStream
        Hamiltonian accepted by the projection evaluator.
    tensors : Sequence[numpy.ndarray]
        High-accuracy qubit MPS arrays accepted by
        ``evaluate_qubit_mps_arrays_prefix_projection_energy``.  The MPS and
        Hamiltonian must be in the frame where symmetry-sector bits occupy the
        leading sites.
    sectors : Sequence[tuple[int, ...]]
        Unique sector labels such as ``[(0, 0, 1), (0, 1, 1), ...]``.  Labels
        are converted to the binary prefix strings required by the projection
        function.  On each run, every listed sector except one is retained.
    n_prefix_sites : int | None
        Number of leading symmetry sites.  By default this is inferred from
        the common sector-label length.
    mpo_builder : str
        ``"blocked_sum"`` or ``"expression"``.
    mpo_cutoff : float
        Numerical cutoff used while constructing the Block2 MPO.
    sum_mpo_mod : int
        Block size used by the blocked-sum MPO builder.
    n_threads : int
        Number of Block2 computational threads.
    stack_mem_gb : float
        Block2 stack-memory allocation in GiB.
    scratch : str | pathlib.Path | None
        Optional Block2 scratch directory.
    tag : str
        Base tag used to distinguish Block2 projection runs.
    iprint : int
        Block2 verbosity level.

    Returns
    -------
    result : dict[str, object]
        Dictionary containing the original MPS energy, the baseline energy
        projected onto all supplied sectors, projected energies after removing
        each sector, signed leave-one-out energy increases relative to that
        baseline, and the resulting ranking.  Projection tensors, probability
        diagnostics, and energy-bound quantities are intentionally discarded.

        In particular, ``result["energy_increases"]`` has type
        ``dict[tuple[int, ...], float]`` and ``result["ranking"]`` has type
        ``tuple[tuple[int, ...], ...]``.

    Notes
    -----
    The requested convenience evaluator constructs a fresh Block2 driver and
    Hamiltonian MPO for every omitted sector.  This keeps the wrapper simple
    and stateless, but rebuilding the MPO can dominate runtime for many sectors.
    """
    from .block2_qubit_benchmark import (
        evaluate_qubit_mps_arrays_prefix_projection_energy,
    )

    sector_tuples = []
    for position, sector in enumerate(sectors):
        try:
            normalized = tuple(int(bit) for bit in sector)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"sectors[{position}] must be a sequence of binary integers."
            ) from exc
        if any(bit not in (0, 1) for bit in normalized):
            raise ValueError(f"sectors[{position}] contains a non-binary value.")
        sector_tuples.append(normalized)

    if len(sector_tuples) < 2:
        raise ValueError("At least two sectors are required for leave-one-out.")
    if len(set(sector_tuples)) != len(sector_tuples):
        raise ValueError("sectors must not contain duplicate labels.")

    inferred_prefix_sites = len(sector_tuples[0])
    if inferred_prefix_sites == 0:
        raise ValueError("Sector labels cannot be empty.")
    if any(len(sector) != inferred_prefix_sites for sector in sector_tuples):
        raise ValueError("All sector labels must have the same length.")
    if n_prefix_sites is None:
        n_prefix_sites = inferred_prefix_sites
    n_prefix_sites = int(n_prefix_sites)
    if n_prefix_sites != inferred_prefix_sites:
        raise ValueError(
            "n_prefix_sites must equal the number of bits in each sector label."
        )

    sector_strings = {
        sector: "".join(str(bit) for bit in sector)
        for sector in sector_tuples
    }
    projected_energies = {}
    energy_increases = {}
    _baseline_tensors, baseline_projection = (
        evaluate_qubit_mps_arrays_prefix_projection_energy(
            hamiltonian=hamiltonian,
            tensors=tensors,
            retained_configurations=list(sector_strings.values()),
            n_prefix_sites=n_prefix_sites,
            mpo_builder=mpo_builder,
            mpo_cutoff=mpo_cutoff,
            sum_mpo_mod=sum_mpo_mod,
            n_threads=n_threads,
            stack_mem_gb=stack_mem_gb,
            scratch=scratch,
            tag=f"{tag}-BASELINE",
            iprint=iprint,
        )
    )
    original_energy = float(baseline_projection["original_energy"])
    baseline_energy = float(baseline_projection["projected_energy"])

    for omitted_position, omitted_sector in enumerate(sector_tuples):
        retained_configurations = [
            sector_strings[sector]
            for sector in sector_tuples
            if sector != omitted_sector
        ]
        _projected_tensors, projection = (
            evaluate_qubit_mps_arrays_prefix_projection_energy(
                hamiltonian=hamiltonian,
                tensors=tensors,
                retained_configurations=retained_configurations,
                n_prefix_sites=n_prefix_sites,
                mpo_builder=mpo_builder,
                mpo_cutoff=mpo_cutoff,
                sum_mpo_mod=sum_mpo_mod,
                n_threads=n_threads,
                stack_mem_gb=stack_mem_gb,
                scratch=scratch,
                tag=f"{tag}-DROP-{omitted_position:04d}",
                iprint=iprint,
            )
        )
        run_original_energy = float(projection["original_energy"])
        projected_energy = float(projection["projected_energy"])
        if not np.isclose(
            run_original_energy,
            original_energy,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                "The unprojected MPS energy changed between leave-one-out runs: "
                f"{original_energy} versus {run_original_energy}."
            )
        projected_energies[omitted_sector] = projected_energy
        energy_increases[omitted_sector] = projected_energy - baseline_energy

    ranking = tuple(
        sorted(
            sector_tuples,
            key=lambda sector: (-energy_increases[sector], sector),
        )
    )
    ranking_rows = tuple(
        {
            "rank": rank,
            "sector": sector,
            "sector_string": sector_strings[sector],
            "projected_energy": projected_energies[sector],
            "energy_increase": energy_increases[sector],
        }
        for rank, sector in enumerate(ranking, start=1)
    )
    return {
        "original_energy": original_energy,
        "baseline_energy": baseline_energy,
        "projected_energies": projected_energies,
        "energy_increases": energy_increases,
        "ranking": ranking,
        "ranking_rows": ranking_rows,
    }


def sector_ranking_ndcg(
    predicted_order: Sequence[Sector],
    predicted_energy_lowerings: Mapping[Sector, float],
    leave_one_out_energies: Mapping[Sector, float],
    *,
    baseline_energy: Optional[float] = None,
    tie_tolerance: float = 1e-10,
    k: Optional[int] = None,
) -> float:
    """Calculate NDCG with ties in both predicted and exact sector scores.

    Parameters
    ----------
    predicted_order : Sequence[tuple[int, ...]]
        Sector vectors only, ordered from most to least important by the
        approximate method.  For example,
        ``[(0, 1, 0), (1, 0, 0), (0, 0, 1)]``.  Energies are supplied separately
        in ``predicted_energy_lowerings``.  Every sector must appear once.
    predicted_energy_lowerings : Mapping[tuple[int, ...], float]
        Approximate nonnegative importance assigned to every sector, for
        example ``{(0, 1, 0): 0.03, (1, 0, 0): 0.02}``.  These values detect
        predicted ties and must be nonincreasing along ``predicted_order`` up to
        ``tie_tolerance``.  A predicted tie block is scored by averaging DCG
        over all orderings within that block, so arbitrary tie-breaking neither
        helps nor hurts the result.
    leave_one_out_energies : Mapping[tuple[int, ...], float]
        Mapping from sector to either its leave-one-out projected energy or its
        already-computed nonnegative energy increase.  For example,
        ``{(0, 1, 0): -107.5, (1, 0, 0): -107.4}``.  Supply
        ``baseline_energy`` for raw projected energies; omit it for energy
        increases.
    baseline_energy : float | None
        Energy with all candidate sectors retained.  When supplied, exact
        relevance is ``E_without_sector - baseline_energy``.
    tie_tolerance : float
        Absolute energy tolerance used independently for predicted and exact
        ties.  A group's largest and smallest scores must differ by no more than
        this value.  Exact tie groups receive common relevance; predicted tie
        groups use permutation-averaged discounted gain.
    k : int | None
        Optional ranking cutoff.  The default uses the complete ordering.

    Returns
    -------
    float
        NDCG in ``[0, 1]`` up to floating-point error.  A value of one means
        the predicted order is ideal modulo the specified score ties.
    """
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative.")
    sectors = list(leave_one_out_energies)
    if not sectors:
        raise ValueError("leave_one_out_energies cannot be empty.")
    predicted = list(predicted_order)
    try:
        predicted_set = set(predicted)
        sector_set = set(sectors)
    except TypeError as exc:
        raise TypeError("Sector labels must be hashable.") from exc
    if len(predicted_set) != len(predicted):
        raise ValueError("predicted_order contains duplicate sectors.")
    predicted_score_set = set(predicted_energy_lowerings)
    if predicted_set != sector_set or predicted_score_set != sector_set:
        missing = sector_set - predicted_set
        unexpected = predicted_set - sector_set
        missing_scores = sector_set - predicted_score_set
        unexpected_scores = predicted_score_set - sector_set
        raise ValueError(
            "predicted_order, predicted_energy_lowerings, and "
            "leave_one_out_energies must contain the same sectors; "
            f"missing_order={missing!r}, unexpected_order={unexpected!r}, "
            f"missing_scores={missing_scores!r}, "
            f"unexpected_scores={unexpected_scores!r}."
        )

    n_sectors = len(sectors)
    if k is None:
        k = n_sectors
    if isinstance(k, bool) or int(k) != k or not 1 <= int(k) <= n_sectors:
        raise ValueError(f"k must be an integer between 1 and {n_sectors}.")
    k = int(k)

    relevance = {}
    for sector, energy in leave_one_out_energies.items():
        value = float(np.real_if_close(energy))
        if baseline_energy is not None:
            value -= float(baseline_energy)
        # Projection of an exact ground state cannot lower the energy.  Small
        # negative values arise from MPS/contraction error and carry no gain.
        relevance[sector] = max(0.0, value)

    predicted_scores = {
        sector: float(np.real_if_close(score))
        for sector, score in predicted_energy_lowerings.items()
    }
    for previous, current in zip(predicted, predicted[1:]):
        if predicted_scores[current] > (
            predicted_scores[previous] + tie_tolerance
        ):
            raise ValueError(
                "predicted_order is not nonincreasing according to "
                "predicted_energy_lowerings: "
                f"{previous!r} has {predicted_scores[previous]:.12g}, while "
                f"{current!r} has {predicted_scores[current]:.12g}."
            )

    exact_order = sorted(
        sectors,
        key=lambda sector: (-relevance[sector], repr(sector)),
    )
    tied_relevance = dict(relevance)
    group_start = 0
    while group_start < n_sectors:
        group_maximum = relevance[exact_order[group_start]]
        group_end = group_start + 1
        while group_end < n_sectors and (
            group_maximum - relevance[exact_order[group_end]]
            <= tie_tolerance
        ):
            group_end += 1
        group = exact_order[group_start:group_end]
        group_value = float(np.mean([relevance[sector] for sector in group]))
        for sector in group:
            tied_relevance[sector] = group_value
        group_start = group_end

    ideal_order = sorted(
        sectors,
        key=lambda sector: (-tied_relevance[sector], repr(sector)),
    )

    def discounted_gain(order) -> float:
        return float(
            sum(
                tied_relevance[sector] / np.log2(position + 1)
                for position, sector in enumerate(order[:k], start=1)
            )
        )

    def predicted_discounted_gain() -> float:
        """Expected DCG under uniform permutations within predicted ties."""
        gain = 0.0
        group_start = 0
        while group_start < n_sectors:
            group_maximum = predicted_scores[predicted[group_start]]
            group_end = group_start + 1
            while group_end < n_sectors and (
                group_maximum - predicted_scores[predicted[group_end]]
                <= tie_tolerance
            ):
                group_end += 1
            group = predicted[group_start:group_end]
            discounts = [
                1.0 / np.log2(position + 1)
                for position in range(group_start + 1, min(group_end, k) + 1)
            ]
            if discounts:
                mean_relevance = float(
                    np.mean([tied_relevance[sector] for sector in group])
                )
                gain += mean_relevance * sum(discounts)
            group_start = group_end
        return float(gain)

    ideal_gain = discounted_gain(ideal_order)
    if ideal_gain == 0.0:
        return 1.0
    score = predicted_discounted_gain() / ideal_gain
    return float(np.clip(score, 0.0, 1.0))

def coupled_computational_basis_states(
    op: QubitOperator,
    reference_state: Sequence[int],
    include_diagonal: bool = True,
) -> Set[Tuple[int, ...]]:
    """
    Return computational basis states coupled to a reference state by a QubitOperator.

    Each Pauli string maps a computational basis state to another basis state by
    flipping exactly the qubits where the Pauli string has X or Y. Z operators
    only contribute phases, so they do not change the bitstring.

    Args:
        op:
            OpenFermion QubitOperator.
        reference_state:
            Computational basis state as a list/tuple of 0s and 1s.
        include_diagonal:
            If True, diagonal Pauli strings with only I/Z operators include the
            unchanged reference state in the returned set.

    Returns:
        Set of coupled basis states as tuples of 0s and 1s. Tuples are used
        because lists cannot be stored in a Python set.
    """
    ref = tuple(reference_state)
    if any(bit not in (0, 1) for bit in ref):
        raise ValueError("reference_state must contain only 0 and 1.")

    coupled_states = set()
    n_qubits = len(ref)

    for term, coeff in op.terms.items():
        if coeff == 0:
            continue

        flipped = list(ref)
        has_flip = False

        for q, pauli in term:
            if q >= n_qubits:
                raise ValueError(
                    f"Pauli term acts on qubit {q}, but reference_state has "
                    f"length {n_qubits}."
                )

            if pauli in ("X", "Y"):
                flipped[q] = 1 - flipped[q]
                has_flip = True
            elif pauli != "Z":
                raise ValueError(f"Unknown Pauli operator {pauli!r} on qubit {q}.")

        if has_flip or include_diagonal:
            coupled_states.add(tuple(flipped))

    return coupled_states

def computational_basis_matrix_element(
    bra_state: Sequence[int],
    op: QubitOperator,
    ket_state: Sequence[int],
) -> complex:
    """
    Compute <bra_state| op |ket_state> without constructing a matrix.

    Args:
        bra_state:
            Computational basis bra as 0/1 bits.
        op:
            OpenFermion QubitOperator.
        ket_state:
            Computational basis ket as 0/1 bits.

    Returns:
        Complex matrix element <bra_state| op |ket_state>.
    """
    bra = tuple(bra_state)
    ket = tuple(ket_state)

    if len(bra) != len(ket):
        raise ValueError("bra_state and ket_state must have the same length.")
    if any(bit not in (0, 1) for bit in bra):
        raise ValueError("bra_state must contain only 0 and 1.")
    if any(bit not in (0, 1) for bit in ket):
        raise ValueError("ket_state must contain only 0 and 1.")

    n_qubits = len(ket)
    matrix_element = 0.0 + 0.0j

    for term, coeff in op.terms.items():
        if coeff == 0:
            continue

        phase = 1.0 + 0.0j
        transformed_ket = list(ket)

        for q, pauli in term:
            if q >= n_qubits:
                raise ValueError(
                    f"Pauli term acts on qubit {q}, but basis states have "
                    f"length {n_qubits}."
                )

            bit = transformed_ket[q]
            if pauli == "X":
                transformed_ket[q] = 1 - bit
            elif pauli == "Y":
                phase *= 1.0j if bit == 0 else -1.0j
                transformed_ket[q] = 1 - bit
            elif pauli == "Z":
                phase *= 1.0 if bit == 0 else -1.0
            else:
                raise ValueError(f"Unknown Pauli operator {pauli!r} on qubit {q}.")

        if tuple(transformed_ket) == bra:
            matrix_element += coeff * phase

    return matrix_element

def complete_S_sorted_insertion(
    symmetries: Sequence[QubitOperator],
    HQ: QubitOperator,
    n_qubits: int,
    target_rank: Optional[int] = None,
) -> List[QubitOperator]:
    """Extend a commuting Pauli basis with large commuting terms from ``HQ``.

    Hamiltonian terms are considered in descending absolute-coefficient order.
    A term is inserted only when it commutes with every generator selected so
    far and increases their binary symplectic GF(2) rank. Inserted terms are
    normalized to coefficient ``+1``.

    The input generators must already be nonidentity, mutually commuting, and
    independent. The input sequence itself is never mutated.
    """
    if n_qubits < 0:
        raise ValueError("n_qubits must be nonnegative.")
    if target_rank is None:
        target_rank = n_qubits
    if not 0 <= target_rank <= n_qubits:
        raise ValueError(
            "target_rank must satisfy 0 <= target_rank <= n_qubits."
        )

    def single_mask(
        operator: QubitOperator,
        *,
        label: str,
    ) -> Tuple[Tuple[Tuple[int, str], ...], PauliMask]:
        if len(operator.terms) != 1:
            raise ValueError(f"{label} must be a single Pauli string.")
        (term, coefficient), = operator.terms.items()
        if abs(complex(coefficient)) <= 1e-12:
            raise ValueError(f"{label} must have a nonzero coefficient.")
        if any(q < 0 or q >= n_qubits for q, _ in term):
            raise ValueError(
                f"{label} acts outside n_qubits={n_qubits}."
            )
        mask = term_to_masks(term, n_qubits)
        if mask == (0, 0):
            raise ValueError(f"{label} cannot be the identity.")
        return term, mask

    extended_symmetries = list(symmetries)
    selected_masks: List[PauliMask] = []
    rref_rows: List[int] = []
    n_bits = 2 * n_qubits

    for index, symmetry in enumerate(extended_symmetries):
        _, mask = single_mask(symmetry, label=f"symmetries[{index}]")
        if not all(
            symplectic_commutes(mask, previous)
            for previous in selected_masks
        ):
            raise ValueError("Input symmetries must mutually commute.")
        new_rows = try_add_to_span(
            combine_mask(mask, n_qubits),
            rref_rows,
            n_bits,
        )
        if new_rows is None:
            raise ValueError("Input symmetries must be independent.")
        rref_rows = new_rows
        selected_masks.append(mask)

    current_rank = len(rref_rows)
    if current_rank >= target_rank:
        return extended_symmetries

    commuting_terms = find_commuting_paulis(
        HQ,
        extended_symmetries,
        verbose=False,
    )
    commuting_terms.sort(
        key=lambda operator: sum(
            abs(coefficient) for coefficient in operator.terms.values()
        ),
        reverse=True,
    )

    for candidate_index, candidate in enumerate(commuting_terms):
        if len(candidate.terms) == 1:
            (candidate_term, _), = candidate.terms.items()
            if not candidate_term:
                continue
        term, mask = single_mask(
            candidate,
            label=f"Hamiltonian candidate {candidate_index}",
        )
        if not all(
            symplectic_commutes(mask, selected)
            for selected in selected_masks
        ):
            continue

        new_rows = try_add_to_span(
            combine_mask(mask, n_qubits),
            rref_rows,
            n_bits,
        )
        if new_rows is None:
            continue

        extended_symmetries.append(QubitOperator(term, 1.0))
        selected_masks.append(mask)
        rref_rows = new_rows
        current_rank += 1
        if current_rank == target_rank:
            return extended_symmetries

    print("Insufficient generators identified: ", current_rank)
    return extended_symmetries
