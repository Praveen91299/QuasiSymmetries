"""Scalable chemistry-state preparation helpers."""

from __future__ import annotations

import numpy as np

from .state_utils import SparseQubitState


def _spin_strings_to_interleaved_jw_indices(
    alpha_strings,
    beta_strings,
    n_spatial_orbitals: int,
) -> np.ndarray:
    """Convert PySCF alpha/beta occupation strings to repository JW indices."""
    alpha_strings = np.asarray(alpha_strings, dtype=np.int64).reshape(-1)
    beta_strings = np.asarray(beta_strings, dtype=np.int64).reshape(-1)
    if alpha_strings.shape != beta_strings.shape:
        raise ValueError("alpha_strings and beta_strings must have equal shape")
    indices = np.zeros(len(alpha_strings), dtype=np.int64)
    n_qubits = 2 * int(n_spatial_orbitals)
    for orbital in range(int(n_spatial_orbitals)):
        alpha_occupied = (alpha_strings >> orbital) & 1
        beta_occupied = (beta_strings >> orbital) & 1
        indices |= alpha_occupied << (n_qubits - 1 - 2 * orbital)
        indices |= beta_occupied << (n_qubits - 2 - 2 * orbital)
    return indices


def _pyscf_to_interleaved_jw_phases(
    alpha_strings,
    beta_strings,
    n_spatial_orbitals: int,
) -> np.ndarray:
    """Return fermionic phases for PySCF-to-interleaved-JW conversion.

    Parameters
    ----------
    alpha_strings, beta_strings
        Equal-length arrays of PySCF occupation bit strings. PySCF's CI
        coefficient convention groups the complete alpha determinant before
        the complete beta determinant.
    n_spatial_orbitals
        Number of spatial orbitals encoded by each occupation string.

    Returns
    -------
    phases
        Real array containing ``+1`` or ``-1`` for each determinant. Multiplying
        a PySCF CI coefficient by this phase expresses it in ascending
        interleaved spin-orbital order
        ``alpha_0, beta_0, alpha_1, beta_1, ...``.

    Notes
    -----
    Every occupied beta orbital ``j`` must cross every occupied alpha orbital
    ``i > j`` when changing from grouped alpha/beta creation operators to the
    interleaved order. The phase is minus one when the number of such crossings
    is odd.
    """
    alpha_strings = np.asarray(alpha_strings, dtype=np.int64).reshape(-1)
    beta_strings = np.asarray(beta_strings, dtype=np.int64).reshape(-1)
    if alpha_strings.shape != beta_strings.shape:
        raise ValueError("alpha_strings and beta_strings must have equal shape")
    inversion_parity = np.zeros(len(alpha_strings), dtype=np.int8)
    beta_occupied_below = np.zeros(len(alpha_strings), dtype=np.int64)
    for orbital in range(int(n_spatial_orbitals)):
        alpha_occupied = (alpha_strings >> orbital) & 1
        inversion_parity ^= (
            alpha_occupied & (beta_occupied_below & 1)
        ).astype(np.int8, copy=False)
        beta_occupied_below += (beta_strings >> orbital) & 1
    return np.where(inversion_parity == 0, 1.0, -1.0)


def restricted_cisd_vector_to_sparse_qubit_state(
    cisd_vector,
    n_spatial_orbitals: int,
    n_occupied_spatial_orbitals: int,
    *,
    coefficient_tolerance: float = 0.0,
) -> SparseQubitState:
    """Expand a restricted PySCF CISD vector directly into sparse JW form.

    Parameters
    ----------
    cisd_vector
        One-dimensional spin-adapted vector returned by ``pyscf.ci.CISD``.
        It contains the reference, restricted singles, and restricted doubles
        amplitudes in PySCF's packed CISD convention.
    n_spatial_orbitals
        Number of active spatial orbitals represented by ``cisd_vector``.
    n_occupied_spatial_orbitals
        Number of doubly occupied active reference orbitals.
    coefficient_tolerance
        Determinant amplitudes with magnitude at or below this threshold are
        omitted after the exact packed-to-determinant expansion.

    Returns
    -------
    state
        ``SparseQubitState`` on ``2 * n_spatial_orbitals`` interleaved-spin
        Jordan--Wigner qubits. Qubit ``2*p`` is alpha orbital ``p`` and qubit
        ``2*p+1`` is beta orbital ``p``. No FCI-sized dense array is created.

    Notes
    -----
    This reproduces ``pyscf.ci.cisd.to_fcivec`` only at its nonzero CISD
    addresses. Its memory use is proportional to the number of CISD
    determinants rather than to the full-CI determinant-space dimension.
    """
    from pyscf.ci import cisd
    from pyscf.fci import cistring

    nmo = int(n_spatial_orbitals)
    nocc = int(n_occupied_spatial_orbitals)
    if not 0 < nocc <= nmo:
        raise ValueError("n_occupied_spatial_orbitals must lie in [1, nmo]")
    vector = np.asarray(cisd_vector).reshape(-1)
    expected_size = 1 + nocc * (nmo - nocc) + (
        nocc * nocc * (nmo - nocc) * (nmo - nocc)
    )
    if vector.size != expected_size:
        raise ValueError(
            f"CISD vector has size {vector.size}; expected {expected_size}"
        )

    c0, c1, c2 = cisd.cisdvec_to_amplitudes(
        vector, nmo, nocc, copy=False
    )
    t1addr, t1sign = cisd.tn_addrs_signs(nmo, nocc, 1)
    t1addr = np.asarray(t1addr, dtype=np.int64)
    t1sign = np.asarray(t1sign)

    alpha_addresses = [np.asarray([0], dtype=np.int64)]
    beta_addresses = [np.asarray([0], dtype=np.int64)]
    coefficients = [np.asarray([c0])]

    singles = c1.reshape(-1) * t1sign
    alpha_addresses.extend((t1addr, np.zeros_like(t1addr)))
    beta_addresses.extend((np.zeros_like(t1addr), t1addr))
    coefficients.extend((singles, singles))

    opposite_spin = c2.transpose(0, 2, 1, 3).reshape(
        nocc * (nmo - nocc), -1
    )
    opposite_spin = np.einsum(
        "i,j,ij->ij", t1sign, t1sign, opposite_spin
    )
    alpha_addresses.append(
        np.repeat(t1addr, len(t1addr)).astype(np.int64, copy=False)
    )
    beta_addresses.append(np.tile(t1addr, len(t1addr)))
    coefficients.append(opposite_spin.reshape(-1))

    if nocc > 1 and nmo - nocc > 1:
        same_spin = c2 - c2.transpose(1, 0, 2, 3)
        occupied_pairs = np.tril_indices(nocc, -1)
        virtual_pairs = np.tril_indices(nmo - nocc, -1)
        same_spin = same_spin[occupied_pairs][
            :, virtual_pairs[0], virtual_pairs[1]
        ]
        t2addr, t2sign = cisd.tn_addrs_signs(nmo, nocc, 2)
        t2addr = np.asarray(t2addr, dtype=np.int64)
        t2coeff = same_spin.reshape(-1) * np.asarray(t2sign)
        alpha_addresses.extend((t2addr, np.zeros_like(t2addr)))
        beta_addresses.extend((np.zeros_like(t2addr), t2addr))
        coefficients.extend((t2coeff, t2coeff))

    alpha_addresses = np.concatenate(alpha_addresses)
    beta_addresses = np.concatenate(beta_addresses)
    coefficients = np.concatenate(coefficients).astype(
        np.complex128, copy=False
    )
    keep = np.abs(coefficients) > float(coefficient_tolerance)
    alpha_addresses = alpha_addresses[keep]
    beta_addresses = beta_addresses[keep]
    coefficients = coefficients[keep]

    unique_addresses = np.unique(
        np.concatenate((alpha_addresses, beta_addresses))
    )
    strings = cistring.addrs2str(nmo, nocc, unique_addresses)
    address_to_string = {
        int(address): int(string)
        for address, string in zip(unique_addresses, strings)
    }
    alpha_strings = np.fromiter(
        (address_to_string[int(address)] for address in alpha_addresses),
        dtype=np.int64,
        count=len(alpha_addresses),
    )
    beta_strings = np.fromiter(
        (address_to_string[int(address)] for address in beta_addresses),
        dtype=np.int64,
        count=len(beta_addresses),
    )
    indices = _spin_strings_to_interleaved_jw_indices(
        alpha_strings, beta_strings, nmo
    )
    coefficients *= _pyscf_to_interleaved_jw_phases(
        alpha_strings, beta_strings, nmo
    )
    return SparseQubitState(
        indices,
        coefficients,
        n_qubits=2 * nmo,
        drop_tol=float(coefficient_tolerance),
    )


def run_restricted_cisd_from_molecular_data(
    molecule,
    *,
    frozen_core_orbitals: int = 0,
    convergence_tolerance: float = 1e-10,
    max_cycle: int = 100,
    coefficient_tolerance: float = 0.0,
    verbose: int = 0,
) -> tuple[float, SparseQubitState, dict]:
    """Run RHF-CISD and return its wavefunction without an FCI allocation.

    Parameters
    ----------
    molecule
        SCF-processed OpenFermion ``MolecularData`` containing canonical
        orbitals, orbital energies, and the Hartree--Fock energy.
    frozen_core_orbitals
        Number of lowest doubly occupied spatial orbitals excluded from the
        active qubit state and treated as a frozen core by PySCF CISD.
    convergence_tolerance
        PySCF CISD residual convergence tolerance.
    max_cycle
        Maximum number of PySCF CISD iterations.
    coefficient_tolerance
        Determinant amplitudes at or below this magnitude are omitted from the
        returned sparse state. The default retains every nonzero CISD term.
    verbose
        PySCF output verbosity.

    Returns
    -------
    energy, state, metadata
        Total CISD energy in Hartree; normalized active-space interleaved-JW
        ``SparseQubitState``; and calculation metadata including convergence,
        determinant count, raw expanded norm, and active-space dimensions.

    Notes
    -----
    The saved canonical molecular orbitals are installed into a reconstructed
    PySCF RHF object. This keeps the CISD determinant basis aligned with the
    OpenFermion Hamiltonian while avoiding a repeated orbital optimization.
    """
    from pyscf import ci, gto, scf

    ncore = int(frozen_core_orbitals)
    nmo_full = int(molecule.n_orbitals)
    nelec_full = int(molecule.n_electrons)
    if nelec_full % 2:
        raise ValueError("restricted closed-shell CISD requires even electrons")
    if not 0 <= ncore < nelec_full // 2:
        raise ValueError("invalid frozen_core_orbitals for occupied space")
    if ncore >= nmo_full:
        raise ValueError("frozen_core_orbitals removes every orbital")

    pyscf_molecule = gto.M(
        atom=molecule.geometry,
        basis=molecule.basis,
        charge=int(molecule.charge),
        spin=int(molecule.multiplicity) - 1,
        unit="Angstrom",
        verbose=int(verbose),
    )
    mean_field = scf.RHF(pyscf_molecule)
    mean_field.mo_coeff = np.asarray(molecule.canonical_orbitals)
    mean_field.mo_energy = np.asarray(molecule.orbital_energies)
    mean_field.mo_occ = np.zeros(nmo_full)
    mean_field.mo_occ[: nelec_full // 2] = 2.0
    mean_field.e_tot = float(molecule.hf_energy)
    mean_field.converged = True

    solver = ci.CISD(mean_field, frozen=(ncore if ncore else None))
    solver.conv_tol = float(convergence_tolerance)
    solver.max_cycle = int(max_cycle)
    correlation_energy, cisd_vector = solver.kernel()
    if not solver.converged:
        raise RuntimeError("PySCF restricted CISD did not converge")

    active_nmo = nmo_full - ncore
    active_nocc = nelec_full // 2 - ncore
    state = restricted_cisd_vector_to_sparse_qubit_state(
        cisd_vector,
        active_nmo,
        active_nocc,
        coefficient_tolerance=coefficient_tolerance,
    )
    raw_norm = state.norm()
    state.normalize()
    energy = float(mean_field.e_tot + correlation_energy)
    return energy, state, {
        "method": "pyscf_restricted_cisd_sparse_expansion",
        "converged": bool(solver.converged),
        "correlation_energy": float(correlation_energy),
        "energy": energy,
        "full_spatial_orbitals": nmo_full,
        "frozen_core_orbitals": ncore,
        "active_spatial_orbitals": active_nmo,
        "active_occupied_spatial_orbitals": active_nocc,
        "active_electrons": 2 * active_nocc,
        "n_qubits": 2 * active_nmo,
        "packed_cisd_vector_size": int(np.asarray(cisd_vector).size),
        "sparse_determinants": int(state.nnz),
        "expanded_norm_before_normalization": raw_norm,
        "coefficient_tolerance": float(coefficient_tolerance),
        "jw_phase_convention": "interleaved_spin_orbital_v1",
        "fci_layout_materialized": False,
    }
