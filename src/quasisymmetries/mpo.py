### MPO bond dimenions

### June 8, 2026

import os
import tempfile
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes


def _safe_len(x):
    if x is None:
        return None

    for attr in ("size", "n", "m", "n_rows", "n_cols"):
        if hasattr(x, attr):
            try:
                v = getattr(x, attr)
                if callable(v):
                    v = v()
                if isinstance(v, int):
                    return v
            except Exception:
                pass

    try:
        return len(x)
    except Exception:
        pass

    try:
        shape = x.shape
        if len(shape) > 0:
            return max(shape)
    except Exception:
        pass

    return None


def infer_largest_mpo_bond_dimension(mpo, verbose=True):
    """
    Best-effort extraction of the largest MPO virtual bond dimension.

    PyBlock2/Block2 object internals vary across versions, so this checks
    several likely attributes.
    """
    candidates = []

    for name in ("bond_dims", "bond_dim", "bond_dimensions", "dims"):
        if hasattr(mpo, name):
            try:
                obj = getattr(mpo, name)
                obj = obj() if callable(obj) else obj

                if isinstance(obj, int):
                    candidates.append((name, obj))
                else:
                    vals = list(obj)
                    if vals:
                        candidates.append((name, max(int(v) for v in vals)))
            except Exception:
                pass

    for name in (
        "left_operator_names",
        "right_operator_names",
        "middle_operator_names",
    ):
        if hasattr(mpo, name):
            try:
                arr = getattr(mpo, name)
                local_dims = []

                for item in arr:
                    d = _safe_len(item)
                    if d is not None:
                        local_dims.append(d)

                if local_dims:
                    candidates.append((name, max(local_dims)))
            except Exception:
                pass

    # Common when the returned MPO is a wrapper around prim_mpo.
    if hasattr(mpo, "prim_mpo"):
        try:
            inner = getattr(mpo, "prim_mpo")
            inner_dim = infer_largest_mpo_bond_dimension(inner, verbose=False)
            if inner_dim is not None:
                candidates.append(("prim_mpo", inner_dim))
        except Exception:
            pass

    if not candidates:
        if verbose:
            print("Could not infer MPO bond dimension from this MPO object.")
            print("Try setting iprint=2 or iprint=3 when constructing the MPO.")
        return None

    largest = max(v for _, v in candidates)

    if verbose:
        print("\nDetected MPO bond-dimension candidates:")
        for name, value in sorted(candidates, key=lambda x: str(x[0])):
            print(f"  {name:30s}: {value}")
        print(f"\nLargest detected MPO bond dimension: {largest}")

    return largest


def _get_openfermion_integrals(molecule):
    """
    Return h1e, g2e from an openfermion.MolecularData object.

    Prefer get_integrals(), but fall back to attributes.
    """
    try:
        h1e, g2e = molecule.get_integrals()
    except Exception:
        h1e = molecule.one_body_integrals
        g2e = molecule.two_body_integrals

    if h1e is None or g2e is None:
        raise ValueError(
            "MolecularData does not contain integrals. "
            "Make sure you have already run openfermionpyscf.run_pyscf "
            "with integrals computed."
        )

    return np.asarray(h1e, dtype=float), np.asarray(g2e, dtype=float)


def _openfermion_to_block2_g2e(g2e):
    """
    Convert OpenFermion's spatial two-electron tensor to the chemists'
    notation expected by DMRGDriver.get_qc_mpo().

    OpenFermion's tensor enters

        1/2 * g[p,q,r,s] a_p^+ a_q^+ a_r a_s,

    whereas Block2's spatial tensor ``G[i,j,k,l]`` enters

        1/2 * G[i,j,k,l] a_i^+ a_k^+ a_l a_j.

    Therefore G[i,j,k,l] = g[i,k,l,j].
    """
    return np.asarray(g2e, dtype=float).transpose(0, 3, 1, 2).copy()


def build_qc_mpo_from_openfermion_molecule(
    molecule,
    ncore=0,
    active_orbitals=None,
    symm_type=SymmetryTypes.SU2,
    n_threads=None,
    stack_mem=int(2 * 1024**3),
    scratch=None,
    iprint=2,
    orb_sym=None,
):
    """
    Construct a PyBlock2 quantum-chemistry MPO from an OpenFermion MolecularData
    object that has already been processed by openfermionpyscf.run_pyscf.

    Parameters
    ----------
    molecule :
        openfermion.MolecularData object.
    ncore : int
        Number of lowest spatial orbitals to freeze as doubly occupied.
        Example: N2/STO-3G frozen core -> ncore=2.
    active_orbitals : list[int] or None
        Spatial orbital indices to keep active, in the OpenFermion MO ordering.
        If None, uses all orbitals except the first ncore orbitals.
    symm_type :
        Usually SymmetryTypes.SU2 for closed-shell/singlet DMRG.
    scratch : str, Path, or None
        Scratch directory. If None, creates a temporary directory and keeps it
        alive by returning the TemporaryDirectory object.
    orb_sym : list[int] or None
        Orbital symmetry labels. OpenFermion MolecularData usually does not
        carry Block2-compatible point-group labels, so None defaults to all 1s.

    Returns
    -------
    out : dict
        Contains driver, mpo, active-space integrals, active electron count,
        and largest detected MPO bond dimension.
    """
    if n_threads is None:
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))

    h1e_full, g2e_full = _get_openfermion_integrals(molecule)

    n_orb_total = h1e_full.shape[0]

    if h1e_full.shape != (n_orb_total, n_orb_total):
        raise ValueError(f"Bad one-body integral shape: {h1e_full.shape}")

    if g2e_full.shape != (n_orb_total, n_orb_total, n_orb_total, n_orb_total):
        raise ValueError(f"Bad two-body integral shape: {g2e_full.shape}")

    if active_orbitals is None:
        active_orbitals = list(range(ncore, n_orb_total))
    else:
        active_orbitals = list(active_orbitals)

    core_orbitals = list(range(ncore))

    if set(core_orbitals) & set(active_orbitals):
        raise ValueError("Core and active orbital sets overlap.")

    if any(i < 0 or i >= n_orb_total for i in core_orbitals + active_orbitals):
        raise ValueError("Core or active orbital index out of range.")

    n_elec_total = int(molecule.n_electrons)
    n_elec_active = n_elec_total - 2 * len(core_orbitals)

    if n_elec_active < 0:
        raise ValueError("ncore freezes more electrons than the molecule has.")

    # Let OpenFermion perform its own frozen-core contraction. This avoids
    # accidentally applying chemists'-notation formulas to OpenFermion's
    # differently ordered tensor.
    if core_orbitals:
        core_adjustment, h1e, g2e_openfermion = (
            molecule.get_active_space_integrals(
                occupied_indices=core_orbitals,
                active_indices=active_orbitals,
            )
        )
    else:
        core_adjustment = 0.0
        h1e = h1e_full[np.ix_(active_orbitals, active_orbitals)].copy()
        g2e_openfermion = g2e_full[
            np.ix_(
                active_orbitals,
                active_orbitals,
                active_orbitals,
                active_orbitals,
            )
        ].copy()

    ecore = (
        float(getattr(molecule, "nuclear_repulsion", 0.0))
        + float(core_adjustment)
    )
    g2e = _openfermion_to_block2_g2e(g2e_openfermion)

    ncas = len(active_orbitals)

    # PyBlock2 wants spin target as 2S, not PySCF's N_alpha - N_beta.
    # For the common closed-shell ground-state case, use S = 0.
    #
    # If you need open-shell states, pass a MolecularData object with known
    # multiplicity and this maps multiplicity = 2S + 1 -> spin_target = 2S.
    multiplicity = int(getattr(molecule, "multiplicity", 1))
    spin_target = multiplicity - 1

    if orb_sym is None:
        # OpenFermion MolecularData usually lacks Block2-compatible point group
        # orbital irreps. Use trivial symmetry labels.
        orb_sym = [1] * ncas

    if scratch is None:
        scratch_obj = tempfile.TemporaryDirectory(prefix="pyblock2_of_mpo_")
        scratch_path = Path(scratch_obj.name)
    else:
        scratch_obj = None
        scratch_path = Path(scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)

    driver = DMRGDriver(
        scratch=str(scratch_path),
        symm_type=symm_type,
        n_threads=n_threads,
        stack_mem=stack_mem,
    )

    driver.initialize_system(
        n_sites=ncas,
        n_elec=n_elec_active,
        spin=spin_target,
        orb_sym=orb_sym,
    )

    mpo = driver.get_qc_mpo(
        h1e=h1e,
        g2e=g2e,
        ecore=ecore,
        iprint=iprint,
    )

    largest_mpo_bond_dim = infer_largest_mpo_bond_dimension(mpo, verbose=True)

    print("\nOpenFermion -> PyBlock2 MPO summary:")
    print(f"  name                 : {getattr(molecule, 'name', '<unknown>')}")
    print(f"  total spatial orbs   : {n_orb_total}")
    print(f"  frozen core orbs     : {core_orbitals}")
    print(f"  active orbitals      : {active_orbitals}")
    print(f"  active space         : CAS({n_elec_active}e, {ncas}o)")
    print(f"  spin target 2S       : {spin_target}")
    print(f"  ecore                : {ecore:.12f}")
    print(f"  scratch              : {scratch_path}")

    return {
        "driver": driver,
        "mpo": mpo,
        "ncas": ncas,
        "n_elec": n_elec_active,
        "spin": spin_target,
        "ecore": ecore,
        "h1e": h1e,
        "g2e": g2e,
        "orb_sym": orb_sym,
        "active_orbitals": active_orbitals,
        "core_orbitals": core_orbitals,
        "largest_mpo_bond_dim": largest_mpo_bond_dim,
        "_scratch_obj": scratch_obj,  # keeps temp dir alive if scratch=None
    }


def cleanup_qc_mpo_result(result):
    """Remove the temporary scratch directory owned by an MPO result.

    A caller-supplied scratch directory is not owned and is never removed.
    Returns ``True`` when an owned temporary directory was cleaned.
    """
    scratch_obj = result.get("_scratch_obj")
    if scratch_obj is None:
        return False
    scratch_obj.cleanup()
    result["_scratch_obj"] = None
    return True
