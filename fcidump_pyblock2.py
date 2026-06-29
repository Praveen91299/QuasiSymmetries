"""
Small helpers for running PyBlock2 DMRG directly from an FCIDUMP file.

Example
-------
from fcidump_pyblock2 import run_pyblock2_dmrg_from_fcidump

result = run_pyblock2_dmrg_from_fcidump(
    "FCIDUMP",
    max_bond_dim=500,
    n_sweeps=20,
)
print(result["energy"])
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes


def _set_block2_random_seed(seed):
    """Set Block2's RNG seed when the installed version exposes it."""
    if seed is None:
        return

    try:
        from block2 import Random

        Random.rand_seed(seed)
    except Exception:
        pass


def _default_sweep_schedule(max_bond_dim, n_sweeps):
    """
    Conservative finite-DMRG schedule for a single random initial MPS.
    """
    max_bond_dim = int(max_bond_dim)
    n_sweeps = int(n_sweeps)
    warmup_dim = max(4, min(max_bond_dim, max_bond_dim // 2))

    bond_dims = [warmup_dim] * min(4, n_sweeps)
    bond_dims += [max_bond_dim] * max(0, n_sweeps - len(bond_dims))

    noises = (
        [1.0e-4] * min(4, n_sweeps)
        + [1.0e-5] * min(4, max(0, n_sweeps - 4))
        + [1.0e-6] * min(4, max(0, n_sweeps - 8))
    )
    noises += [0.0] * max(0, n_sweeps - len(noises))

    thrds = [1.0e-9] * min(4, n_sweeps)
    thrds += [1.0e-10] * min(8, max(0, n_sweeps - 4))
    thrds += [1.0e-11] * max(0, n_sweeps - len(thrds))

    return bond_dims[:n_sweeps], noises[:n_sweeps], thrds[:n_sweeps]


def load_fcidump_as_pyblock2_mpo(
    fcidump_path,
    symm_type=SymmetryTypes.SU2,
    pg="d2h",
    scratch=None,
    n_threads=None,
    n_mkl_threads=1,
    stack_mem=int(2 * 1024**3),
    read_iprint=1,
    mpo_iprint=1,
):
    """
    Load an FCIDUMP file into a PyBlock2 driver and build its QC MPO.

    Returns a dict containing ``driver``, ``mpo``, metadata copied from the
    FCIDUMP, and ``_scratch_obj`` when a temporary scratch directory was made.
    Keep the returned dict alive while using the driver/MPO.
    """
    fcidump_path = Path(fcidump_path)
    with fcidump_path.open("r", errors="ignore") as f:
        first_line = f.readline().strip()
    if first_line.startswith("version https://git-lfs.github.com/spec/"):
        raise ValueError(
            f"{fcidump_path} is a Git LFS pointer, not the FCIDUMP data. "
            "Fetch the large file with `git lfs pull` or use a path to the "
            "downloaded FCIDUMP."
        )
    if "&FCI" not in first_line.upper():
        raise ValueError(
            f"{fcidump_path} does not look like an FCIDUMP file. "
            f"First line was: {first_line!r}"
        )

    if n_threads is None:
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))

    if scratch is None:
        scratch_obj = tempfile.TemporaryDirectory(prefix="pyblock2_fcidump_")
        scratch_path = Path(scratch_obj.name)
    else:
        scratch_obj = None
        scratch_path = Path(scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)

    driver = DMRGDriver(
        scratch=str(scratch_path),
        symm_type=symm_type,
        n_threads=n_threads,
        n_mkl_threads=n_mkl_threads,
        stack_mem=stack_mem,
    )

    driver.read_fcidump(str(fcidump_path), pg=pg, iprint=read_iprint)
    driver.initialize_system(
        n_sites=driver.n_sites,
        n_elec=driver.n_elec,
        spin=driver.spin,
        pg_irrep=driver.pg_irrep,
        orb_sym=driver.orb_sym,
    )

    g2e = driver.g2e
    unpack_g2e = not (isinstance(g2e, np.ndarray) and g2e.ndim == 1)
    mpo = driver.get_qc_mpo(
        h1e=driver.h1e,
        g2e=g2e,
        ecore=driver.ecore,
        unpack_g2e=unpack_g2e,
        symmetrize=unpack_g2e,
        iprint=mpo_iprint,
    )

    return {
        "driver": driver,
        "mpo": mpo,
        "n_sites": driver.n_sites,
        "n_elec": driver.n_elec,
        "spin": driver.spin,
        "pg_irrep": driver.pg_irrep,
        "orb_sym": driver.orb_sym,
        "ecore": driver.ecore,
        "h1e": driver.h1e,
        "g2e": driver.g2e,
        "scratch": scratch_path,
        "_scratch_obj": scratch_obj,
    }


def run_pyblock2_dmrg_from_fcidump(
    fcidump_path,
    max_bond_dim=500,
    n_sweeps=20,
    bond_dims=None,
    noises=None,
    thrds=None,
    seed=None,
    ket_tag="KET",
    dav_max_iter=50,
    dmrg_iprint=1,
    **load_kwargs,
):
    """
    Load FCIDUMP integrals, build the PyBlock2 MPO, and run ground-state DMRG.

    Extra keyword arguments are forwarded to ``load_fcidump_as_pyblock2_mpo``.
    """
    _set_block2_random_seed(seed)
    out = load_fcidump_as_pyblock2_mpo(fcidump_path, **load_kwargs)
    driver = out["driver"]
    mpo = out["mpo"]

    if bond_dims is None or noises is None or thrds is None:
        default_bond_dims, default_noises, default_thrds = (
            _default_sweep_schedule(max_bond_dim, n_sweeps)
        )
        bond_dims = default_bond_dims if bond_dims is None else bond_dims
        noises = default_noises if noises is None else noises
        thrds = default_thrds if thrds is None else thrds

    n_sweeps = int(n_sweeps)
    ket = driver.get_random_mps(
        tag=ket_tag,
        bond_dim=int(max_bond_dim),
        nroots=1,
    )

    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=n_sweeps,
        bond_dims=list(bond_dims)[:n_sweeps],
        noises=list(noises)[:n_sweeps],
        thrds=list(thrds)[:n_sweeps],
        dav_max_iter=dav_max_iter,
        iprint=dmrg_iprint,
    )

    out["ket"] = ket
    out["energy"] = float(energy)
    out["bond_dims"] = list(bond_dims)[:n_sweeps]
    out["noises"] = list(noises)[:n_sweeps]
    out["thrds"] = list(thrds)[:n_sweeps]
    return out


def cleanup_pyblock2_result(result):
    """Remove the temporary scratch directory owned by a returned result dict."""
    scratch_obj = result.get("_scratch_obj")
    if scratch_obj is not None:
        scratch_obj.cleanup()
        result["_scratch_obj"] = None


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fcidump")
    parser.add_argument("--max-bond-dim", type=int, default=500)
    parser.add_argument("--n-sweeps", type=int, default=20)
    parser.add_argument("--scratch", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--iprint", type=int, default=1)
    args = parser.parse_args()

    result = run_pyblock2_dmrg_from_fcidump(
        args.fcidump,
        max_bond_dim=args.max_bond_dim,
        n_sweeps=args.n_sweeps,
        scratch=args.scratch,
        n_threads=args.threads,
        read_iprint=args.iprint,
        mpo_iprint=args.iprint,
        dmrg_iprint=args.iprint,
    )
    print(f"DMRG energy: {result['energy']:.15f}")


if __name__ == "__main__":
    _main()
