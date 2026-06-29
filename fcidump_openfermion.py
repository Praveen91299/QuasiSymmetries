"""
Convert FCIDUMP files to OpenFermion objects.

FCIDUMP files contain integrals and electron/spin counts, but usually not
geometry or basis metadata. The returned MolecularData object is therefore a
useful OpenFermion integral container, not a recoverable molecule specification.
"""

from pathlib import Path

import numpy as np
from openfermion import MolecularData, get_fermion_operator
from pyscf import ao2mo
from pyscf.tools import fcidump


def _check_fcidump_file(fcidump_path):
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

    return fcidump_path


def read_fcidump_integrals(
    fcidump_path,
    molpro_orbsym=False,
    verbose=False,
):
    """
    Read an FCIDUMP file and return OpenFermion-compatible spatial integrals.
    """
    fcidump_path = _check_fcidump_file(fcidump_path)
    data = fcidump.read(
        str(fcidump_path),
        molpro_orbsym=molpro_orbsym,
        verbose=verbose,
    )

    n_orbitals = int(data["NORB"])
    n_electrons = int(data["NELEC"])
    ms2 = int(data.get("MS2", data.get("MS", 0)))
    ecore = float(data.get("ECORE", 0.0))

    one_body_integrals = np.asarray(data["H1"], dtype=float)
    two_body_integrals = np.asarray(
        ao2mo.restore(1, data["H2"], n_orbitals),
        dtype=float,
    )

    return {
        "n_orbitals": n_orbitals,
        "n_electrons": n_electrons,
        "ms2": ms2,
        "ecore": ecore,
        "one_body_integrals": one_body_integrals,
        "two_body_integrals": two_body_integrals,
        "raw": data,
    }


def molecular_data_from_fcidump(
    fcidump_path,
    filename=None,
    description=None,
    molpro_orbsym=False,
    verbose=False,
):
    """
    Build an OpenFermion MolecularData container from an FCIDUMP file.
    """
    fcidump_path = Path(fcidump_path)
    ints = read_fcidump_integrals(
        fcidump_path,
        molpro_orbsym=molpro_orbsym,
        verbose=verbose,
    )

    ms2 = ints["ms2"]
    multiplicity = abs(ms2) + 1
    if description is None:
        description = fcidump_path.stem

    molecule = MolecularData(
        geometry="FCIDUMP",
        basis="unknown",
        multiplicity=multiplicity,
        charge=0,
        description=description,
        filename="" if filename is None else filename,
    )

    molecule.n_orbitals = ints["n_orbitals"]
    molecule.n_qubits = 2 * ints["n_orbitals"]
    molecule.n_electrons = ints["n_electrons"]
    molecule.nuclear_repulsion = ints["ecore"]
    molecule.one_body_integrals = ints["one_body_integrals"]
    molecule.two_body_integrals = ints["two_body_integrals"]

    return molecule


def fermion_operator_from_fcidump(
    fcidump_path,
    molpro_orbsym=False,
    verbose=False,
):
    """
    Build an OpenFermion FermionOperator from an FCIDUMP file.
    """
    molecule = molecular_data_from_fcidump(
        fcidump_path,
        molpro_orbsym=molpro_orbsym,
        verbose=verbose,
    )
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    return get_fermion_operator(molecular_hamiltonian)
