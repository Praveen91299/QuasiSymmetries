"""Benchmark fermionic and qubit MPO bond dimensions."""

import pickle

import numpy as np
import pandas as pd
from openfermion import MolecularData, count_qubits, jordan_wigner

from quasisymmetries.benchmark import BenchmarkData
from quasisymmetries.fiedler import do_fiedler_reordering
from quasisymmetries.metrics import get_permuted_bipartite_entanglement
from quasisymmetries.mpo import (
    build_qc_mpo_from_openfermion_molecule,
    infer_largest_mpo_bond_dimension,
)
from quasisymmetries.tn import MPO_from_QubitOperator

__all__ = [
    "build_qc_mpo_from_openfermion_molecule",
    "infer_largest_mpo_bond_dimension",
    "main",
]


def main():
    directory = "./saved/hamiltonians/"
    systems = [
        "H4chain_eqm",
        "H4chain_corr",
        "H4chain_diss",
        "H4rect_corr",
        "H4rect_diss",
        "LIH_eqm",
        "LIH_corr",
        "H2O_eqm",
        "H2O_corr",
        "H2O_diss",
        "N2frozen_eqm",
        "N2frozen_corr",
        "N2frozen_diss",
    ]
    bond_sizes = {}
    bd_rows = []
    symmetry_indices = [1, 3, 4]  # BS N, HCT N, seniority

    for system in systems:
        system_bond_sizes = {}
        is_n2 = system.startswith("N2")

        filename = directory + system
        with open(filename + ".pkl", "rb") as file_obj:
            H, fci_e, fci_gs, cisd_e, cisd_gs = pickle.load(file_obj)
        molecule = MolecularData(filename=filename)

        HQ = jordan_wigner(H)
        n_qubits = count_qubits(H)

        out = build_qc_mpo_from_openfermion_molecule(
            molecule,
            ncore=2 if is_n2 else 0,
            iprint=0,
        )
        print(
            "Largest MPO bond dimension:",
            out["largest_mpo_bond_dim"],
        )
        system_bond_sizes["fermionic"] = out["largest_mpo_bond_dim"]

        mpo = MPO_from_QubitOperator(
            HQ,
            None,
            mpo_cutoff=1e-20,
            compression_freq=20,
            verbose=True,
        )
        system_bond_sizes["qubit"] = max(mpo.bond_sizes())

        datasets = BenchmarkData.load_datasets(
            "./saved/results/MAY27/"
            f"_nc_exp_cisd_MAY27{system}_datasets"
        )
        for index in symmetry_indices:
            symdata = datasets[index]
            print(f"Importing {symdata.tag} symmetries for {system}:")
            print(symdata.symmetries)

            entropies, H_perm, clifford, gs_rot = (
                get_permuted_bipartite_entanglement(
                    symdata.symmetries,
                    HQ,
                    n_qubits,
                    fci_energy=fci_e,
                    fci_gs=fci_gs,
                    verbose=True,
                    return_state=True,
                    return_clifford=True,
                    log_base=np.e,
                )
            )
            mpo = MPO_from_QubitOperator(
                H_perm,
                None,
                mpo_cutoff=1e-20,
                compression_freq=20,
                verbose=False,
            )
            system_bond_sizes[symdata.tag] = max(mpo.bond_sizes())

            ent_reord, H_reord, psi_reord, fiedler_info = (
                do_fiedler_reordering(
                    H_perm,
                    gs_rot,
                    n_qubits=n_qubits,
                    verbose=True,
                    log_base=np.e,
                )
            )
            mpo = MPO_from_QubitOperator(
                H_reord,
                None,
                mpo_cutoff=1e-20,
                compression_freq=20,
                verbose=False,
            )
            system_bond_sizes[symdata.tag + " + fiedler"] = max(
                mpo.bond_sizes()
            )

        bond_sizes[system] = system_bond_sizes
        print("\n\n" + "#" * 50)
        print(system)
        print(system_bond_sizes)
        print("\n\n" + "#" * 50)

        columns = ["system", *system_bond_sizes]
        values = [system, *system_bond_sizes.values()]
        bd_rows.append(dict(zip(columns, values)))
        pd.DataFrame(bd_rows).to_csv(
            "./saved/beam_hct_mpo_bd_Jun24.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
