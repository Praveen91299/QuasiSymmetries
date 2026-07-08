import csv
import pickle
from pathlib import Path

from openfermion import count_qubits, jordan_wigner

from quasisymmetries.benchmark import BenchmarkData
from quasisymmetries.metrics import l1norm
from quasisymmetries.op_utils import separate_H


ham_dir = Path("./saved/hamiltonians")
dataset_dir = Path("./saved/results/MAY27")
output_file = Path("./saved/results/JUL06/sep_norms_l1.csv")

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

symmetry_sets = [
    ("BS (N/2)", "BS N/2 Comm"),
    ("HCT (N/2)", "HCT N/2 Comm"),
    ("Seniority", "SEN N/2 Comm"),
]


rows = []

for system in systems:
    with open(ham_dir / f"{system}.pkl", "rb") as file_obj:
        H, fci_e, fci_gs, cisd_e, cisd_gs = pickle.load(file_obj)

    HQ = jordan_wigner(H)
    n_qubits = count_qubits(HQ)

    datasets = BenchmarkData.load_datasets(
        str(dataset_dir / f"_nc_exp_cisd_MAY27{system}_datasets")
    )
    datasets_by_tag = {data.tag: data for data in datasets}

    print(f"\n{system}")

    for sym_label, dataset_tag in symmetry_sets:
        symmetries = datasets_by_tag[dataset_tag].symmetries
        Z0, V0, V, _U = separate_H(HQ, symmetries, n_qubits)

        Z0_l1 = l1norm(Z0, remove_const=True)
        V0_l1 = l1norm(V0, remove_const=True)
        V_l1 = l1norm(V, remove_const=True)

        print(f"{sym_label:10s}: Z0={Z0_l1:.12g}, V0={V0_l1:.12g}, V={V_l1:.12g}")

        rows.append(
            {
                "system": system,
                "symmetry_class": sym_label,
                "dataset_tag": dataset_tag,
                "n_qubits": n_qubits,
                "n_symmetries": len(symmetries),
                "fci_energy": fci_e,
                "Z0_l1": Z0_l1,
                "V0_l1": V0_l1,
                "V_l1": V_l1,
                "total_l1": Z0_l1 + V0_l1 + V_l1,
            }
        )

output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} rows to {output_file}")
