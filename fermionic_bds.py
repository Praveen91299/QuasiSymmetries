from pathlib import Path

import pandas as pd

from test_mpo_ferm import benchmark_system


systems = [
    "H4chain_corr",
    "H4chain_diss",
    "H8cube_eqm",
    "H8cube_3A"
]

output_csv = Path("./saved/results/JUNE29/_nc_exp_cisd_JUNE29_fermionic_bd.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)

rows = []
for system in systems:
    data = benchmark_system(system)
    rows.append({
        "system": system,
        "Fermionic MPO": data["mpo_bond_dimension"],
        "Fermionic MPS": data["mps_converged_bond_dimension"],
        "converged": data["converged"],
    })
    pd.DataFrame(rows).to_csv(output_csv, index=False)

print("Saved fermionic bond dimensions to:", output_csv)
