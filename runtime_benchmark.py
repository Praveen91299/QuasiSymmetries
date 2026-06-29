def runtime_factors(n_s, d, chi_mps, chi_mpo):
    t1, t2 = n_s * chi_mpo * (chi_mps **3) * (d ** 2), n_s * (chi_mpo **2) * (chi_mps **2) * (d ** 3)
    return max(t1, t2), (t1, t2)

def memory_factors(n_s, d, chi_mps, chi_mpo):
    m1, m2 = n_s * chi_mpo * (chi_mps**2),  n_s * (chi_mpo **2) * (d**2)
    return max(m1, m2), (m1, m2)

systems = [
    'H4chain_eqm',
    'H4chain_corr',
    'H4chain_diss',
    'H4rect_corr',
    'H4rect_diss',
    'LIH_eqm',
    'LIH_corr',
    'H2O_eqm',
    'H2O_corr',
    'H2O_diss',
    'N2frozen_eqm',
    'N2frozen_corr',
    'N2frozen_diss'
]

import pandas as pd

ferm_data = pd.read_csv('./saved/results/fermionic_mpo_mps_bond_dimensions_test.csv')
qubit_bs_fd_mps_bd = {
    'H4chain_eqm': 3,
    'H4chain_corr': 3,
    'H4chain_diss': 2,
    'H4rect_corr': 3,
    'H4rect_diss': 2,
    'LIH_eqm': 2,
    'LIH_corr': 3,
    'H2O_eqm': 8,
    'H2O_corr': 5,
    'H2O_diss': 1,
    'N2frozen_eqm': 14,
    'N2frozen_corr': 12,
    'N2frozen_diss': 8
}

qub_data = pd.read_csv('./saved/beam_hct_mpo_bd_Jun24.csv')

old_qub_data = pd.read_csv('./saved/results/MAY27/_nc_exp_cisd_MAY27_dmrg_bd.csv')
qubit_bs_mps_bd = {system: bd for system, bd in zip(old_qub_data["system"], old_qub_data["BS N Comm"])}

qubit_bs_fd_mpo_bd = {system: bd for system, bd in zip(qub_data["system"], qub_data["BS N Comm + fiedler"])}
qubit_bs_mpo_bd = {system: bd for system, bd in zip(qub_data["system"], qub_data["BS N Comm"])}


ferm_mps_bd = {system: bd for system, bd in zip(ferm_data["system"], ferm_data["mps_converged_bond_dimension"])}
ferm_mpo_bd = {system: bd for system, bd in zip(ferm_data["system"], ferm_data["mpo_bond_dimension"])}


qubit_counts = {
    'H4chain_eqm': 8,
    'H4chain_corr': 8,
    'H4chain_diss': 8,
    'H4rect_corr': 8,
    'H4rect_diss': 8,
    'LIH_eqm': 12,
    'LIH_corr': 12,
    'H2O_eqm': 14,
    'H2O_corr': 14,
    'H2O_diss': 14,
    'N2frozen_eqm': 16,
    'N2frozen_corr': 16,
    'N2frozen_diss': 16
}

cost_rows = []
for system in systems:
    costs = {}
    print(system)
    n_qubits = qubit_counts[system]
    
    T = runtime_factors(n_qubits//2, 4, ferm_mps_bd[system], ferm_mpo_bd[system])[0]
    M = memory_factors(n_qubits//2, 4, ferm_mps_bd[system], ferm_mpo_bd[system])[0]
    costs["T Ferm"] = T
    costs["M Ferm"] = M

    T = runtime_factors(n_qubits, 2, qubit_bs_mps_bd[system], qubit_bs_mpo_bd[system])[0]
    M = memory_factors(n_qubits, 2, qubit_bs_mps_bd[system], qubit_bs_mpo_bd[system])[0]
    costs["T BS"] = T
    costs["M BS"] = M

    T = runtime_factors(n_qubits, 2, qubit_bs_fd_mps_bd[system], qubit_bs_fd_mpo_bd[system])[0]
    M = memory_factors(n_qubits, 2, qubit_bs_fd_mps_bd[system], qubit_bs_fd_mpo_bd[system])[0]
    costs["T BS + fiedler"] = T
    costs["M BS + fiedler"] = M

    cols = ["system"] + [x for x in costs.keys()]
    cost_rows.append(dict(zip(cols, [system] + [float(x) for x in costs.values()])))
    df = pd.DataFrame(cost_rows)
    output_filename = "./saved/beam_costs"
    df.to_csv(output_filename + "_Jun24.csv", float_format="{:.1e}".format, index=False)