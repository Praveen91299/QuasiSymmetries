### hct number of symmetry dependence:
from quasisymmetries.metrics import get_ent, variance, comm_sq_exp_fast
from quasisymmetries.sym import hct_mod, bs_hct
import matplotlib.pyplot as plt
import pickle
from openfermion import MolecularData, jordan_wigner, count_qubits, get_sparse_operator

directory = './saved/hamiltonians/'
system = 'H2O_corr'

with open(directory+system+".pkl", "rb") as f:
    data = pickle.load(f)
H, fci_e, fci_gs, cisd_e, cisd_gs = data
molecule = MolecularData(filename=directory+system)
HQ = jordan_wigner(H)
n_qubits = count_qubits(HQ)
Hs = get_sparse_operator(HQ, n_qubits)

comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_gs, n_qubits)
comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, fci_gs, n_qubits)
var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)
var_fci = lambda s_list: variance(s_list, fci_gs, n_qubits)

sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized
sym_metric_func = lambda s: (-1)*sym_group_score_func([s]) # HCT minimized

method = 'bshct'

if method == 'hct':
    hct_sym, _ = hct_mod(HQ, n_qubits, sym_metric_func=sym_metric_func, use_coeffs_eps=True)

ent_dict = {}
n_sym_list = list(range(n_qubits))
bw=16
for n_sym in n_sym_list:
    print(n_sym)
    if method=='bshct':
        bs_hct_res = bs_hct(HQ, n_sym, beam_width=bw, sym_metric_func=sym_metric_func, use_coeffs_eps=True)
        sym_list = bs_hct_res[0].syms
    if method == 'hct':
        sym_list = hct_sym[:n_sym]
    
    ent, _ = get_ent(sym_list, HQ, n_qubits, log_base='e')
    ent_dict[n_sym] = ent

import pandas as pd

# x values used in your plot
x = list(range(1, n_qubits))

# Make table:
# rows = MPS bond index
# columns = n_sym
ent_df = pd.DataFrame(ent_dict, index=x)

# Clean labels
ent_df.index.name = "MPS Bond Index"
ent_df.columns.name = "n_sym"

# Save as CSV

filename = './saved/{}_{}_nc_exp_cisd_entanglement_by_nsym'.format(system, method)
ent_df.to_csv(filename+'.csv')

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(6.5, 4.5))

cmap = LinearSegmentedColormap.from_list(
    "black_to_blue",
    ["#000000", "#0072B2"]  # black to color-blind friendly blue
)

# Color-blind friendly sequential colormap
cmap = plt.cm.cividis

# Normalize n_sym values onto the colormap
norm = mpl.colors.Normalize(
    vmin=min(n_sym_list),
    vmax=max(n_sym_list)
)

for n_sym, ent in ent_dict.items():
    ax.plot(
        x,
        ent,
        color=cmap(norm(n_sym)),
        linewidth=2
    )

ax.set_xlabel("MPS Bond Index", fontsize=14)
ax.set_ylabel(r"Bipartite entanglement $S_{vN}$", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(x)

# Colorbar replacing the legend
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.03)
cbar.set_label(r"$n_{\mathrm{sym}}$", fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.tick_params(labelsize=12)
fig.tight_layout()

plt.savefig(filename+'.pdf', dpi=500)
plt.show()