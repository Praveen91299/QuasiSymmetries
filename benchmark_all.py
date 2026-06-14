# %%
# %%
from __future__ import annotations
# benchmark scripts
import pickle
import numpy as np
#from src.state_utils import get_cisd_gs, get_fci_state_openfermion
#from src.op_utils import build_H_chain_for_R, h2o_geometry
# hct
#from src.state_utils import get_hf_occ, get_hf_wfn
from dataclasses import dataclass, field, fields
from pathlib import Path

@dataclass
class BenchmarkData:
    tag: str = ''
    symmetries: list[QubitOperator] = field(default_factory=list)
    non_commuting_l1: float = 0
    num_commuting_terms: int = 0
    sym_entropy: float = 0
    cut_entropies: list[float] = field(default_factory=list)
    dmrg_bd: int = 0
    single_sector_e: float = 0

    @staticmethod
    def _with_pkl_suffix(filename):
        path = Path(filename)
        return path if path.suffix == ".pkl" else path.with_suffix(path.suffix + ".pkl")

    def to_dict(self):
        """
        Return all dataclass fields as a plain dictionary.
        """
        return {data_field.name: getattr(self, data_field.name) for data_field in fields(self)}

    @classmethod
    def from_dict(cls, data):
        """
        Build a BenchmarkData object from a saved attributes dictionary.
        """
        field_names = {data_field.name for data_field in fields(cls)}
        return cls(**{name: data[name] for name in field_names if name in data})

    @classmethod
    def _from_saved_payload(cls, payload):
        if isinstance(payload, cls):
            return payload

        if isinstance(payload, dict) and payload.get("__type__") == cls.__name__:
            return cls.from_dict(payload["attributes"])

        if isinstance(payload, dict):
            field_names = {data_field.name for data_field in fields(cls)}
            if field_names.intersection(payload):
                return cls.from_dict(payload)

        raise TypeError(f"Cannot convert saved payload of type {type(payload).__name__} to {cls.__name__}")

    @classmethod
    def _pickle_load(cls, file_obj):
        class BenchmarkDataUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__" and name == cls.__name__:
                    return cls
                return super().find_class(module, name)

        return BenchmarkDataUnpickler(file_obj).load()

    def __str__(self):
        lines = [f"{self.__class__.__name__}:"]
        for data_field in fields(self):
            value = getattr(self, data_field.name)
            if isinstance(value, list):
                lines.append(f"{data_field.name}:")
                if value:
                    lines.extend(f"  {item}" for item in value)
                else:
                    lines.append("  []")
            else:
                lines.append(f"{data_field.name}: {value}")
        return "\n".join(lines)

    def write_to_file(self, filename):
        ent_str = "\n".join([val.__str__() for val in self.cut_entropies])
        with open(filename, 'a') as f:
            print(self.tag, file=f)
            print("Symmetries:", file=f)
            print("\n".join([sym.__str__() for sym in self.symmetries]), file=f)
            print("Non-commutator L1: ", self.non_commuting_l1, file=f)
            print("Entropy: ", self.sym_entropy, file=f)
            print("Commuting terms: ", self.num_commuting_terms, file=f)
            print("Cut entropies:\n", ent_str, file=f)
            print("DMRG conv BD: ", self.dmrg_bd, file=f)
            print("Single sector energy: ", self.single_sector_e, file=f)
    
    def save(self, filename):
        """
        Save all BenchmarkData attributes as a pickle payload.
        """
        payload = {
            "__type__": self.__class__.__name__,
            "attributes": self.to_dict(),
        }
        with open(self._with_pkl_suffix(filename), "wb") as f:
            pickle.dump(payload, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Import one saved BenchmarkData file.
        """
        with open(cls._with_pkl_suffix(filename), "rb") as f:
            payload = cls._pickle_load(f)
        return cls._from_saved_payload(payload)
    
    @classmethod
    def save_datasets(cls, datasets: list[BenchmarkData], filename):
        """
        Save datasets into a single file, preserving every entry and all attributes.
        """
        payload = {
            "__type__": f"{cls.__name__}.datasets",
            "datasets": [data.to_dict() for data in datasets],
        }
        with open(cls._with_pkl_suffix(filename), "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_datasets(cls, filename):
        """
        Import a saved datasets file.

        This also accepts the older tag-keyed dictionary format, but returns a
        list so duplicate tags are not lost in newly saved files.
        """
        with open(cls._with_pkl_suffix(filename), "rb") as f:
            payload = cls._pickle_load(f)

        if isinstance(payload, dict) and payload.get("__type__") == f"{cls.__name__}.datasets":
            return [cls.from_dict(data) for data in payload["datasets"]]

        if isinstance(payload, dict) and payload.get("__type__") == cls.__name__:
            return [cls._from_saved_payload(payload)]

        if isinstance(payload, list):
            return [cls._from_saved_payload(data) for data in payload]

        if isinstance(payload, dict):
            return [cls._from_saved_payload(data) for data in payload.values()]

        return [cls._from_saved_payload(payload)]

    @classmethod
    def view_saved_files(cls, filenames):
        """
        Import and print saved BenchmarkData file(s).
        """
        if isinstance(filenames, (str, Path)):
            filenames = [filenames]

        outputs = []
        for filename in filenames:
            datasets = cls.load_datasets(filename)
            outputs.append(f"{filename}:")
            outputs.extend(str(data) for data in datasets)

        view = "\n\n".join(outputs)
        print(view)
        return view

    @classmethod
    def view_saved_file(cls, filename):
        """
        Import and print one saved BenchmarkData file.
        """
        return cls.view_saved_files([filename])


    @classmethod
    def plot_cut_entropies(cls, datasets: list[BenchmarkData], gs=None, filename: str = None):
        """
        Save and return plot for entropies.
        """
        import matplotlib.pyplot as plt
        from src.metrics import get_entropies_at_cuts

        fig, ax = plt.subplots()

        for data in datasets:
            n_qubits = len(data.cut_entropies) + 1
            x = range(1, n_qubits)
            ax.plot(x, data.cut_entropies, label=data.tag)
        
        if gs is not None:
            #reference state
            n_qubits = int(np.log2(len(gs)))
            gs_ent = get_entropies_at_cuts(gs, n_qubits)
            x = range(1, n_qubits)
            ax.plot(x, gs_ent, label="Reference")

        ax.legend()
        ax.set_xlabel("MPS Bond Index", fontsize=14)
        ax.set_ylabel(r"Bipartite entanglement $S_{vN}$", fontsize=14)
        ax.set_xticks(x, x)

        if filename is not None: fig.savefig(filename, dpi=300, bbox_inches="tight")

        return fig

def benchmark_syms(list_syms, HQ, fci_gs, fci_e, n_qubits, N_2_sym=False, verbose=True, print_to_file=None, tag="",
                   compress_cutoff = 1e-10):
    """
    Run all benchmarks for symmetries

    """
    import quimb.tensor as qtn
    from src.metrics import (
        entropy_pauli_syms,
        find_commuting_paulis,
        get_permuted_bipartite_entanglement,
        get_single_sector_energies,
        universal_grading,
    )
    from src.tn import find_dmrg_conv_bd_quimb

    print(tag)
    nc_l1 = universal_grading(list_syms, HQ, verbose=verbose)
    c = len(find_commuting_paulis(HQ, list_syms, verbose=verbose))

    ent, H_perm, U, gs_rot = get_permuted_bipartite_entanglement(list_syms, HQ, n_qubits, fci_e, fci_gs, verbose, True, True, 'e', False)
    
    gs_rot_mps = qtn.MatrixProductState.from_dense(gs_rot, cutoff = 1e-20)     
    dmrg_bd, _ = find_dmrg_conv_bd_quimb(H_perm, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                        reps=1, verbose=False, compress_cutoff = 1e-20, sweep_tol = 1e-6,
                        noise = 1e0, bsz=2, guess_mps = gs_rot_mps, seed=0)

    #ent and dmrg
    if N_2_sym:
        ent_N_2 = entropy_pauli_syms(list_syms, fci_gs, n_qubits, verbose=verbose)
        ss_energies = get_single_sector_energies(HQ, list_syms, n_qubits, verbose=verbose)
        ss_e = np.min(ss_energies)
        #N/2 syms, single sector, BO energies TODO K and BO energies, but they are not really relevant here

        data = BenchmarkData(tag=tag, symmetries=list_syms, non_commuting_l1 = nc_l1, num_commuting_terms=c,  sym_entropy=ent_N_2, cut_entropies=ent, dmrg_bd=dmrg_bd, single_sector_e=ss_e)
    else:
        data = BenchmarkData(tag=tag, symmetries=list_syms, non_commuting_l1 = nc_l1, num_commuting_terms=c, cut_entropies=ent, dmrg_bd=dmrg_bd)
    
    if print_to_file is not None:
        data.write_to_file(print_to_file)

    return data

def main():
    import pandas as pd
    from openfermion import count_qubits, jordan_wigner, get_sparse_operator, MolecularData
    from src.bs.beam import find_commuting_symmetry_generators
    from src.metrics import comm_sq_exp_fast, variance
    from src.sym import get_quartic_symmetries, get_seniority_symmetries, hct_mod, bs_hct

    directory = "./saved/hamiltonians/"

    systems = [
        'H4chain_eqm',
        'H4chain_corr',
        'H4chain_diss',
        'H4rect_corr',
        'H4rect_diss',
        'LiH_eqm',
        'LiH_corr',
        'H2O_eqm',
        'H2O_corr',
        'H2O_diss',
        'N2frozen_eqm',
        'N2frozen_corr',
        'N2frozen_diss',
        ]
    bd_rows = []

    for system in systems:
        print(f'Starting system: {system}')
        filename= system

        date="_MAY26" #to keep track of outputs
        cost_func_tag = '_nc_exp_cisd'
        output_filename = "./saved/" + cost_func_tag + date
        with open(output_filename, 'a') as f:
            print('\n\n' + system, file=f)
        
        with open(directory+system+".pkl", "rb") as f:
            data = pickle.load(f)
        H, fci_e, fci_gs, cisd_e, cisd_gs = data
        molecule = MolecularData(filename=directory+system)
        HQ = jordan_wigner(H)
        n_qubits = count_qubits(HQ)
        Hs = get_sparse_operator(HQ, n_qubits)

        #state specific cost functions
        comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_gs, n_qubits)
        comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, fci_gs, n_qubits)
        var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)
        var_fci = lambda s_list: variance(s_list, fci_gs, n_qubits)

        sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized
        sym_group_var_func = lambda s_list: (-1)*var_cisd(s_list) # BS score maximized
        sym_metric_func = lambda s: (-1)*sym_group_score_func([s]) # HCT minimized

        cf_dict = {'Comm': sym_group_score_func, 'Var' : sym_group_var_func, '1-norm': None}

        #make symmetries
        bw=16 # beam width for bs-hct and bs
        n_sym = n_qubits//2
        sym_hct_N_2, eps = hct_mod(HQ, n_sym, use_coeffs_eps=True, sym_metric_func=sym_metric_func)
        sym_hct_N, eps = hct_mod(HQ, n_qubits, use_coeffs_eps=True, sym_metric_func=sym_metric_func)
        sym_sen = get_seniority_symmetries(n_qubits)
        sym_quar = get_quartic_symmetries(n_qubits)[:n_sym-1]
        bs_hct_N_2 = bs_hct(HQ, n_sym, beam_width=bw, sym_metric_func=sym_metric_func, use_coeffs_eps=True)
        sym_bs_hct_N_2 = bs_hct_N_2[0].syms
        bs_hct_N = bs_hct(HQ, n_qubits, beam_width=bw, sym_metric_func=sym_metric_func, use_coeffs_eps=True)
        sym_bs_hct_N = bs_hct_N[0].syms

        print("\nBeam Search ({}) with exact-symmetry seeding:".format(bw))

        datasets = []

        cost_function =  'Comm'

        print(f'Starting cosf function: {cost_function}')

        sym_bs_N_2 = find_commuting_symmetry_generators(
            HQ,
            target_rank=n_sym,
            beam_width=bw,
            heavy_core_fraction=0.95,
            include_pairwise_products=True,
            pairwise_seed_terms=12,
            seed_with_exact_symmetries=True,
            score_func= cf_dict[cost_function], # this function maximizes the cost function TODO invert this
            include_hct_symmetries = True,
            hct_n_sym = n_qubits//2,
            hct_use_coeffs_eps = True,
        )

        sym_bs_N = find_commuting_symmetry_generators(
            HQ,
            target_rank=n_qubits,
            beam_width=bw,
            heavy_core_fraction=0.95,
            include_pairwise_products=True,
            pairwise_seed_terms=12,
            seed_with_exact_symmetries=True,
            score_func= cf_dict[cost_function], # this function maximizes the cost function TODO invert this
            include_hct_symmetries = True,
            hct_n_sym = n_qubits,
            hct_use_coeffs_eps = True,
        )

        datasets.append(benchmark_syms(sym_bs_N_2, HQ, fci_gs, fci_e, 
                                        n_qubits, N_2_sym=True, print_to_file=output_filename, 
                                        tag="BS N/2" + f" {cost_function}", verbose = False))
        
        datasets.append(benchmark_syms(sym_bs_N, HQ, fci_gs, fci_e, 
                                        n_qubits, N_2_sym=False, print_to_file=output_filename, 
                                        tag="BS N" + f" {cost_function}",verbose = False))

        #diagnostics
        datasets.append(benchmark_syms(sym_hct_N_2, HQ, fci_gs, fci_e,
                                        n_qubits, N_2_sym=True, print_to_file=output_filename,
                                        tag="HCT N/2" + f" {cost_function}", verbose=False))

        datasets.append(benchmark_syms(sym_hct_N, HQ, fci_gs, fci_e,
                                        n_qubits, N_2_sym=False, print_to_file=output_filename,
                                        tag="HCT N" + f" {cost_function}", verbose=False))

        datasets.append(benchmark_syms(sym_sen, HQ, fci_gs, fci_e,
                                        n_qubits, N_2_sym=True, print_to_file=output_filename,
                                        tag="SEN N/2" + f" {cost_function}", verbose=False))

        datasets.append(benchmark_syms(sym_quar, HQ, fci_gs, fci_e,
                                        n_qubits, N_2_sym=True, print_to_file=output_filename,
                                        tag="QUAR N/2" + f" {cost_function}", verbose=False))

        datasets.append(benchmark_syms(sym_bs_hct_N_2, HQ, fci_gs, fci_e,
                                        n_qubits, N_2_sym=True, print_to_file=output_filename,
                                        tag="BS-HCT N/2" + f" {cost_function}", verbose=False))

        datasets.append(benchmark_syms(sym_bs_hct_N, HQ, fci_gs, fci_e,
                                        n_qubits, N_2_sym=False, print_to_file=output_filename,
                                        tag="BS-HCT N" + f" {cost_function}", verbose=False))

        #save data objects
        save_filename = output_filename + system + "_datasets"
        BenchmarkData.save_datasets(datasets, save_filename)
        
        #analysis
        #entropy graphs
        _ = BenchmarkData.plot_cut_entropies(datasets, fci_gs, output_filename + system + "_cutentropy.png")

        #dmrg bds
        #un rotated DMRG (as paulis)
        compress_cutoff = 1e-20
        import quimb.tensor as qtn
        from src.tn import find_dmrg_conv_bd_quimb
        gs_mps = qtn.MatrixProductState.from_dense(fci_gs, cutoff = compress_cutoff)     
        dmrg_bd, _ = find_dmrg_conv_bd_quimb(HQ, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                            reps=1, verbose=False, compress_cutoff = compress_cutoff, sweep_tol = 1e-6,
                            noise = 1e0, bsz=2, guess_mps = gs_mps, seed=0)

        cols = ["system"] + ["Original"] + [data.tag for data in datasets]
        bd_rows.append(dict(zip(cols, [system] + [dmrg_bd] + [data.dmrg_bd for data in datasets])))
        df = pd.DataFrame(bd_rows)
        df.to_csv(output_filename + "_dmrg_bd.csv", index=False)


if __name__ == "__main__":
    main()
# %%
