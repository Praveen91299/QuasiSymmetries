"""Run the full symmetry benchmark workflow."""

import pickle

import pandas as pd
import quimb.tensor as qtn
from openfermion import MolecularData, count_qubits, get_sparse_operator, jordan_wigner

from quasisymmetries.benchmark import BenchmarkData, benchmark_syms
from quasisymmetries.bs.beam import find_commuting_symmetry_generators
from quasisymmetries.metrics import comm_sq_exp_fast, variance
from quasisymmetries.sym import (
    bs_hct,
    get_quartic_symmetries,
    get_seniority_symmetries,
    hct_mod,
)
from quasisymmetries.tn import find_dmrg_conv_bd_quimb

__all__ = ["BenchmarkData", "benchmark_syms", "main"]


def main():
    directory = "./saved/hamiltonians/"

    systems = [
        "H4chain_eqm",
        "H4chain_corr",
        "H4chain_diss",
        "H4rect_corr",
        "H4rect_diss",
        "LiH_eqm",
        "LiH_corr",
        "H2O_eqm",
        "H2O_corr",
        "H2O_diss",
        "N2frozen_eqm",
        "N2frozen_corr",
        "N2frozen_diss",
    ]
    bd_rows = []

    for system in systems:
        print(f"Starting system: {system}")
        date = "_MAY26"
        cost_func_tag = "_nc_exp_cisd"
        output_filename = "./saved/" + cost_func_tag + date
        with open(output_filename, "a") as file_obj:
            print("\n\n" + system, file=file_obj)

        with open(directory + system + ".pkl", "rb") as file_obj:
            data = pickle.load(file_obj)
        H, fci_e, fci_gs, cisd_e, cisd_gs = data
        MolecularData(filename=directory + system)
        HQ = jordan_wigner(H)
        n_qubits = count_qubits(HQ)
        Hs = get_sparse_operator(HQ, n_qubits)

        comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(
            s_list, Hs, cisd_gs, n_qubits
        )
        var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)

        sym_group_score_func = lambda s_list: -comm_sq_exp_cisd(s_list)
        sym_group_var_func = lambda s_list: -var_cisd(s_list)
        sym_metric_func = lambda symmetry: -sym_group_score_func([symmetry])

        cost_functions = {
            "Comm": sym_group_score_func,
            "Var": sym_group_var_func,
            "1-norm": None,
        }

        beam_width = 16
        n_sym = n_qubits // 2
        sym_hct_N_2, _ = hct_mod(
            HQ,
            n_sym,
            use_coeffs_eps=True,
            sym_metric_func=sym_metric_func,
        )
        sym_hct_N, _ = hct_mod(
            HQ,
            n_qubits,
            use_coeffs_eps=True,
            sym_metric_func=sym_metric_func,
        )
        sym_sen = get_seniority_symmetries(n_qubits)
        sym_quar = get_quartic_symmetries(n_qubits)[: n_sym - 1]
        sym_bs_hct_N_2 = bs_hct(
            HQ,
            n_sym,
            beam_width=beam_width,
            sym_metric_func=sym_metric_func,
            use_coeffs_eps=True,
        )[0].syms
        sym_bs_hct_N = bs_hct(
            HQ,
            n_qubits,
            beam_width=beam_width,
            sym_metric_func=sym_metric_func,
            use_coeffs_eps=True,
        )[0].syms

        print(
            f"\nBeam Search ({beam_width}) with exact-symmetry seeding:"
        )
        cost_function = "Comm"
        print(f"Starting cost function: {cost_function}")

        sym_bs_N_2 = find_commuting_symmetry_generators(
            HQ,
            target_rank=n_sym,
            beam_width=beam_width,
            heavy_core_fraction=0.95,
            include_pairwise_products=True,
            pairwise_seed_terms=12,
            seed_with_exact_symmetries=True,
            score_func=cost_functions[cost_function],
            include_hct_symmetries=True,
            hct_n_sym=n_qubits // 2,
            hct_use_coeffs_eps=True,
            score_is_separable=True,
        )
        sym_bs_N = find_commuting_symmetry_generators(
            HQ,
            target_rank=n_qubits,
            beam_width=beam_width,
            heavy_core_fraction=0.95,
            include_pairwise_products=True,
            pairwise_seed_terms=12,
            seed_with_exact_symmetries=True,
            score_func=cost_functions[cost_function],
            include_hct_symmetries=True,
            hct_n_sym=n_qubits,
            hct_use_coeffs_eps=True,
            score_is_separable=True,
        )

        benchmark_inputs = [
            ("BS N/2", sym_bs_N_2, True),
            ("BS N", sym_bs_N, False),
            ("HCT N/2", sym_hct_N_2, True),
            ("HCT N", sym_hct_N, False),
            ("SEN N/2", sym_sen, True),
            ("QUAR N/2", sym_quar, True),
            ("BS-HCT N/2", sym_bs_hct_N_2, True),
            ("BS-HCT N", sym_bs_hct_N, False),
        ]
        datasets = [
            benchmark_syms(
                symmetries,
                HQ,
                fci_gs,
                fci_e,
                n_qubits,
                N_2_sym=n_2_sym,
                print_to_file=output_filename,
                tag=f"{tag} {cost_function}",
                verbose=False,
            )
            for tag, symmetries, n_2_sym in benchmark_inputs
        ]

        save_filename = output_filename + system + "_datasets"
        BenchmarkData.save_datasets(datasets, save_filename)
        BenchmarkData.plot_cut_entropies(
            datasets,
            fci_gs,
            output_filename + system + "_cutentropy.png",
        )

        compress_cutoff = 1e-20
        gs_mps = qtn.MatrixProductState.from_dense(
            fci_gs,
            cutoff=compress_cutoff,
        )
        dmrg_bd, _ = find_dmrg_conv_bd_quimb(
            HQ,
            n_qubits,
            fci_e,
            tol=1.6e-3,
            n_sweeps=100,
            reps=1,
            verbose=False,
            compress_cutoff=compress_cutoff,
            sweep_tol=1e-6,
            noise=1e0,
            bsz=2,
            guess_mps=gs_mps,
            seed=0,
        )

        columns = ["system", "Original"] + [data.tag for data in datasets]
        values = [system, dmrg_bd] + [data.dmrg_bd for data in datasets]
        bd_rows.append(dict(zip(columns, values)))
        pd.DataFrame(bd_rows).to_csv(
            output_filename + "_dmrg_bd.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
