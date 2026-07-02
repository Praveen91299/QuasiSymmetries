from __future__ import annotations
### July 1, repeated Beam search on h2o/n2

#load data
import pickle
import numpy as np
from pathlib import Path
from openfermion import MolecularData, jordan_wigner, count_qubits
from mpo_bds import build_qc_mpo_from_openfermion_molecule
from src.bs.beam import *
from benchmark_all import BenchmarkData, benchmark_syms



from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openfermion import QubitOperator, get_sparse_operator, expectation
from src.bs.utils import *
from src.clifford_symmetry_optimized import (
    inverse_conjugate_qubit_operator_by_clifford_factors_exact,
    invert_permutation,
    permute_qubits_in_qubit_operator,
)
from src.metrics import (
    comm_sq_exp_fast,
    comm_sq_exp_sparse_syms,
    get_permuted_bipartite_entanglement,
    prepare_sparse_symmetries,
)
from copy import deepcopy


def inverse_transform_symmetries_to_original_frame(
    symmetries,
    frame_transforms,
    n_qubits,
):
    """Map Pauli symmetries from the current repeated-search frame to frame 0."""
    original_frame_symmetries = [deepcopy(sym) for sym in symmetries]

    for transform in reversed(frame_transforms):
        inverse_perm = invert_permutation(transform["permutation"])
        original_frame_symmetries = [
            permute_qubits_in_qubit_operator(sym, inverse_perm)
            for sym in original_frame_symmetries
        ]
        original_frame_symmetries = [
            inverse_conjugate_qubit_operator_by_clifford_factors_exact(
                sym,
                transform["parsed_gates"],
                n_qubits=n_qubits,
            )
            for sym in original_frame_symmetries
        ]

    return original_frame_symmetries


def BeamSearch_Symmetries_rep(
    hamiltonian: QubitOperator,
    cisd_gs,
    cisd_e,
    fci_gs,
    fci_e,
    *,
    reps: int = 1,
    target_rank: int = None,
    n_qubits: Optional[int] = None,
    beam_width: int = 16,
    heavy_core_fraction: float = 0.95,
    max_candidates_from_terms: Optional[int] = 256,
    include_hct_symmetries: bool = True,
    hct_n_sym: Optional[int] = None,
    hct_use_coeffs_eps: bool = True,
    include_pairwise_products: bool = False,
    pairwise_seed_terms: int = 24,
    max_pauli_weight: Optional[int] = None,
    do_local_refine: bool = True,
    local_refine_passes: int = 10,
    seed_with_exact_symmetries: bool = False,
    max_exact_symmetry_seeds: Optional[int] = None,
    benchmark_output_file: Optional[str] = None,
) -> List[QubitOperator]:
    """
    Beam search for exact and approximate symmetries

    Main workflow:
      1. Convert Hamiltonian to binary symplectic form
      2. Restrict candidate generator pool
      3. Keep a heavy core of the Hamiltonian
      4. Run beam search on the heavy core
      5. Optionally refine by local swaps on the full Hamiltonian

    Optional:
      - seed the search with exact Pauli symmetries of the Hamiltonian
    """

    #build pool
    seed_generators: Optional[List[QubitOperator]] = None

    if seed_with_exact_symmetries:
        exact_syms = exact_pauli_symmetry_basis(hamiltonian, n_qubits=n_qubits)
        if max_exact_symmetry_seeds is not None:
            exact_syms = exact_syms[:max_exact_symmetry_seeds]
        seed_generators = exact_syms

    n_qubits, terms = qubit_operator_terms(hamiltonian, n_qubits)
    
    candidate_pool = build_candidate_pool_hct(
        terms,
        n_qubits,
        max_candidates_from_terms=max_candidates_from_terms,
        include_pairwise_products=include_pairwise_products,
        pairwise_seed_terms=pairwise_seed_terms,
        max_pauli_weight=max_pauli_weight,
        include_hct_symmetries = include_hct_symmetries,
        hct_n_sym = hct_n_sym,
        hct_use_coeffs_eps = hct_use_coeffs_eps,
    )
    current_hamiltonian = deepcopy(hamiltonian)
    current_cisd = deepcopy(cisd_gs)
    current_fci = deepcopy(fci_gs)
    benchmark_datasets: List[BenchmarkData] = []
    frame_transforms = []

    print("Starting search...")
    for r in range(reps):
        print("Repeat: ", r)

        current_hamiltonian_sparse = get_sparse_operator(current_hamiltonian, n_qubits)

        print(expectation(current_hamiltonian_sparse, current_cisd))
        score_func = lambda s_list: (-1)*comm_sq_exp_fast(s_list, current_hamiltonian_sparse, current_cisd, n_qubits)

        syms = beam_search_symmetries(
            current_hamiltonian,
            candidate_pool,
            target_rank=target_rank,
            n_qubits=n_qubits,
            beam_width=beam_width,
            heavy_core_fraction=heavy_core_fraction,
            initial_generators=seed_generators,
            score_func=score_func
        )
        

        if do_local_refine:
            syms = local_swap_refine(
                current_hamiltonian,
                syms,
                candidate_pool,
                n_qubits=n_qubits,
                max_passes=local_refine_passes,
                score_func=score_func
            )

        print(syms)
        score_before_rotation = score_func(syms)
        print("Score before rotation:", score_before_rotation)
        syms_sparse = prepare_sparse_symmetries(syms, n_qubits)
        original_frame_syms = inverse_transform_symmetries_to_original_frame(
            syms,
            frame_transforms,
            n_qubits,
        )
        print("Symmetries in the original frame:")
        for sym in original_frame_syms:
            print(sym)

        if benchmark_output_file is not None:
            benchmark_data, processed_data = benchmark_syms(
                syms,
                current_hamiltonian,
                current_fci,
                fci_e,
                n_qubits,
                N_2_sym=(len(syms) == n_qubits // 2),
                verbose=True,
                tag=f"Repeated beam search iteration {r + 1}",
                return_processed_data=True,
                log_base=np.e,
            )
            benchmark_data.write_to_file(benchmark_output_file)
            benchmark_datasets.append(benchmark_data)
            ent = benchmark_data.cut_entropies
            current_hamiltonian = processed_data["H_perm"]
            U = processed_data["U"]
            current_fci = processed_data["gs_rot"]
            clifford_info = processed_data["clifford_info"]
        else:
            # Transform without running the more expensive DMRG benchmark.
            ent, current_hamiltonian, U, current_fci, clifford_info = get_permuted_bipartite_entanglement(
                syms,
                current_hamiltonian,
                n_qubits,
                fci_e,
                current_fci,
                True,
                True,
                True,
                'e',
                False,
                return_clifford_info=True,
            )

        if benchmark_output_file is not None:
            with open(benchmark_output_file, "a") as f:
                print("Symmetries in original frame:", file=f)
                for sym in original_frame_syms:
                    print(sym, file=f)
                print("Forward Clifford factors for next frame:", file=f)
                for factor in clifford_info["factor_descriptions"]:
                    print(f"  {factor}", file=f)
                print(
                    f"Qubit permutation: {clifford_info['permutation']}",
                    file=f,
                )

        frame_transforms.append(clifford_info)

        # Transform the state and the original symmetry generators in the same
        # direction as the Hamiltonian: A -> U A U^\dagger.
        current_cisd = U @ current_cisd
        U_dag = U.getH()
        rotated_hamiltonian_sparse = get_sparse_operator(current_hamiltonian, n_qubits).tocsr()

        def rotated_sparse_score_func(s_list_sparse):
            rotated_syms_sparse = [
                (U @ sym_sparse @ U_dag).tocsr()
                for sym_sparse in s_list_sparse
            ]
            return (-1)*comm_sq_exp_sparse_syms(
                rotated_syms_sparse,
                rotated_hamiltonian_sparse,
                current_cisd,
            )

        score_after_rotation = rotated_sparse_score_func(syms_sparse)
        print("Score after rotation:", score_after_rotation)

        if not np.isclose(score_before_rotation, score_after_rotation):
            raise AssertionError(
                "Score changed under the unitary rotation: "
                f"{score_before_rotation} -> {score_after_rotation}"
            )

    return syms

directory = "./saved/hamiltonians/"
system = "H4chain_corr"
benchmark_directory = Path("./saved/results/JULY02")
benchmark_output_file = benchmark_directory / f"{system}_rep_bs_benchmark.txt"
benchmark_directory.mkdir(parents=True, exist_ok=True)
with benchmark_output_file.open("w") as f:
    print(f"{system} repeated beam-search benchmarks", file=f)

is_n2 = True if system[:2] == 'N2' else False
        
filename = directory+system 
with open(filename+".pkl", "rb") as f:
    data = pickle.load(f)
mol = MolecularData(filename=filename)
H, fci_e, fci_gs, cisd_e, cisd_gs = data

HQ = jordan_wigner(H)
n_qubits = count_qubits(H)
print(fci_e)

#repeat search and transform
syms = BeamSearch_Symmetries_rep(
    HQ,
    cisd_gs=cisd_gs,
    cisd_e=cisd_e,
    fci_gs=fci_gs,
    fci_e=fci_e,
    reps=10,
    target_rank=n_qubits,
    benchmark_output_file=str(benchmark_output_file),
)
