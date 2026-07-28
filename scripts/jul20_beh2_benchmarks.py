"""Generate and benchmark BeH2/STO-3G Clifford symmetry data for JUL20.

The systems are linear symmetric H--Be--H geometries:

* BeH2_eqm:  R(Be--H) = 1.326 Angstrom
* BeH2_corr: R(Be--H) = 2.000 Angstrom
* BeH2_diss: R(Be--H) = 4.000 Angstrom

Outputs are written in the same style as the JUL08 Clifford/Fiedler script:
human-readable text, portable JSON, BenchmarkData JSON collections, and DMRG/MPO
bond-dimension CSVs.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import quimb.tensor as qtn
from openfermion import (
    MolecularData,
    count_qubits,
    get_fermion_operator,
    get_ground_state,
    get_sparse_operator,
    jordan_wigner,
)
from openfermionpyscf import run_pyscf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from july08_clifford_benchmarks import (  # noqa: E402
    MPO_from_QubitOperator,
    as_jsonable,
    bliss_info_summary,
    find_dmrg_conv_bd_quimb,
    memory_factors,
    runtime_factors,
    save_json,
    sym_strings,
    write_csv,
    write_entropies,
)
from quasisymmetries.benchmark import BenchmarkData  # noqa: E402
from quasisymmetries.bliss import lp_bliss_paper_real_pauli_1norm  # noqa: E402
from quasisymmetries.bs.beam import (  # noqa: E402
    BeamSearch_Symmetries,
    validate_symmetry_generators,
)
from quasisymmetries.clifford_symmetry_optimized import (  # noqa: E402
    permute_qubits_in_qubit_operator,
)
from quasisymmetries.fiedler import (  # noqa: E402
    do_fiedler_reordering,
    invert_ordering,
    reorder_statevector_axes,
)
from quasisymmetries.metrics import (  # noqa: E402
    comm_sq_exp_fast,
    entropy_pauli_syms,
    find_commuting_paulis,
    get_entropies_at_cuts,
    get_permuted_bipartite_entanglement,
    universal_grading,
)
from quasisymmetries.state_utils import get_cisd_gs, get_hf_occ  # noqa: E402
from quasisymmetries.sym import hct_mod  # noqa: E402


def as_jsonable(value):
    if isinstance(value, np.ndarray):
        return [as_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return as_jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(val) for val in value]
    return value


def save_json(path: Path, data) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        json.dump(as_jsonable(data), file_obj, indent=2, allow_nan=False)
        file_obj.write("\n")


SYSTEM_GEOMETRIES = {
    "BeH2_eqm": 1.326,
    "BeH2_corr": 2.000,
    "BeH2_diss": 4.000,
}

SYNTHESIS_BASIS = "Z"
GENERATOR_MAPPING = "positive_z"
LOG_BASE = np.e
BASIS = "sto-3g"
MULTIPLICITY = 1
CHARGE = 0

HCT_N2_TAG = "HCT N/2 Comm"
HCT_N_TAG = "HCT N Comm"
BLISS_HCT_TAG = "Pauli BLISS+HCT N Comm"
BS_N2_TAG = "BS N/2 Comm"
BS_N_TAG = "BS N Comm"

DMRG_JOBS = [
    ("Original Qubit", None, False),
    ("HCT N/2", HCT_N2_TAG, False),
    ("HCT N", HCT_N_TAG, False),
    ("HCT+BLISS N", BLISS_HCT_TAG, False),
    ("BS N/2", BS_N2_TAG, False),
    ("BS N", BS_N_TAG, False),
    ("BS N + Fiedler", BS_N_TAG, True),
]

MPO_JOBS = [
    ("Original Qubit", None, False),
    ("HCT N", HCT_N_TAG, False),
    ("BS N", BS_N_TAG, False),
    ("HCT N + Fiedler", HCT_N_TAG, True),
    ("BS N + Fiedler", BS_N_TAG, True),
]


def beh2_geometry(be_h_distance: float):
    return [
        ("H", (0.0, 0.0, -be_h_distance)),
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, be_h_distance)),
    ]


def hamiltonian_paths(ham_dir: Path, system: str) -> tuple[Path, Path]:
    return ham_dir / f"{system}.pkl", ham_dir / system


def generate_hamiltonian(system: str, be_h_distance: float, ham_dir: Path, text):
    ham_dir.mkdir(parents=True, exist_ok=True)
    pkl_path, mol_path = hamiltonian_paths(ham_dir, system)

    print(f"\nGenerating Hamiltonian for {system}", file=text)
    print(f"Geometry: linear H-Be-H, R(Be-H) = {be_h_distance:.3f} Angstrom", file=text)

    molecule = MolecularData(
        geometry=beh2_geometry(be_h_distance),
        basis=BASIS,
        multiplicity=MULTIPLICITY,
        charge=CHARGE,
    )
    molecule.filename = str(mol_path)
    molecule = run_pyscf(
        molecule,
        run_scf=True,
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=False,
    )

    H = get_fermion_operator(molecule.get_molecular_hamiltonian())
    HQ = jordan_wigner(H)
    n_qubits = count_qubits(HQ)
    Hs = get_sparse_operator(HQ, n_qubits)
    fci_e, fci_gs = get_ground_state(Hs)
    hf_occ = get_hf_occ(molecule.n_electrons, molecule.n_orbitals, as_str=True)
    cisd_e, cisd_gs = get_cisd_gs(hf_occ, HQ, n_qubits, "wfs", tf="jw")

    with pkl_path.open("wb") as file_obj:
        pickle.dump((H, fci_e, fci_gs, cisd_e, cisd_gs), file_obj)

    molecule.filename = str(mol_path)
    molecule.save()

    print(f"n_qubits: {n_qubits}", file=text)
    print(f"n_electrons: {molecule.n_electrons}", file=text)
    print(f"n_orbitals: {molecule.n_orbitals}", file=text)
    print(f"FCI energy: {fci_e}", file=text)
    print(f"CISD energy: {cisd_e}", file=text)
    print(f"Saved pickle: {pkl_path}", file=text)
    print(f"Saved MolecularData: {mol_path}", file=text)
    return H, fci_e, fci_gs, cisd_e, cisd_gs, molecule


def load_or_generate_hamiltonian(system: str, be_h_distance: float, ham_dir: Path, text, force: bool):
    pkl_path, mol_path = hamiltonian_paths(ham_dir, system)
    if force or not pkl_path.exists():
        return generate_hamiltonian(system, be_h_distance, ham_dir, text)

    with pkl_path.open("rb") as file_obj:
        H, fci_e, fci_gs, cisd_e, cisd_gs = pickle.load(file_obj)
    molecule = MolecularData(filename=str(mol_path))
    print(f"\nLoaded Hamiltonian for {system} from {pkl_path}", file=text)
    return H, fci_e, fci_gs, cisd_e, cisd_gs, molecule


def make_commutator_metric(HQ, cisd_gs, n_qubits: int):
    Hs = get_sparse_operator(HQ, n_qubits)

    def group_cost(symmetries):
        return comm_sq_exp_fast(symmetries, Hs, cisd_gs, n_qubits)

    def singleton_metric(symmetry):
        return group_cost([symmetry])

    def beam_score(symmetries):
        return -group_cost(symmetries)

    return singleton_metric, beam_score


def make_dataset(tag: str, symmetries, HQ, fci_gs, n_qubits: int):
    n2_sym = len(symmetries) == n_qubits // 2
    return BenchmarkData(
        tag=tag,
        symmetries=list(symmetries),
        non_commuting_l1=universal_grading(symmetries, HQ, verbose=False),
        num_commuting_terms=len(find_commuting_paulis(HQ, symmetries, verbose=False)),
        sym_entropy=(
            entropy_pauli_syms(symmetries, fci_gs, n_qubits, verbose=False)
            if n2_sym
            else 0
        ),
        clifford_synthesis_basis=SYNTHESIS_BASIS,
        clifford_generator_mapping=GENERATOR_MAPPING,
    )


def generate_hct_datasets(HQ, cisd_gs, fci_gs, n_qubits: int, text):
    singleton_metric, _beam_score = make_commutator_metric(HQ, cisd_gs, n_qubits)

    print("\nRunning HCT(n/2)...", file=text)
    hct_n2, eps_n2 = hct_mod(
        HQ,
        n_qubits // 2,
        sym_metric_func=singleton_metric,
        use_coeffs_eps=True,
    )
    print("Running HCT(n)...", file=text)
    hct_n, eps_n = hct_mod(
        HQ,
        n_qubits,
        sym_metric_func=singleton_metric,
        use_coeffs_eps=True,
    )
    return {
        HCT_N2_TAG: make_dataset(HCT_N2_TAG, hct_n2, HQ, fci_gs, n_qubits),
        HCT_N_TAG: make_dataset(HCT_N_TAG, hct_n, HQ, fci_gs, n_qubits),
    }, {
        HCT_N2_TAG: eps_n2,
        HCT_N_TAG: eps_n,
    }


def generate_bliss_hct_dataset(H, HQ, cisd_gs, fci_gs, molecule, n_qubits: int, text):
    singleton_metric, _beam_score = make_commutator_metric(HQ, cisd_gs, n_qubits)

    print("\nRunning Pauli BLISS before HCT(n)...", file=text)
    print(f"n_electrons used for BLISS: {molecule.n_electrons}", file=text)
    H_bliss, info = lp_bliss_paper_real_pauli_1norm(
        H,
        n_electrons=molecule.n_electrons,
        n_orb=n_qubits,
    )
    info_summary = bliss_info_summary(info)
    print("BLISS info:", info_summary, file=text)

    HQ_bliss = jordan_wigner(H_bliss)
    sym_hct_bliss, eps_bliss = hct_mod(
        HQ_bliss,
        n_qubits,
        sym_metric_func=singleton_metric,
        use_coeffs_eps=True,
    )
    return (
        make_dataset(BLISS_HCT_TAG, sym_hct_bliss, HQ, fci_gs, n_qubits),
        eps_bliss,
        info_summary,
    )


def generate_beam_dataset(
    tag: str,
    target_rank: int,
    HQ,
    cisd_gs,
    fci_gs,
    n_qubits: int,
    args,
    text,
):
    _singleton_metric, beam_score = make_commutator_metric(HQ, cisd_gs, n_qubits)
    print(f"\nRunning beam search target_rank={target_rank} ({tag})...", file=text)
    t0 = perf_counter()
    symmetries = BeamSearch_Symmetries(
        HQ,
        target_rank=target_rank,
        n_qubits=n_qubits,
        beam_width=args.beam_width,
        heavy_core_fraction=args.heavy_core_fraction,
        max_candidates_from_terms=args.max_candidates_from_terms,
        include_hct_symmetries=args.include_hct_symmetries,
        hct_n_sym=n_qubits,
        hct_use_coeffs_eps=True,
        include_pairwise_products=args.include_pairwise_products,
        pairwise_seed_terms=args.pairwise_seed_terms,
        do_local_refine=not args.no_local_refine,
        local_refine_passes=args.local_refine_passes,
        seed_with_exact_symmetries=args.seed_with_exact_symmetries,
        score_func=beam_score,
        score_is_separable=True,
        n_processes=args.n_processes,
        mp_start_method=args.mp_start_method,
    )
    seconds = perf_counter() - t0
    diagnostics = validate_symmetry_generators(HQ, symmetries, n_qubits=n_qubits)
    diagnostics["search_seconds"] = seconds
    diagnostics["requested_target_rank"] = target_rank
    print(f"Beam-search seconds: {seconds:.6f}", file=text)
    print(f"Beam diagnostics: {diagnostics}", file=text)
    return make_dataset(tag, symmetries, HQ, fci_gs, n_qubits), diagnostics


def write_symmetry_block(text, label: str, symmetries):
    print(label, file=text)
    for symmetry in symmetries:
        print(f"  {symmetry}", file=text)


def benchmark_system(system: str, distance: float, args, text):
    H, fci_e, fci_gs, cisd_e, cisd_gs, molecule = load_or_generate_hamiltonian(
        system,
        distance,
        args.ham_dir,
        text,
        force=args.force_hamiltonians,
    )
    HQ = jordan_wigner(H)
    n_qubits = count_qubits(HQ)

    print("\n" + "=" * 80, file=text)
    print(system, file=text)
    print("=" * 80, file=text)
    print(f"R(Be-H) = {distance:.3f} Angstrom", file=text)
    print(f"n_qubits = {n_qubits}", file=text)
    print(f"FCI energy = {fci_e}", file=text)
    print(f"CISD energy = {cisd_e}", file=text)

    original_fci_ent = get_entropies_at_cuts(fci_gs, n_qubits, log_base=LOG_BASE)
    original_cisd_ent = get_entropies_at_cuts(cisd_gs, n_qubits, log_base=LOG_BASE)
    write_entropies(text, "\nFCI entanglement before Clifford:", original_fci_ent)
    write_entropies(text, "\nCISD entanglement before Clifford:", original_cisd_ent)

    datasets_by_tag, hct_eps = generate_hct_datasets(HQ, cisd_gs, fci_gs, n_qubits, text)
    bliss_dataset, bliss_eps, bliss_info = generate_bliss_hct_dataset(
        H,
        HQ,
        cisd_gs,
        fci_gs,
        molecule,
        n_qubits,
        text,
    )
    datasets_by_tag[BLISS_HCT_TAG] = bliss_dataset

    beam_n2, beam_n2_diag = generate_beam_dataset(
        BS_N2_TAG,
        n_qubits // 2,
        HQ,
        cisd_gs,
        fci_gs,
        n_qubits,
        args,
        text,
    )
    beam_n, beam_n_diag = generate_beam_dataset(
        BS_N_TAG,
        n_qubits,
        HQ,
        cisd_gs,
        fci_gs,
        n_qubits,
        args,
        text,
    )
    datasets_by_tag[BS_N2_TAG] = beam_n2
    datasets_by_tag[BS_N_TAG] = beam_n

    BenchmarkData.save_datasets(
        list(datasets_by_tag.values()),
        args.output_dir / f"jul20_{system}_datasets",
    )

    system_result = {
        "geometry": {
            "description": "linear H-Be-H",
            "be_h_distance_angstrom": distance,
            "basis": BASIS,
            "multiplicity": MULTIPLICITY,
            "charge": CHARGE,
        },
        "n_qubits": n_qubits,
        "n_electrons": molecule.n_electrons,
        "n_orbitals": molecule.n_orbitals,
        "fci_energy": fci_e,
        "cisd_energy": cisd_e,
        "fci_entanglement_before_clifford": original_fci_ent,
        "cisd_entanglement_before_clifford": original_cisd_ent,
        "search": {
            "hct_epsilons": hct_eps,
            "bliss_hct_epsilons": bliss_eps,
            "bliss_info": bliss_info,
            "beam": {
                BS_N2_TAG: beam_n2_diag,
                BS_N_TAG: beam_n_diag,
            },
        },
        "benchmarks": {},
    }

    transform_cache = {}
    fiedler_cache = {}

    def get_transformed(dataset_tag):
        if dataset_tag in transform_cache:
            return transform_cache[dataset_tag]
        symdata = datasets_by_tag[dataset_tag]
        ent, H_perm, clifford, fci_rot = get_permuted_bipartite_entanglement(
            symdata.symmetries,
            HQ,
            n_qubits,
            fci_energy=fci_e,
            fci_gs=fci_gs,
            verbose=False,
            return_state=True,
            return_clifford=True,
            log_base=LOG_BASE,
            use_dmrg=False,
            synthesis_basis=SYNTHESIS_BASIS,
            generator_mapping=GENERATOR_MAPPING,
        )
        cisd_rot = clifford.transform_state(cisd_gs)
        cisd_ent = get_entropies_at_cuts(cisd_rot, n_qubits, log_base=LOG_BASE)
        transformed_symmetries = list(clifford.transformed_symmetries)
        transform_cache[dataset_tag] = {
            "symdata": symdata,
            "ent": ent,
            "H": H_perm,
            "fci_state": fci_rot,
            "state": cisd_rot,
            "cisd_ent": cisd_ent,
            "clifford": clifford,
            "transformed_symmetries": transformed_symmetries,
        }
        return transform_cache[dataset_tag]

    def get_fiedler(dataset_tag):
        if dataset_tag in fiedler_cache:
            return fiedler_cache[dataset_tag]
        transformed = get_transformed(dataset_tag)
        ent_reord, H_reord, state_reord, fiedler_info = do_fiedler_reordering(
            transformed["H"],
            transformed["state"],
            n_qubits=n_qubits,
            verbose=False,
            log_base=LOG_BASE,
        )
        fci_state_reord = reorder_statevector_axes(
            transformed["fci_state"],
            fiedler_info["ordering"],
            n_qubits,
        )
        fci_ent_reord = get_entropies_at_cuts(
            fci_state_reord,
            n_qubits,
            log_base=LOG_BASE,
        )
        perm = invert_ordering(fiedler_info["ordering"])
        reordered_symmetries = [
            permute_qubits_in_qubit_operator(sym, perm)
            for sym in transformed["transformed_symmetries"]
        ]
        fiedler_cache[dataset_tag] = {
            "ent": ent_reord,
            "fci_ent": fci_ent_reord,
            "H": H_reord,
            "state": state_reord,
            "fci_state": fci_state_reord,
            "info": fiedler_info,
            "reordered_symmetries": reordered_symmetries,
        }
        return fiedler_cache[dataset_tag]

    for dataset_tag, symdata in datasets_by_tag.items():
        transformed = get_transformed(dataset_tag)
        print("\n" + "-" * 80, file=text)
        print(dataset_tag, file=text)
        write_symmetry_block(text, "Saved symmetries:", symdata.symmetries)
        write_symmetry_block(
            text,
            "Symmetries after Clifford:",
            transformed["transformed_symmetries"],
        )
        write_entropies(text, "FCI entanglement after Clifford:", transformed["ent"])
        write_entropies(
            text,
            "CISD entanglement after Clifford:",
            transformed["cisd_ent"],
        )
        system_result["benchmarks"][dataset_tag] = {
            "saved_symmetries": sym_strings(symdata.symmetries),
            "symmetries_after_clifford": sym_strings(
                transformed["transformed_symmetries"]
            ),
            "non_commuting_l1": symdata.non_commuting_l1,
            "num_commuting_terms": symdata.num_commuting_terms,
            "sym_entropy": symdata.sym_entropy,
            "fci_entanglement_after_clifford": transformed["ent"],
            "cisd_entanglement_after_clifford": transformed["cisd_ent"],
        }
        symdata.cut_entropies = list(transformed["ent"])

    for dataset_tag in [HCT_N_TAG, BS_N_TAG]:
        fiedler = get_fiedler(dataset_tag)
        print("\n" + "-" * 80, file=text)
        print(f"{dataset_tag} + Fiedler", file=text)
        print(f"Fiedler ordering: {fiedler['info']['ordering']}", file=text)
        write_symmetry_block(
            text,
            "Symmetries after Clifford + Fiedler:",
            fiedler["reordered_symmetries"],
        )
        write_entropies(
            text,
            "FCI entanglement after Clifford + Fiedler:",
            fiedler["fci_ent"],
        )
        write_entropies(
            text,
            "CISD entanglement after Clifford + Fiedler:",
            fiedler["ent"],
        )
        system_result["benchmarks"][dataset_tag]["fiedler_ordering"] = fiedler[
            "info"
        ]["ordering"]
        system_result["benchmarks"][dataset_tag]["symmetries_after_fiedler"] = (
            sym_strings(fiedler["reordered_symmetries"])
        )
        system_result["benchmarks"][dataset_tag]["fci_entanglement_after_fiedler"] = (
            fiedler["fci_ent"]
        )
        system_result["benchmarks"][dataset_tag]["cisd_entanglement_after_fiedler"] = (
            fiedler["ent"]
        )
        system_result["benchmarks"][dataset_tag]["fiedler_info"] = fiedler["info"]

    BenchmarkData.save_datasets(
        list(datasets_by_tag.values()),
        args.output_dir / f"jul20_{system}_datasets",
    )

    dmrg_rows = []
    mpo_rows = []

    for label, dataset_tag, use_fiedler in DMRG_JOBS:
        if dataset_tag is None:
            working_H = HQ
            working_state = fci_gs
        else:
            working = get_fiedler(dataset_tag) if use_fiedler else get_transformed(dataset_tag)
            working_H = working["H"]
            working_state = working["fci_state"]

        if args.skip_dmrg:
            dmrg_bd = ""
            dmrg_energy = ""
        else:
            guess_mps = qtn.MatrixProductState.from_dense(
                working_state,
                cutoff=1e-20,
            )
            dmrg_bd, dmrg_energy, _ = find_dmrg_conv_bd_quimb(
                working_H,
                n_qubits,
                fci_e,
                tol=1.6e-3,
                n_sweeps=args.dmrg_sweeps,
                reps=1,
                verbose=False,
                compress_cutoff=1e-20,
                sweep_tol=1e-6,
                noise=1e0,
                bsz=2,
                guess_mps=guess_mps,
                seed=0,
                return_data=True,
            )
            if dataset_tag is not None and not use_fiedler:
                datasets_by_tag[dataset_tag].dmrg_bd = int(dmrg_bd)

        row = {
            "system": system,
            "benchmark": label,
            "dataset_tag": dataset_tag if dataset_tag is not None else "",
            "fiedler": use_fiedler,
            "n_qubits": n_qubits,
            "dmrg_bd": dmrg_bd,
            "dmrg_energy": dmrg_energy,
        }
        dmrg_rows.append(row)
        print(f"DMRG {label}: {row}", file=text)

    for label, dataset_tag, use_fiedler in MPO_JOBS:
        if dataset_tag is None:
            working_H = HQ
        else:
            working = get_fiedler(dataset_tag) if use_fiedler else get_transformed(dataset_tag)
            working_H = working["H"]

        if args.skip_mpo:
            mpo_bd = ""
        else:
            mpo = MPO_from_QubitOperator(
                working_H,
                None,
                mpo_cutoff=1e-20,
                compression_freq=20,
                verbose=False,
            )
            mpo_bd = max(mpo.bond_sizes())

        row = {
            "system": system,
            "benchmark": label,
            "dataset_tag": dataset_tag if dataset_tag is not None else "",
            "fiedler": use_fiedler,
            "n_qubits": n_qubits,
            "mpo_bd": mpo_bd,
        }
        mpo_rows.append(row)
        print(f"MPO {label}: {row}", file=text)

    BenchmarkData.save_datasets(
        list(datasets_by_tag.values()),
        args.output_dir / f"jul20_{system}_datasets",
    )
    return system_result, dmrg_rows, mpo_rows


def make_runtime_rows(dmrg_rows, mpo_rows):
    dmrg_bd = {
        (row["system"], row["benchmark"]): row["dmrg_bd"]
        for row in dmrg_rows
        if row["dmrg_bd"] != ""
    }
    mpo_bd = {
        (row["system"], row["benchmark"]): row["mpo_bd"]
        for row in mpo_rows
        if row["mpo_bd"] != ""
    }
    rows = []
    systems = []
    for row in dmrg_rows:
        if row["system"] not in systems:
            systems.append(row["system"])
    for system in systems:
        n_qubits = next(int(row["n_qubits"]) for row in dmrg_rows if row["system"] == system)
        out = {"system": system}
        for label in ["Original Qubit", "BS N", "BS N + Fiedler"]:
            if (system, label) not in dmrg_bd or (system, label) not in mpo_bd:
                continue
            out[f"T {label}"], _ = runtime_factors(
                n_qubits,
                2,
                dmrg_bd[(system, label)],
                mpo_bd[(system, label)],
            )
            out[f"M {label}"], _ = memory_factors(
                n_qubits,
                2,
                dmrg_bd[(system, label)],
                mpo_bd[(system, label)],
            )
        rows.append(out)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=list(SYSTEM_GEOMETRIES),
        default=list(SYSTEM_GEOMETRIES),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "saved" / "results" / "Jul20",
    )
    parser.add_argument(
        "--ham-dir",
        type=Path,
        default=ROOT / "saved" / "results" / "Jul20" / "hamiltonians",
    )
    parser.add_argument("--force-hamiltonians", action="store_true")
    parser.add_argument("--skip-dmrg", action="store_true")
    parser.add_argument("--skip-mpo", action="store_true")
    parser.add_argument("--dmrg-sweeps", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--heavy-core-fraction", type=float, default=0.95)
    parser.add_argument("--max-candidates-from-terms", type=int, default=256)
    parser.add_argument("--include-hct-symmetries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-pairwise-products", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pairwise-seed-terms", type=int, default=24)
    parser.add_argument("--seed-with-exact-symmetries", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-local-refine", action="store_true")
    parser.add_argument("--local-refine-passes", type=int, default=10)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mp-start-method", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ham_dir.mkdir(parents=True, exist_ok=True)

    text_file = args.output_dir / "jul20_beh2_clifford_benchmarks.txt"
    json_file = args.output_dir / "jul20_beh2_clifford_benchmarks.json"
    dmrg_csv = args.output_dir / "jul20_beh2_dmrg_bond_dimensions.csv"
    mpo_csv = args.output_dir / "jul20_beh2_mpo_bond_dimensions.csv"
    runtime_csv = args.output_dir / "jul20_beh2_runtime_memory.csv"

    all_results = {
        "synthesis_basis": SYNTHESIS_BASIS,
        "generator_mapping": GENERATOR_MAPPING,
        "log_base": "e",
        "basis": BASIS,
        "systems": {},
        "geometry_notes": {
            "BeH2_eqm": "R(Be-H)=1.326 Angstrom, literature equilibrium value.",
            "BeH2_corr": "R(Be-H)=2.000 Angstrom, intermediate stretch chosen between equilibrium and strongly broken-symmetry region.",
            "BeH2_diss": "R(Be-H)=4.000 Angstrom, endpoint of the cited symmetric dissociation scan.",
        },
        "settings": {
            "beam_width": args.beam_width,
            "heavy_core_fraction": args.heavy_core_fraction,
            "max_candidates_from_terms": args.max_candidates_from_terms,
            "include_hct_symmetries": args.include_hct_symmetries,
            "include_pairwise_products": args.include_pairwise_products,
            "pairwise_seed_terms": args.pairwise_seed_terms,
            "seed_with_exact_symmetries": args.seed_with_exact_symmetries,
            "local_refine": not args.no_local_refine,
            "local_refine_passes": args.local_refine_passes,
            "dmrg_sweeps": args.dmrg_sweeps,
        },
    }
    all_dmrg_rows = []
    all_mpo_rows = []

    with text_file.open("w") as text:
        print("JUL20 BeH2/STO-3G Clifford/Fiedler benchmarks", file=text)
        print(f"synthesis_basis = {SYNTHESIS_BASIS}", file=text)
        print(f"generator_mapping = {GENERATOR_MAPPING}", file=text)
        print(f"output_dir = {args.output_dir}", file=text)
        print(f"ham_dir = {args.ham_dir}", file=text)
        print(f"systems = {args.systems}", file=text)
        print(f"settings = {as_jsonable(all_results['settings'])}", file=text)

        for system in args.systems:
            print(f"Starting {system}", flush=True)
            result, dmrg_rows, mpo_rows = benchmark_system(
                system,
                SYSTEM_GEOMETRIES[system],
                args,
                text,
            )
            all_results["systems"][system] = result
            all_dmrg_rows.extend(dmrg_rows)
            all_mpo_rows.extend(mpo_rows)
            write_csv(dmrg_csv, all_dmrg_rows)
            write_csv(mpo_csv, all_mpo_rows)
            write_csv(runtime_csv, make_runtime_rows(all_dmrg_rows, all_mpo_rows))
            save_json(json_file, all_results)

    write_csv(dmrg_csv, all_dmrg_rows)
    write_csv(mpo_csv, all_mpo_rows)
    write_csv(runtime_csv, make_runtime_rows(all_dmrg_rows, all_mpo_rows))
    save_json(json_file, all_results)
    print(f"Wrote text output to {text_file}")
    print(f"Wrote JSON output to {json_file}")
    print(f"Wrote DMRG CSV to {dmrg_csv}")
    print(f"Wrote MPO CSV to {mpo_csv}")
    print(f"Wrote runtime/memory CSV to {runtime_csv}")


if __name__ == "__main__":
    main()
