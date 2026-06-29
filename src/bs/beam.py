from __future__ import annotations
from .utils import *

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openfermion import QubitOperator
from src.bs.utils import *

### AI code, seems to work

# ============================================================
# Candidate pool construction
# ============================================================

def build_candidate_pool(
    terms: Sequence[WeightedTerm],
    n_qubits: int,
    *,
    max_candidates_from_terms: Optional[int] = 256,
    include_pairwise_products: bool = False,
    pairwise_seed_terms: int = 24,
    max_pauli_weight: Optional[int] = None,
) -> List[PauliMask]:
    """
    Restricted candidate pool used during heuristic search.

    Current choices:
      1. Pauli strings already appearing in H
      2. Optionally products of the heaviest few Hamiltonian terms
      3. All single-qubit X/Y/Z operators

    Optional filtering:
      - truncate Hamiltonian terms to the top `max_candidates_from_terms`
      - discard candidates above `max_pauli_weight`
    """
    ordered = sorted(terms, key=lambda t: t.abs_coeff, reverse=True)
    base = ordered if max_candidates_from_terms is None else ordered[:max_candidates_from_terms]

    pool: Dict[PauliMask, None] = {}

    for t in base:
        if max_pauli_weight is None or pauli_weight(t.mask) <= max_pauli_weight:
            pool[t.mask] = None

    if include_pairwise_products:
        seed = ordered[:pairwise_seed_terms]
        for i in range(len(seed)):
            for j in range(i + 1, len(seed)):
                prod = pauli_product_mod_phase(seed[i].mask, seed[j].mask)
                if prod != (0, 0):
                    if max_pauli_weight is None or pauli_weight(prod) <= max_pauli_weight:
                        pool[prod] = None

    for q in range(n_qubits):
        for p in ("X", "Y", "Z"):
            pool[term_to_masks(((q, p),), n_qubits)] = None

    return list(pool.keys())


def build_candidate_pool_hct(
    terms: Sequence[WeightedTerm],
    n_qubits: int,
    *,
    max_candidates_from_terms: Optional[int] = 256,
    include_pairwise_products: bool = False,
    pairwise_seed_terms: int = 24,
    max_pauli_weight: Optional[int] = None,
    include_hct_symmetries: bool = True,
    hct_n_sym: Optional[int] = None,
    hct_use_coeffs_eps: bool = True,
) -> List[PauliMask]:
    """
    Same as build_candidate_pool, plus HCT-derived approximate symmetries
    as a 4th source. HCT candidates are inserted first so they take priority
    in beam-search iteration order.

    Extra options:
      - include_hct_symmetries: toggle the HCT source
      - hct_n_sym:              how many symmetries to request (default n_qubits)
      - hct_use_coeffs_eps:     forwarded to hct_mod
    """
    pool: Dict[PauliMask, None] = {}

    if include_hct_symmetries:
        print("Adding HCT symmetries to the pool:")
        from src.sym import hct_mod
        from src.bs.utils import terms_to_HQ, qubitops_to_masks

        HQ_rt = terms_to_HQ(terms)
        n_sym = hct_n_sym if hct_n_sym is not None else n_qubits
        try:
            hct_syms, _ = hct_mod(
                HQ_rt,
                n_sym=n_sym,
                use_coeffs_eps=hct_use_coeffs_eps,
                verbose=False,
            )
        except Exception:
            print("Warning: HCT symmetries not included!")
            hct_syms = []

        for op in hct_syms:
            m = qubitops_to_masks([op], n_qubits)[0]
            if max_pauli_weight is None or pauli_weight(m) <= max_pauli_weight:
                pool[m] = None

    ordered = sorted(terms, key=lambda t: t.abs_coeff, reverse=True)
    base = ordered if max_candidates_from_terms is None else ordered[:max_candidates_from_terms]

    for t in base:
        if max_pauli_weight is None or pauli_weight(t.mask) <= max_pauli_weight:
            pool[t.mask] = None

    if include_pairwise_products:
        seed = ordered[:pairwise_seed_terms]
        for i in range(len(seed)):
            for j in range(i + 1, len(seed)):
                prod = pauli_product_mod_phase(seed[i].mask, seed[j].mask)
                if prod != (0, 0):
                    if max_pauli_weight is None or pauli_weight(prod) <= max_pauli_weight:
                        pool[prod] = None

    for q in range(n_qubits):
        for p in ("X", "Y", "Z"):
            pool[term_to_masks(((q, p),), n_qubits)] = None

    return list(pool.keys())



# ============================================================
# Objective evaluation
# ============================================================

def retained_weight(basis_masks: Sequence[PauliMask], terms: Sequence[WeightedTerm]) -> float:
    """
    Checks magnitude of terms (||H_0||_1) that commute with all of basis_masks

    basis_masks - symmetries
    terms - Hamiltonian/operator
    """
    total = 0.0
    for t in terms:
        if all(symplectic_commutes(t.mask, g) for g in basis_masks):
            total += t.abs_coeff
    return total


# ============================================================
# Beam search state
# ============================================================

@dataclass
class SearchState:
    basis: List[PauliMask]
    rref_rows: List[int]
    heavy_score: float


def state_key(state: SearchState) -> Tuple[int, ...]:
    return tuple(state.rref_rows)


def commuting_extension_candidates(
    state: SearchState,
    candidate_pool: Sequence[PauliMask],
    n_qubits: int,
) -> Iterable[PauliMask]:
    for g in candidate_pool:
        if all(symplectic_commutes(g, h) for h in state.basis):
            gv = combine_mask(g, n_qubits)
            if not in_span(gv, state.rref_rows):
                yield g



# ============================================================
# Heavy-core beam search
# ============================================================

def beam_search_symmetries(
    hamiltonian: QubitOperator,
    candidate_pool: List[PauliMask],
    *,
    target_rank: int = None,
    n_qubits: Optional[int] = None,
    beam_width: int = 16,
    heavy_core_fraction: float = 0.95,
    initial_generators: Optional[Sequence[QubitOperator]] = None,
    score_func = None
) -> List[QubitOperator]:
    """
    Heavy-core beam search for a commuting independent generator set of rank n_qubits // 2.

    Optionally starts from an initial commuting independent seed set.
    """
    n_qubits, terms = qubit_operator_terms(hamiltonian, n_qubits)
    heavy_terms = heavy_core(terms, heavy_core_fraction) #for non

    #set defaults
    if n_qubits % 2 != 0:
        raise ValueError("This implementation targets rank n_qubits // 2, so n_qubits must be even.")
    n_bits = 2 * n_qubits
    if target_rank is None:
        target_rank = n_qubits // 2 # TODO take target_rank as input

    # score() takes basis masks as input and returns score
    if score_func is None:
        score = lambda basis: retained_weight(basis, heavy_terms)
    else:
        score = lambda basis: score_func([mask_to_qubit_operator(m, n_qubits) for m in basis])
    
    seed_basis: List[PauliMask] = []
    seed_rows: List[int] = []

    if initial_generators is not None:
        seed_basis = qubitops_to_masks(initial_generators, n_qubits)

        for i in range(len(seed_basis)):
            for j in range(i + 1, len(seed_basis)):
                if not symplectic_commutes(seed_basis[i], seed_basis[j]):
                    raise ValueError("initial_generators must commute pairwise.")

        for g in seed_basis:
            gv = combine_mask(g, n_qubits)
            new_rows = try_add_to_span(gv, seed_rows, n_bits)
            if new_rows is None:
                raise ValueError("initial_generators must be linearly independent.")
            seed_rows = new_rows

        if len(seed_basis) > target_rank:
            raise ValueError("Too many initial generators for target rank n_qubits // 2.")

    init = SearchState(
        basis=seed_basis[:],
        rref_rows=seed_rows[:],
        heavy_score=score(seed_basis[:]),
    )

    beam = [init]

    for _depth in range(len(seed_basis), target_rank):
        children: Dict[Tuple[int, ...], SearchState] = {}

        for state in beam:
            for g in commuting_extension_candidates(state, candidate_pool, n_qubits):
                gv = combine_mask(g, n_qubits)
                new_rref = try_add_to_span(gv, state.rref_rows, n_bits)
                if new_rref is None: #if already in span, try_add.. returns None
                    continue

                new_basis = state.basis + [g]
                new_score = score(new_basis)
                child = SearchState(
                    basis=new_basis,
                    rref_rows=new_rref,
                    heavy_score=new_score,
                )
                key = state_key(child)

                prev = children.get(key)
                if prev is None or child.heavy_score > prev.heavy_score: #if not in children or better than some other child added in current iteration from pool with same rref
                    children[key] = child

        if not children:
            break

        beam = sorted(children.values(), key=lambda s: s.heavy_score, reverse=True)[:beam_width]

    if not beam:
        raise RuntimeError("Beam search failed to produce any commuting generators.")

    #final score function with all terms by default
    if score_func is None:
        score = lambda basis: retained_weight(basis, terms)
    else:
        score = lambda basis: score_func([mask_to_qubit_operator(m, n_qubits) for m in basis])

    best = max(beam, key=lambda s: score(s.basis))
    completed = complete_basis_any(best.basis, n_qubits, target_rank)
    return [mask_to_qubit_operator(g, n_qubits) for g in completed]


# ============================================================
# Local 1-swap refinement on the full Hamiltonian
# ============================================================

def local_swap_refine(
    hamiltonian: QubitOperator,
    symmetries: Sequence[QubitOperator],
    candidate_pool: List[PauliMask],
    *,
    n_qubits: Optional[int] = None,
    max_passes: int = 10,
    score_func = None
) -> List[QubitOperator]:
    """
    Repeatedly try single-generator replacements that improve retained weight on the full Hamiltonian.
    """
    n_qubits_h, terms = qubit_operator_terms(hamiltonian, n_qubits)
    n_qubits = n_qubits_h if n_qubits is None else n_qubits
    n_bits = 2 * n_qubits
    target_rank = len(symmetries)

    current = qubitops_to_masks(symmetries, n_qubits)

    def score(basis: Sequence[PauliMask]) -> float:
        if score_func is not None:
            #convert to qubitops
            basis_qops = [mask_to_qubit_operator(m, n_qubits) for m in basis]
            return score_func(basis_qops)
        else:
            return retained_weight(basis, terms)

    current_score = score(current)

    for _ in range(max_passes):
        improved = False

        for idx in range(target_rank):
            reduced = current[:idx] + current[idx + 1 :]
            reduced_rows, _ = rref([combine_mask(g, n_qubits) for g in reduced], n_bits)

            best_replacement = None
            best_score = current_score

            for g in candidate_pool:
                if all(symplectic_commutes(g, h) for h in reduced):
                    gv = combine_mask(g, n_qubits)
                    if not in_span(gv, reduced_rows):
                        trial = reduced + [g]
                        s = score(trial)
                        if s > best_score + 1e-15:
                            best_score = s
                            best_replacement = g

            constraints = []
            for g in reduced:
                x, z = g
                constraints.append(z | (x << n_qubits))

            for vec in nullspace_basis(constraints, n_bits):
                if vec == 0 or in_span(vec, reduced_rows):
                    continue
                g = split_mask(vec, n_qubits)
                if all(symplectic_commutes(g, h) for h in reduced):
                    trial = reduced + [g]
                    s = score(trial)
                    if s > best_score + 1e-15:
                        best_score = s
                        best_replacement = g

            if best_replacement is not None:
                current = reduced + [best_replacement]
                current_score = best_score
                improved = True
                break

        if not improved:
            break

    return [mask_to_qubit_operator(g, n_qubits) for g in current]


# ============================================================
# Top-level workflow
# ============================================================
import warnings

def find_commuting_symmetry_generators(*args, **kwargs):
    warnings.warn(
        "find_commuting_symmetry_generators is deprecated; use BeamSearch_Symmetries instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return BeamSearch_Symmetries(*args, **kwargs)
        
def BeamSearch_Symmetries(
    hamiltonian: QubitOperator,
    *,
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
    score_func = None
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
    seed_generators: Optional[List[QubitOperator]] = None

    if seed_with_exact_symmetries:
        exact_syms = exact_pauli_symmetry_basis(hamiltonian, n_qubits=n_qubits)
        if max_exact_symmetry_seeds is not None:
            exact_syms = exact_syms[:max_exact_symmetry_seeds]
        seed_generators = exact_syms

    #build candidate pool
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

    syms = beam_search_symmetries(
        hamiltonian,
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
            hamiltonian,
            syms,
            candidate_pool,
            n_qubits=n_qubits,
            max_passes=local_refine_passes,
            score_func=score_func
        )

    return syms


# ============================================================
# Validation / diagnostics
# ============================================================

def validate_symmetry_generators(
    hamiltonian: QubitOperator,
    generators: Sequence[QubitOperator],
    *,
    n_qubits: Optional[int] = None,
) -> Dict[str, object]:
    n_qubits_h, terms = qubit_operator_terms(hamiltonian, n_qubits)
    n_qubits = n_qubits_h if n_qubits is None else n_qubits
    n_bits = 2 * n_qubits

    masks = qubitops_to_masks(generators, n_qubits)

    pairwise_commuting = all(
        symplectic_commutes(masks[i], masks[j])
        for i in range(len(masks))
        for j in range(i + 1, len(masks))
    )

    rref_rows, _ = rref([combine_mask(g, n_qubits) for g in masks], n_bits)
    independent_rank = len(rref_rows)

    retained = retained_weight(masks, terms)
    total = sum(t.abs_coeff for t in terms)

    exact_symmetry_flags = []
    for g in masks:
        exact_symmetry_flags.append(all(symplectic_commutes(t.mask, g) for t in terms))

    return {
        "n_qubits": n_qubits,
        "target_rank": n_qubits // 2,
        "num_generators": len(masks),
        "independent_rank": independent_rank,
        "pairwise_commuting": pairwise_commuting,
        "all_exact_symmetries": all(exact_symmetry_flags),
        "retained_weight": retained,
        "total_weight": total,
        "retained_fraction": retained / total if total > 0 else 1.0,
    }
