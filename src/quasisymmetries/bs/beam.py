from __future__ import annotations
import multiprocessing as mp
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from openfermion import QubitOperator
from .utils import *

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
        from ..sym import hct_mod
        from .utils import terms_to_HQ, qubitops_to_masks

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


class SeparableScoreEvaluator:
    """Cache singleton scores and sum them for a separable basis objective."""

    def __init__(
        self,
        score_func: Callable[[List[QubitOperator]], Any],
        n_qubits: int,
        cache: Optional[MutableMapping[PauliMask, Any]] = None,
    ) -> None:
        self.score_func = score_func
        self.n_qubits = n_qubits
        self.cache = cache if cache is not None else {}

    def singleton(self, mask: PauliMask):
        if mask not in self.cache:
            op = mask_to_qubit_operator(mask, self.n_qubits)
            self.cache[mask] = self.score_func([op])
        return self.cache[mask]

    def prime(self, masks: Sequence[PauliMask]) -> None:
        for mask in masks:
            self.singleton(mask)

    def basis_score(self, basis: Sequence[PauliMask]):
        return sum((self.singleton(mask) for mask in basis), 0)


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


_PARALLEL_CANDIDATE_POOL: Tuple[PauliMask, ...] = ()
_PARALLEL_N_QUBITS = 0
_PARALLEL_N_BITS = 0
_PARALLEL_SINGLETON_SCORES: Optional[Dict[PauliMask, Any]] = None


def _initialize_extension_workers(
    candidate_pool: Sequence[PauliMask],
    n_qubits: int,
    n_bits: int,
    singleton_scores: Optional[MutableMapping[PauliMask, Any]],
) -> None:
    global _PARALLEL_CANDIDATE_POOL
    global _PARALLEL_N_QUBITS
    global _PARALLEL_N_BITS
    global _PARALLEL_SINGLETON_SCORES
    _PARALLEL_CANDIDATE_POOL = tuple(candidate_pool)
    _PARALLEL_N_QUBITS = n_qubits
    _PARALLEL_N_BITS = n_bits
    _PARALLEL_SINGLETON_SCORES = (
        dict(singleton_scores) if singleton_scores is not None else None
    )


def _parallel_extension_chunk(task):
    """Worker-side commutation, span, and RREF checks for one pool slice."""
    state_index, basis, rref_rows, state_score, start, stop = task
    extensions = []
    deduplicated = {}
    for candidate_index in range(start, stop):
        g = _PARALLEL_CANDIDATE_POOL[candidate_index]
        if not all(symplectic_commutes(g, h) for h in basis):
            continue
        gv = combine_mask(g, _PARALLEL_N_QUBITS)
        new_rref = try_add_to_span(gv, rref_rows, _PARALLEL_N_BITS)
        if new_rref is not None:
            if _PARALLEL_SINGLETON_SCORES is None:
                extensions.append(
                    (state_index, candidate_index, g, new_rref, None)
                )
            else:
                child_score = state_score + _PARALLEL_SINGLETON_SCORES[g]
                key = tuple(new_rref)
                previous = deduplicated.get(key)
                if previous is None or child_score > previous[4]:
                    deduplicated[key] = (
                        state_index,
                        candidate_index if previous is None else previous[1],
                        g,
                        new_rref,
                        child_score,
                    )
    if _PARALLEL_SINGLETON_SCORES is not None:
        extensions = sorted(deduplicated.values(), key=lambda item: item[1])
    return extensions


def _extension_tasks(
    states: Sequence["SearchState"],
    n_candidates: int,
    n_processes: int,
):
    if n_candidates == 0:
        return []
    chunks_per_state = max(1, 2 * n_processes // max(1, len(states)))
    chunk_size = max(1, (n_candidates + chunks_per_state - 1) // chunks_per_state)
    return [
        (
            state_index,
            state.basis,
            state.rref_rows,
            state.heavy_score,
            start,
            min(start + chunk_size, n_candidates),
        )
        for state_index, state in enumerate(states)
        for start in range(0, n_candidates, chunk_size)
    ]


def _parallel_extensions(
    states: Sequence["SearchState"],
    candidate_pool: Sequence[PauliMask],
    n_processes: int,
    process_pool,
):
    tasks = _extension_tasks(states, len(candidate_pool), n_processes)
    for chunk in process_pool.imap(_parallel_extension_chunk, tasks):
        yield from chunk


def _serial_extensions(
    states: Sequence["SearchState"],
    candidate_pool: Sequence[PauliMask],
    n_qubits: int,
    n_bits: int,
):
    for state_index, state in enumerate(states):
        for candidate_index, g in enumerate(candidate_pool):
            if not all(symplectic_commutes(g, h) for h in state.basis):
                continue
            gv = combine_mask(g, n_qubits)
            new_rref = try_add_to_span(gv, state.rref_rows, n_bits)
            if new_rref is not None:
                yield state_index, candidate_index, g, new_rref, None


def _make_process_pool(
    n_processes: int,
    candidate_pool: Sequence[PauliMask],
    n_qubits: int,
    n_bits: int,
    mp_start_method: Optional[str],
    singleton_scores: Optional[MutableMapping[PauliMask, Any]] = None,
):
    if n_processes <= 1:
        return nullcontext(None)
    context = mp.get_context(mp_start_method) if mp_start_method else mp.get_context()
    return context.Pool(
        processes=n_processes,
        initializer=_initialize_extension_workers,
        initargs=(candidate_pool, n_qubits, n_bits, singleton_scores),
    )



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
    score_func = None,
    score_is_separable: bool = False,
    separable_score_cache: Optional[MutableMapping[PauliMask, Any]] = None,
    n_processes: int = 1,
    mp_start_method: Optional[str] = None,
    process_pool=None,
) -> List[QubitOperator]:
    """
    Heavy-core beam search for a commuting independent generator set of rank n_qubits // 2.

    Optionally starts from an initial commuting independent seed set.

    When ``score_is_separable`` is true, ``score_func(basis)`` must equal
    ``sum(score_func([generator]) for generator in basis)``. Singleton scores
    are then cached, and child scores are updated in O(1).

    ``n_processes > 1`` parallelizes commutation, independence, and RREF work.
    Scoring remains in the parent process, so score closures need not be
    pickleable. Small candidate pools can be faster with ``n_processes=1``.
    """
    n_qubits, terms = qubit_operator_terms(hamiltonian, n_qubits)
    heavy_terms = heavy_core(terms, heavy_core_fraction) #for non

    #set defaults
    if n_qubits % 2 != 0:
        raise ValueError("This implementation targets rank n_qubits // 2, so n_qubits must be even.")
    n_bits = 2 * n_qubits
    if target_rank is None:
        target_rank = n_qubits // 2 # TODO take target_rank as input

    if n_processes < 1:
        raise ValueError("n_processes must be at least 1.")
    if score_is_separable and score_func is None:
        raise ValueError("score_is_separable=True requires a score_func.")

    separable_evaluator = None
    if score_is_separable:
        separable_evaluator = SeparableScoreEvaluator(
            score_func,
            n_qubits,
            cache=separable_score_cache,
        )
        # Evaluate every pool member exactly once before the search.
        separable_evaluator.prime(candidate_pool)
        score = separable_evaluator.basis_score
    elif score_func is None:
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

    pool_context = (
        nullcontext(process_pool)
        if process_pool is not None
        else _make_process_pool(
            n_processes,
            candidate_pool,
            n_qubits,
            n_bits,
            mp_start_method,
            separable_evaluator.cache if separable_evaluator is not None else None,
        )
    )
    with pool_context as active_process_pool:
        for _depth in range(len(seed_basis), target_rank):
            children: Dict[Tuple[int, ...], SearchState] = {}

            # A single beam state has no useful state-level parallelism and
            # returning its full first-generation pool costs more than RREF.
            if active_process_pool is None or len(beam) == 1:
                extensions = _serial_extensions(
                    beam, candidate_pool, n_qubits, n_bits
                )
            else:
                extensions = _parallel_extensions(
                    beam,
                    candidate_pool,
                    n_processes,
                    active_process_pool,
                )

            for state_index, _candidate_index, g, new_rref, cached_child_score in extensions:
                state = beam[state_index]
                new_basis = state.basis + [g]
                if cached_child_score is not None:
                    new_score = cached_child_score
                elif separable_evaluator is None:
                    new_score = score(new_basis)
                else:
                    new_score = state.heavy_score + separable_evaluator.singleton(g)
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
        final_score = lambda basis: retained_weight(basis, terms)
    else:
        final_score = score

    best = max(beam, key=lambda s: final_score(s.basis))
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
    score_func = None,
    score_is_separable: bool = False,
    separable_score_cache: Optional[MutableMapping[PauliMask, Any]] = None,
    n_processes: int = 1,
    mp_start_method: Optional[str] = None,
    process_pool=None,
) -> List[QubitOperator]:
    """
    Repeatedly try single-generator replacements that improve the full score.

    The separable cache and process pool can be shared with
    ``beam_search_symmetries`` to avoid repeated score evaluations and worker
    startup.
    """
    n_qubits_h, terms = qubit_operator_terms(hamiltonian, n_qubits)
    n_qubits = n_qubits_h if n_qubits is None else n_qubits
    n_bits = 2 * n_qubits
    target_rank = len(symmetries)

    current = qubitops_to_masks(symmetries, n_qubits)

    if n_processes < 1:
        raise ValueError("n_processes must be at least 1.")
    if score_is_separable and score_func is None:
        raise ValueError("score_is_separable=True requires a score_func.")

    separable_evaluator = None
    if score_is_separable:
        separable_evaluator = SeparableScoreEvaluator(
            score_func,
            n_qubits,
            cache=separable_score_cache,
        )
        separable_evaluator.prime(candidate_pool)

    def score(basis: Sequence[PauliMask]):
        if separable_evaluator is not None:
            return separable_evaluator.basis_score(basis)
        if score_func is not None:
            #convert to qubitops
            basis_qops = [mask_to_qubit_operator(m, n_qubits) for m in basis]
            return score_func(basis_qops)
        return retained_weight(basis, terms)

    current_score = score(current)

    pool_context = (
        nullcontext(process_pool)
        if process_pool is not None
        else _make_process_pool(
            n_processes,
            candidate_pool,
            n_qubits,
            n_bits,
            mp_start_method,
            separable_evaluator.cache if separable_evaluator is not None else None,
        )
    )
    with pool_context as active_process_pool:
        for _ in range(max_passes):
            improved = False

            for idx in range(target_rank):
                reduced = current[:idx] + current[idx + 1 :]
                reduced_rows, _ = rref([combine_mask(g, n_qubits) for g in reduced], n_bits)

                best_replacement = None
                best_score = current_score
                if separable_evaluator is not None:
                    reduced_score = (
                        current_score - separable_evaluator.singleton(current[idx])
                    )
                else:
                    reduced_score = None

                reduced_state = SearchState(
                    basis=reduced,
                    rref_rows=reduced_rows,
                    heavy_score=reduced_score if reduced_score is not None else 0,
                )
                if active_process_pool is None:
                    extensions = _serial_extensions(
                        [reduced_state], candidate_pool, n_qubits, n_bits
                    )
                else:
                    extensions = _parallel_extensions(
                        [reduced_state],
                        candidate_pool,
                        n_processes,
                        active_process_pool,
                    )

                for (
                    _state_index,
                    _candidate_index,
                    g,
                    _new_rref,
                    cached_child_score,
                ) in extensions:
                    if cached_child_score is not None:
                        s = cached_child_score
                    elif separable_evaluator is not None:
                        s = reduced_score + separable_evaluator.singleton(g)
                    else:
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
                        if separable_evaluator is not None:
                            s = reduced_score + separable_evaluator.singleton(g)
                        else:
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
    score_func = None,
    score_is_separable: bool = False,
    n_processes: int = 1,
    mp_start_method: Optional[str] = None,
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
      - cache singleton scores when ``score_is_separable=True``
      - parallelize extension/RREF work with ``n_processes``
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
    separable_score_cache = {} if score_is_separable else None
    if score_is_separable:
        if score_func is None:
            raise ValueError("score_is_separable=True requires a score_func.")
        SeparableScoreEvaluator(
            score_func,
            n_qubits,
            cache=separable_score_cache,
        ).prime(candidate_pool)

    with _make_process_pool(
        n_processes,
        candidate_pool,
        n_qubits,
        2 * n_qubits,
        mp_start_method,
        separable_score_cache,
    ) as process_pool:
        syms = beam_search_symmetries(
            hamiltonian,
            candidate_pool,
            target_rank=target_rank,
            n_qubits=n_qubits,
            beam_width=beam_width,
            heavy_core_fraction=heavy_core_fraction,
            initial_generators=seed_generators,
            score_func=score_func,
            score_is_separable=score_is_separable,
            separable_score_cache=separable_score_cache,
            n_processes=n_processes,
            mp_start_method=mp_start_method,
            process_pool=process_pool,
        )

        if do_local_refine:
            syms = local_swap_refine(
                hamiltonian,
                syms,
                candidate_pool,
                n_qubits=n_qubits,
                max_passes=local_refine_passes,
                score_func=score_func,
                score_is_separable=score_is_separable,
                separable_score_cache=separable_score_cache,
                n_processes=n_processes,
                mp_start_method=mp_start_method,
                process_pool=process_pool,
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
