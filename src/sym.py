import numpy as np
from src.op_utils import *
from src.gf2_utils import *
from openfermion import count_qubits, jordan_wigner, QubitOperator
from src.metrics import universal_grading
from src.op_utils import truncate_qubitop
from dataclasses import dataclass

# def find_approx_symm(H, n_sym=None, num_intervals=100, eps_max=None, verbose=True, print_new=True, sym_metric_func=None):
#     """
#     DO NOT USE THIS!!!
#     Perform additive symmetry sweep over epsilon truncation thresholds.
    
#     For each epsilon in grid, truncates H, builds constraint matrix A,
#     finds nullspace basis, and maintains an additive (nested) basis
#     across all epsilon values.

#     if excessive symmetries found at the final step, checks non-commutator and chooses lowest to fill up.
    
#     Parameters
#     ----------
#     H : QubitOperator
#         Full Hamiltonian.
#     num_intervals : int
#         Number of epsilon discretization points.
#     eps_max : float, optional
#         Maximum epsilon. If None, uses 1.000001 * max|coeff|.
#     verbose : bool
#         Print progress messages.
#     print_new : bool
#         Print when new symmetries are discovered.
#     sym_metric_func : callable function
#         Metric to ** MINIMIZE ** when excessive symmetries found. Defaults to commutator Pauli L1 norm.
        
#     Returns
#     -------
#     op_add : List of Symmetries in order found
#     add_epsilon : thresholds they were found at
#     """
#     # Fix n once from original H
#     n_qubits = count_qubits(H)

#     if sym_metric_func is None:
#         #defaults to noncommuting
#         sym_metric_func = lambda s: np.real(universal_grading([s], H))

#     # Epsilon grid
#     max_abs = max((abs(c) for c in H.terms.values()), default=0.0)
#     if eps_max is None:
#         eps_max = max_abs * 1.000001

#     if n_sym is None:
#         n_sym = n_qubits

#     assert n_sym <= n_qubits, "Invalid number of symmetries {} requested for {} qubit Hamiltonian".format(n_sym, n_qubits)

#     eps_grid = np.linspace(0.0, eps_max, num_intervals)

#     basis_add = np.zeros((0, 2 * n_qubits), dtype=np.uint8)
#     op_add = []
#     add_epsilon = []

#     for idx, eps in enumerate(eps_grid):
#         # Truncate H
#         Ht = truncate_qubitop(H, float(eps))

#         # Build constraint matrix A from truncated H
#         Gt, _, _, _ = qubitop_to_G_matrix(Ht, n=n_qubits)
#         A = exchange_Gx_Gz(Gt, n_qubits)

#         # Find symmetries = nullspace(A)
#         basis = gf2_find_commuting_basis(Gt, n_qubits)# gf2_nullspace(A)
#         basis_rref, piv = gf2_rref(basis)

#         # Additive extension
#         # TODO ensure elements in added obtained from gf2_extend_basis_additive have low hamming weight
#         # DOESN"T CHECK ORTHOGONALITY/COMMUTING PROPERTY this only checks independence
#         basis_add, added = gf2_extend_basis_additive(basis_add, basis_rref) ### is rref the best to do? Would the choice of product ma

#         # Sanity check: additive basis should lie in current nullspace
#         ok_null = gf2_check_commuting(Gt, basis_add, n_qubits)# gf2_check_in_nullspace(A, basis_add)
#         assert ok_null, "Symmetry set not in null space of truncated Hamiltonian!"

#         # Convert to strings
#         add_strs = [symplectic_to_pauli_string(v, n_qubits) for v in added] if added.size else []
#         basis_add_strs = [symplectic_to_pauli_string(v, n_qubits) for v in basis_add] if basis_add.size else []

#         syms = [QubitOperator(add_str, 1.0) for add_str in add_strs]
#         syms_sorted = sorted(syms, key=lambda sym: sym_metric_func(sym), reverse=False)
        
#         if len(add_strs) + len(op_add) > n_sym:
#             #rank and choose
#             print("Excessive symmetries found, ranking and selecting best.")
            
#             n_req = n_sym - len(op_add)
#             op_add.extend(syms_sorted[:n_req])
#             print("Selected symmetries:")
#             for sym in syms_sorted[:n_req]:
#                 print(sym, " with metric value ", sym_metric_func(sym))
            
#             add_epsilon.append([eps]*n_req)
            
#             return op_add, add_epsilon
#         else:
#             for i, sym in enumerate(syms_sorted): #for every added symmetry
#                 add_epsilon.append(eps)
#                 op_add.append(sym)
#                 if verbose: print("Added Pauli string: {} at threshold: {} with metric value: {}".format(sym, eps, sym_metric_func(sym)))

#         if len(basis_add) >= n_sym:
#             return op_add, add_epsilon
    
#     print("Error: Did not find {} symmetries, only {} found!!".format(n_sym, len(op_add)))
#     return op_add, add_epsilon

def get_seniority_symmetries(n_qubits):
    return [QubitOperator('Z{} Z{}'.format(2*i, 2*i+1), 1.0) for i in range(n_qubits//2)]

def get_quartic_symmetries(n_qubits):
    Z_sen = get_seniority_symmetries(n_qubits)
    quar_syms = []
    for i in range((n_qubits//2)-1):
        quar_syms.append(Z_sen[i]*Z_sen[i+1])
    
    quar_syms.append(Z_sen[0]*Z_sen[-1])

    return quar_syms

def hct_mod(HQ, n_sym=None, sym_metric_func = None, use_coeffs_eps=False, num_intervals=100, eps_max=None, verbose=True, add_gen_type='rref', tol=1e-5):
    """
    HCT, _Praveen's version_ (slightly distinct from HCT paper)

    use_coeffs_eps - if set True - then use all non-negligible coefficients as thresholds

    """

    n_qubits = count_qubits(HQ)
    max_abs = max((abs(c) for c in HQ.terms.values()), default=0.0)

    #defaults
    if n_sym is None: n_sym = n_qubits
    if sym_metric_func is None: sym_metric_func = lambda s: np.real(universal_grading([s], HQ)) # Pauli L1 of NC
    if eps_max is None: eps_max = max_abs * 1.000001

    #input checks
    assert n_sym <= n_qubits, "Invalid number of symmetries {} requested for {} qubit Hamiltonian".format(n_sym, n_qubits)

    if use_coeffs_eps:
        eps_grid = [0.0]
        eps_grid.extend(sorted([np.abs(c) for c in truncate_qubitop(HQ, tol).terms.values()]))
    else:
        eps_grid = np.linspace(0.0, eps_max, num_intervals)
    
    S = np.zeros((0, 2 * n_qubits), dtype=np.uint8)
    Symmetries = [] #QubitOperator representations of added S
    add_epsilon = [] #eps at which added symmetries where found

    for idx, eps in enumerate(eps_grid):
        # Truncate H
        Ht = truncate_qubitop(HQ, float(eps))

        G, _, _, _ = qubitop_to_G_matrix(Ht, n=n_qubits)
        S_de = gf2_symp_nullspace(G, n_qubits, True)

        # SS = Sde \int null_symp(S) generating set for new symmetries commute with existing
        SS = gf2_intersection(S_de, gf2_symp_nullspace(S, n_qubits), n_qubits)

        # SS\S ensuring generating set is "new" (ie increases gf2 rank when appended to S)
        C = gf2_complement(SS, S)

        # modify generating set to satisfy some desirable property TODO (or) generate candidate pool to consider (for generalization)
        if add_gen_type == 'rref':
            C_mod, _ = gf2_rref(C)

        #prepare candidate list
        C_mod_ops = [(c, QubitOperator(symplectic_to_pauli_string(c, n_qubits), 1.0)) for c in C_mod]
        C_mod_sorted = sorted(C_mod_ops, key=lambda c: sym_metric_func(c[1]), reverse=False)

        #add to S and Symmetries if rank increases
        for c, sym in C_mod_sorted:
            if len(Symmetries) >= n_sym:
                return Symmetries, add_epsilon
            
            #check of c is independent of current S (can become dependent after addition of previous symmetries at same eps)
            if gf2_rank(concatenate_matrices(S, np.array([c]))) > gf2_rank(S) and gf2_check_commuting(np.array([c]), S, n_qubits):
                S = concatenate_matrices(S, np.array([c]))
                Symmetries.append(sym)
                add_epsilon.append(eps)

                print(sym, " added at threshold {} with metric value ".format(eps), sym_metric_func(sym))
        
        if len(Symmetries) >= n_sym:
            return Symmetries, add_epsilon
    
    assert False, print("Insufficient symmetries {}/{} found, check for bugs/logical errors!".format(len(Symmetries), n_sym))

@dataclass
class SearchStateHCT:
    basis: list
    syms: list
    score: float
    eps_list: list[float] # thresholds at which Symmetries were added

class Beam:
    """
    Beam for beam search - holds objects with a score data field to compare.
    Score is minimized!!

    """
    def __init__(self, beam_width: int, candidates = None):
        self.beam_width = beam_width
        self.candidates = [] if candidates is None else list(candidates)
    
    def sort_beam(self):
        """
        Sorts beam candidates according to score
        """
        self.candidates = sorted(self.candidates, key=lambda C: C.score, reverse=False)
    
    def num_candidates(self):
        return len(self.candidates)
    
    def prune(self):
        """
        Remove bad branches
        """
        excess = self.num_candidates() - self.beam_width
        if excess > 0:
            self.sort_beam()
            self.candidates = self.candidates[:self.beam_width]

            return excess
        else:
            return 0
    
    def add_candidate(self, candidate, check_identical_func = None, prune=False):
        """
        Adds new candidate to beam  
        optionally checks if candidate is identical to existing candidates using provided function - If found does not add new.

        """
        if check_identical_func is None:
            self.candidates.append(candidate)
        else:
            if all([check_identical_func(candidate, C) == False for C in self.candidates]):
                self.candidates.append(candidate)
        
        if prune:
            return self.prune()

    def add_candidates(self, candidates, check_identical_func = None, prune=False):
        for candidate in candidates:
            self.add_candidate(candidate, check_identical_func=check_identical_func, prune=False)
        
        if prune:
            return self.prune()

    def best_candidate(self):
        self.sort_beam()
        return self.candidates[0]

def bs_hct(HQ, n_sym=None, beam_width=16, list_sym_metric_func = None, sym_metric_func = None, use_coeffs_eps=False, num_intervals=100, eps_max=None, verbose=True, add_gen_type='rref', tol=1e-5):
    """
    HCT with BS to keep track of multiple candidates, for lower cost metric list_sym_metric_func
    stops when all candidate sets have n_sym symmetries

    HCT params
    use_coeffs_eps : if set True - then use all non-negligible coefficients as thresholds
    sym_metric_func : single symmetry ranking cost function for HCT candidate ranking
    use_coeffs_eps : Uses abs(term coeffs) for thresholds (essentially drops one term at a time)
    num_intervals : Truncation scheduling, number of divisions in max coeff - min coeffs
    add_gen_type : determines how the generating set for the candidate symmetries is formed (when there is multiple identified at one eps).

    BS params
    beam_size : number of candidate sets to keep track
    list_sym_metric_func : cost function for candidate symmetry set (defaults to average of sym_metric_func)

    """
    def extend_candidates(candidates, new_direction, n_qubits, eps=0):
        new_candidates = []

        for candidate in candidates:
            #check if new direction is commuting and new
            if gf2_rank(concatenate_matrices(candidate.basis, np.array([new_direction]))) > gf2_rank(candidate.basis) and gf2_check_commuting(np.array([new_direction]), candidate.basis, n_qubits):
                new_basis = concatenate_matrices(candidate.basis, np.array([new_direction]))
                new_syms = candidate.syms + [QubitOperator(symplectic_to_pauli_string(new_direction, n_qubits), 1.0)]
                new_candidates.append(SearchStateHCT(new_basis, syms = new_syms, score=list_sym_metric_func(new_syms), eps_list=candidate.eps_list+[eps]))
        
        return new_candidates

    def check_same_func(A, B):
        ra, rb, rab = gf2_rank(A.basis), gf2_rank(B.basis), gf2_rank(concatenate_matrices(A.basis, B.basis))
        return ra == rb and ra == rab

    n_qubits = count_qubits(HQ)
    max_abs = max((abs(c) for c in HQ.terms.values()), default=0.0)

    #defaults
    if n_sym is None: n_sym = n_qubits
    if sym_metric_func is None: sym_metric_func = lambda sym: np.real(universal_grading([sym], HQ)) # Pauli L1 of NC
    if list_sym_metric_func is None: list_sym_metric_func = lambda sym_list: np.sum([sym_metric_func(sym) for sym in sym_list])
    if eps_max is None: eps_max = max_abs * 1.000001

    #input checks
    assert n_sym <= n_qubits, "Invalid number of symmetries {} requested for {} qubit Hamiltonian".format(n_sym, n_qubits)

    if use_coeffs_eps:
        eps_grid = [0.0]
        eps_grid.extend(sorted([np.abs(c) for c in truncate_qubitop(HQ, tol).terms.values()]))
    else:
        eps_grid = np.linspace(0.0, eps_max, num_intervals)
    
    # init with exact symmetries
    G, _, _, _ = qubitop_to_G_matrix(HQ, n=n_qubits)
    directions = gf2_symp_nullspace(G, n_qubits, True)
    exact_syms = [QubitOperator(symplectic_to_pauli_string(c, n_qubits), 1.0) for c in directions]
    n_exact_syms = len(exact_syms)
    n = min(n_sym, n_exact_syms)
    init_candidate = SearchStateHCT(directions, exact_syms[:n], list_sym_metric_func(exact_syms[:n]), [0]*n)

    beams = {}
    if n_exact_syms >= n_sym:
        return init_candidate, beams
    
    for num_syms in range(n_exact_syms, n_sym+1):
        beams[num_syms] = Beam(beam_width=beam_width)
    pruned = beams[n_exact_syms].add_candidate(init_candidate, prune=True)

    #start beaming!
    for idx, eps in enumerate(eps_grid):
        # Truncate H
        Ht = truncate_qubitop(HQ, float(eps))

        G, _, _, _ = qubitop_to_G_matrix(Ht, n=n_qubits)
        S_de = gf2_symp_nullspace(G, n_qubits, True)

        # # SS = Sde \int null_symp(S) generating set for new symmetries commute with existing
        # SS = gf2_intersection(S_de, gf2_symp_nullspace(S, n_qubits), n_qubits)

        # SS\S ensuring generating set is "new" (ie increases gf2 rank when appended to S)
        new_directions = gf2_complement(S_de, directions)
        
        if len(new_directions) == 0:
            #no new directions identified
            continue
        
        if verbose:
            print("At threshold e: {}, {} new directions found!".format(eps, len(new_directions)))

        # modify generating set to satisfy some desirable property TODO (or) generate candidate pool to consider (for generalization)
        if add_gen_type == 'rref':
            new_directions, _ = gf2_rref(new_directions)
        
        #attempt adding to beams upto n_sym-1
        for new_direction in new_directions:
            for sym_count in reversed(range(n_exact_syms, n_sym)):
                beam = beams[sym_count]
                if beam.num_candidates() > 0:
                    #if verbose: print("Adding ", new_direction, " to beam ", sym_count+1)
                    new_candidates = extend_candidates(beam.candidates, new_direction=new_direction, n_qubits=n_qubits, eps=eps)
                    pruned += beams[sym_count+1].add_candidates(new_candidates, check_identical_func=check_same_func, prune=True)
        
        ### this should be fixed
        directions = S_de
        #print("Current directions:\n", directions)
    
    if verbose: print("Completed search, {} pruned branches.".format(pruned))

    if beams[n_sym].num_candidates() == 0:
        print("No sufficient candidates found, debugging required!!")
        return beams
    
    best =  beams[n_sym].best_candidate()
    if verbose: 
        print("Best candidate with score: {}".format(best.score))
        for sym in best.syms:
            print(sym)
    
    return best, beams