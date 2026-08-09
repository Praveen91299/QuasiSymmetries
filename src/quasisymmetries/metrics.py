
from openfermion import commutator, get_sparse_operator, expectation, get_ground_state, hermitian_conjugated, QubitOperator, jordan_wigner, FermionOperator
import numpy as np
from scipy.sparse import csc_matrix, identity as sparse_id
from copy import deepcopy
from .op_utils import freeze_qubits, permute_sym_to_start
from .clifford_symmetry_optimized import Clifford


class PauliTermOverlapCommutatorEvaluator:
    """Matrix-free repeated ``||[H, S] psi||^2`` for Pauli products ``S``.

    For a Pauli product ``S``, only Hamiltonian terms that anticommute with it
    contribute, and ``||[H,S] psi||^2 = 4 ||H_anti(S) psi||^2``.  The action
    of each Hamiltonian Pauli term on a determinant-sparse state is prepared
    once.  Their comparatively small term-by-term Gram matrix is retained;
    the full ``2**n_qubits`` square Hamiltonian matrix is never constructed.
    """

    def __init__(self, hamiltonian, sparse_state):
        from .bs.utils import as_pauli_term_stream
        from .state_utils import PauliActionMask, SparseQubitState

        if not isinstance(sparse_state, SparseQubitState):
            raise TypeError("sparse_state must be a SparseQubitState")
        self.n_qubits = int(sparse_state.n_qubits)
        stream = as_pauli_term_stream(hamiltonian, self.n_qubits)
        terms = stream.terms
        self.hamiltonian_masks = tuple(item.mask for item in terms)
        n_terms = len(terms)
        n_det = int(sparse_state.nnz)
        rows = []
        columns = []
        values = []
        for column, item in enumerate(terms):
            action = PauliActionMask.from_pauli_mask(
                item.mask,
                self.n_qubits,
                item.signed_coefficient,
            )
            rows.append(
                (sparse_state.indices ^ action.flip_mask).astype(
                    np.int32, copy=False
                )
            )
            columns.append(np.full(n_det, column, dtype=np.int32))
            values.append(
                action.phases(sparse_state.indices) * sparse_state.coeffs
            )
        if n_terms:
            action_matrix = csc_matrix(
                (
                    np.concatenate(values),
                    (np.concatenate(rows), np.concatenate(columns)),
                ),
                shape=(1 << self.n_qubits, n_terms),
            )
            self.term_overlap = (action_matrix.getH() @ action_matrix).tocsr()
        else:
            self.term_overlap = csc_matrix((0, 0), dtype=np.complex128).tocsr()

    @staticmethod
    def _anticommutes(first, second):
        first_x, first_z = first
        second_x, second_z = second
        return (
            (bin(int(first_x & second_z)).count("1")
             + bin(int(first_z & second_x)).count("1"))
            & 1
        ) == 1

    def cost_mask(self, symmetry_mask):
        selected = np.fromiter(
            (
                self._anticommutes(term_mask, symmetry_mask)
                for term_mask in self.hamiltonian_masks
            ),
            dtype=np.float64,
            count=len(self.hamiltonian_masks),
        )
        value = 4.0 * np.vdot(selected, self.term_overlap @ selected)
        value = float(np.real_if_close(value))
        if value < 0 and abs(value) < 1e-10:
            value = 0.0
        return value

    def cost(self, symmetries):
        from .bs.utils import term_to_masks

        total = 0.0
        for symmetry in symmetries:
            if len(symmetry.terms) != 1:
                raise ValueError("Every symmetry must be one Pauli product.")
            (term, _coefficient), = symmetry.terms.items()
            total += self.cost_mask(term_to_masks(term, self.n_qubits))
        return total


class GroupedSparsePauliCommutatorEvaluator:
    r"""Evaluate CISD squared-commutator costs without dense operators.

    For a Hermitian Pauli product ``S`` and Pauli Hamiltonian ``H``, this
    evaluator uses

    .. math::

        \langle\psi|[H,S]^\dagger[H,S]|\psi\rangle
        = 4\|H_{\mathrm{anti}(S)}|\psi\rangle\|^2,

    where ``H_anti(S)`` contains exactly the Hamiltonian Pauli terms that
    anticommute with ``S``. Hamiltonian terms with the same computational-basis
    bit-flip pattern are combined before their action is stored. This recovers
    the cancellations among Jordan--Wigner terms and is particularly effective
    for diagonal (Z-only) candidate symmetries.

    Parameters
    ----------
    hamiltonian
        A ``PauliTermStream`` or an OpenFermion ``QubitOperator``.
    sparse_state
        Normalized :class:`~quasisymmetries.state_utils.SparseQubitState`, such
        as the determinant-sparse CISD state.
    cancellation_tolerance
        Absolute tolerance used only to discard numerical roundoff remaining
        after terms with a common flip pattern have been summed.

    Attributes
    ----------
    preparation_seconds
        Wall time used to prepare and cache the grouped Hamiltonian action.
    grouped_action_nnz
        Total number of cached nonzero amplitudes across all flip groups.

    Notes
    -----
    No vector of length ``2**n_qubits`` and no Hamiltonian sparse matrix is
    constructed. Individual results are cached by packed Pauli mask, which is
    useful because HCT may rank the same candidate more than once.
    """

    def __init__(self, hamiltonian, sparse_state, cancellation_tolerance=1e-14):
        from collections import defaultdict
        from time import perf_counter

        from .bs.utils import as_pauli_term_stream
        from .state_utils import PauliActionMask, SparseQubitState

        if not isinstance(sparse_state, SparseQubitState):
            raise TypeError("sparse_state must be a SparseQubitState")
        if cancellation_tolerance < 0:
            raise ValueError("cancellation_tolerance must be nonnegative")
        norm = sparse_state.norm()
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError(f"sparse_state must be normalized; norm={norm}")

        start = perf_counter()
        self.n_qubits = int(sparse_state.n_qubits)
        self.sparse_state = sparse_state
        self.cancellation_tolerance = float(cancellation_tolerance)
        stream = as_pauli_term_stream(hamiltonian, self.n_qubits)
        groups = defaultdict(list)
        self._term_records = []
        for item in stream.terms:
            action = PauliActionMask.from_pauli_mask(
                item.mask,
                self.n_qubits,
                item.signed_coefficient,
            )
            x_mask, z_mask = (int(item.mask[0]), int(item.mask[1]))
            groups[x_mask].append((z_mask, action))
            self._term_records.append((x_mask, z_mask, action))

        self._groups = tuple(
            (x_mask, tuple(records)) for x_mask, records in groups.items()
        )
        self._z_candidate_actions = []
        grouped_action_nnz = 0
        for x_mask, records in self._groups:
            indices, coefficients = self._combined_group_action(records)
            self._z_candidate_actions.append(
                (x_mask, indices, coefficients)
            )
            grouped_action_nnz += len(indices)
        self._z_candidate_actions = tuple(self._z_candidate_actions)
        self.grouped_action_nnz = int(grouped_action_nnz)
        self._cost_cache = {}
        self.preparation_seconds = float(perf_counter() - start)

    @staticmethod
    def _parity(value):
        """Return the population-count parity of a nonnegative integer."""
        return bin(int(value)).count("1") & 1

    def _combined_group_action(self, records):
        """Return coalescible output indices/amplitudes for one flip group."""
        coefficients = np.zeros(
            self.sparse_state.nnz, dtype=np.complex128
        )
        for _z_mask, action in records:
            coefficients += (
                action.phases(self.sparse_state.indices)
                * self.sparse_state.coeffs
            )
        keep = np.abs(coefficients) > self.cancellation_tolerance
        if not np.any(keep):
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.complex128),
            )
        flip_mask = records[0][1].flip_mask
        return (
            (self.sparse_state.indices ^ int(flip_mask))[keep],
            coefficients[keep],
        )

    @staticmethod
    def _coalesced_norm_squared(indices, coefficients):
        """Return the squared norm after summing equal basis indices."""
        if len(indices) == 0:
            return 0.0
        order = np.argsort(indices, kind="stable")
        sorted_indices = indices[order]
        sorted_coefficients = coefficients[order]
        starts = np.concatenate(
            ([0], np.flatnonzero(sorted_indices[1:] != sorted_indices[:-1]) + 1)
        )
        summed = np.add.reduceat(sorted_coefficients, starts)
        return float(np.vdot(summed, summed).real)

    def _selected_group_actions(self, symmetry_mask):
        """Yield grouped actions of terms anticommuting with one candidate."""
        symmetry_x, symmetry_z = map(int, symmetry_mask)
        if symmetry_x == 0:
            for hamiltonian_x, indices, coefficients in self._z_candidate_actions:
                if self._parity(hamiltonian_x & symmetry_z):
                    yield indices, coefficients
            return

        for hamiltonian_x, records in self._groups:
            xz_parity = self._parity(hamiltonian_x & symmetry_z)
            selected = tuple(
                record
                for record in records
                if xz_parity ^ self._parity(record[0] & symmetry_x)
            )
            if selected:
                indices, coefficients = self._combined_group_action(selected)
                if len(indices):
                    yield indices, coefficients

    def cost_mask(self, symmetry_mask):
        r"""Return ``<psi|[H,S]^dagger[H,S]|psi>`` for one packed mask.

        Parameters
        ----------
        symmetry_mask
            ``(x_mask, z_mask)`` using the packed convention in
            :mod:`quasisymmetries.bs.utils`.

        Returns
        -------
        cost
            Nonnegative squared-commutator expectation as a Python ``float``.
        """
        key = tuple(map(int, symmetry_mask))
        if key in self._cost_cache:
            return self._cost_cache[key]
        pieces = list(self._selected_group_actions(key))
        if pieces:
            indices = np.concatenate([piece[0] for piece in pieces])
            coefficients = np.concatenate([piece[1] for piece in pieces])
            value = 4.0 * self._coalesced_norm_squared(indices, coefficients)
        else:
            value = 0.0
        if value < 0 and abs(value) < 1e-12:
            value = 0.0
        self._cost_cache[key] = float(value)
        return float(value)

    def cost(self, symmetries):
        r"""Return the sum of individual squared-commutator expectations.

        Parameters
        ----------
        symmetries
            Iterable of single-product OpenFermion ``QubitOperator`` objects.

        Returns
        -------
        total_cost
            Sum of ``<psi|[H,S_k]^dagger[H,S_k]|psi>`` over the inputs.
        """
        from .bs.utils import term_to_masks

        total = 0.0
        for symmetry in symmetries:
            if len(symmetry.terms) != 1:
                raise ValueError("Every symmetry must be one Pauli product.")
            (term, coefficient), = symmetry.terms.items()
            if not np.isclose(abs(coefficient), 1.0, atol=1e-12):
                raise ValueError("Every symmetry must have a unit-modulus coefficient.")
            total += self.cost_mask(term_to_masks(term, self.n_qubits))
        return float(total)

def construct_projectors(sym_list: list[QubitOperator]):
    """
    Construct projectors to all subspaces defined by Pauli symmetries sym_list

    """
    if len(sym_list) == 0:
        return [QubitOperator('', coefficient=1.0)]
    
    projectors = []

    sym = sym_list[0]
    projectors_rec = construct_projectors(sym_list=sym_list[1:])
    for proj in projectors_rec:
        projectors.append((0.5 + 0.5 * sym)*proj)
        projectors.append((0.5 - 0.5 * sym)*proj)
    return projectors

def construct_projectors_sparse(sym_list_sparse: list, n_qubits):
    if len(sym_list_sparse) == 0:
        return [sparse_id(1<<n_qubits)]
    
    projectors = []

    sym_sparse = sym_list_sparse[0]
    projectors_rec = construct_projectors_sparse(sym_list_sparse=sym_list_sparse[1:], n_qubits=n_qubits)
    for proj in projectors_rec:
        projectors.append(0.5 * (sparse_id(1<<n_qubits) + sym_sparse)@proj)
        projectors.append(0.5 * (sparse_id(1<<n_qubits) - sym_sparse)@proj)
    return projectors

def get_sector_projectors(list_sym, sectors, n_qubits):
    """
    
    
    """

    n_sym = len(list_sym)
    list_sym_sparse = [get_sparse_operator(sym, n_qubits) for sym in list_sym]
    proj = []

    for sec in sectors:
        sec_proj = sparse_id(1<<n_qubits)
        assert len(sec) == n_sym

        for i, s in enumerate(sec):
            assert s == 1 or s == -1, "Invalid sector label {}".format(s)
            sec_proj = sec_proj @ (0.5*(sparse_id(1<<n_qubits) + s*list_sym_sparse[i]))

        proj.append(sec_proj)
    
    return proj

def find_overlaps(sym_ops, state, n_qubits):
    r"""
    Find coefficients of state in different symmetry subspaces

    <\psi Pi_s \psi> for all s vectors

    """
    projectors = construct_projectors(sym_ops)
    return [expectation(get_sparse_operator(proj, n_qubits), state) for proj in projectors]

def entropy(probs, tol=1e-5, log_base='2'):
    """
    Entropy (bits) of given probability distribution, truncates to entries >= tol

    """

    probs_trunc = []
    for p in probs:
        if abs(p) >= tol:
            probs_trunc.append(p)

    probs_trunc = np.array(probs_trunc)
    if log_base == '2' or log_base == 2:
        return np.sum(probs_trunc * np.log2(1/probs_trunc))
    elif log_base == 'e' or log_base == np.e:
        return np.sum(probs_trunc * np.log(1/probs_trunc))

def entropy_pauli_sym(projectors_sparse, state, n_qubits):
    return entropy([expectation(proj, state) for proj in projectors_sparse])

def entropy_pauli_syms(sym_ops, state, n_qubits, verbose=False):
    sym_sparse = [get_sparse_operator(sym, n_qubits) for sym in sym_ops]
    projs = construct_projectors_sparse(sym_sparse, n_qubits)
    ent = entropy_pauli_sym(projs, state, n_qubits)
    if verbose: print("Cut entropy: ", ent)
    return ent
    
def l1norm(op: QubitOperator, remove_const=False):
    """
    Returns Pauli L1

    """
    l1= np.sum(np.abs(list(op.terms.values())))
    if not remove_const:
        return l1
    return l1 - np.abs(op.constant)

def universal_grading(sym_ops, H, verbose=False):
    """
    Returns sum of Paulil1 of [S_i, H]

    """
    nc = sum([l1norm(commutator(sym, H)) for sym in sym_ops])
    if verbose: print("Non commutative l1: ", nc)
    return nc

def variance(sym_ops, state, n_qubits, verbose=False):
    v = np.sum([1 - expectation(get_sparse_operator(sym_op, n_qubits), state)**2 for sym_op in sym_ops])
    if verbose: print("Variance: ", v)
    return v

def find_commuting_paulis(H, sym_ops, verbose=False):
    """
    Finds Pauli products in H that commute with all sym_ops
    returns list[QubitOperators] of commuting terms

    H: QubitOperator
    sym_ops: list[QubitOperator]
    """
    def is_commuting(op1, op2, tol):
        comm = commutator(op1, op2)
        comm.compress()
        return np.isclose(np.sum(np.abs(list(comm.terms.values()))), 0, rtol=tol)
    
    HQ = deepcopy(H)
    c = HQ.constant
    HQ = HQ - c
    HQ.compress()
    n_total_pauli =  len(HQ.terms.keys())

    commuting_terms = []
    for term, coeff in HQ.terms.items():
        Pauli =  QubitOperator(term, coeff)

        if all([is_commuting(sym_op, Pauli, 1e-5) for sym_op in sym_ops]):
            commuting_terms.append(Pauli)
    
    if verbose: print("{}/{} Terms in H found to commute with all symmetries.".format(len(commuting_terms), n_total_pauli))

    return commuting_terms

def find_commuting_terms(H, sym_ops, verbose=False):
    """
    Finds Fermion strings in H that commute with all sym_ops
    """
    def is_commuting(op1, op2, tol):
        comm = commutator(op1, op2)
        comm.compress()
        return np.isclose(np.sum(np.abs(list(comm.terms.values()))), 0, rtol=tol)
    
    HQ = deepcopy(H)
    c = HQ.constant
    HQ = HQ - c
    HQ.compress()
    n_total =  len(H.terms.keys())

    commuting_terms = []
    for term, coeff in HQ.terms.items():
        t =  FermionOperator(term, coeff)

        if all([is_commuting(sym_op, jordan_wigner(t), 1e-5) for sym_op in sym_ops]):
            commuting_terms.append(t)
    
    if verbose: print("{}/{} Terms in H found to commuting with all symmetries.".format(len(commuting_terms), n_total))

    return commuting_terms

def comm_sq_exp_fast(sym_ops, H, state, n_qubits, verbose=False):
    """
    Compute sum_k <state| ( i[H, S_k] )^2 |state> efficiently.

    Parameters
    ----------
    sym_ops : list[QubitOperator]
        Symmetry operators (Pauli products).
    H : sparse operator
        Hamiltonian.
    state : np.ndarray
        State vector.
    n_qubits : int

    Returns
    -------
    float or complex
    """
    
    psi = np.asarray(state)

    # Reused for every symmetry operator
    Hpsi = H @ psi

    total = 0.0 + 0.0j
    for sym in sym_ops:
        S = get_sparse_operator(sym, n_qubits).tocsr()

        Spsi = S @ psi
        delta = 1j * ((H @ Spsi) - (S @ Hpsi))   # delta = i[H,S]|psi>

        # <psi| (i[H,S])^2 |psi> = || delta ||^2
        total += np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp

def comm_sq_exp_pauli_actions(pauli_actions, H, state, verbose=False, weights=None, Hpsi=None):
    """
    comm_sq_exp using direct Pauli-product permutation/phase actions.

    This avoids sparse S_k matvecs.  It is only valid when each symmetry is a
    single Pauli product.
    """
    psi = np.asarray(state).reshape(-1)
    H = H.tocsr()
    if Hpsi is None:
        Hpsi = H @ psi
    else:
        Hpsi = np.asarray(Hpsi).reshape(-1)

    if weights is None:
        weights = np.ones(len(pauli_actions))

    total = 0.0 + 0.0j
    for weight, action in zip(weights, pauli_actions):
        if weight == 0:
            continue
        Spsi = action.apply(psi)
        SHpsi = action.apply(Hpsi)
        delta = 1j * ((H @ Spsi) - SHpsi)
        total += weight * np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp

def prepare_sparse_symmetries(sym_ops, n_qubits):
    """
    Convert symmetry QubitOperators to CSR matrices once for repeated metrics.
    """
    return [get_sparse_operator(sym, n_qubits).tocsr() for sym in sym_ops]

def comm_sq_exp_sparse_syms(sym_ops_sparse, H, state, verbose=False, weights=None):
    """
    Repeated-evaluation version of comm_sq_exp_fast.

    sym_ops_sparse should be the output of prepare_sparse_symmetries.  Avoiding
    get_sparse_operator inside every objective call matters during orbital
    optimization.
    """
    psi = np.asarray(state).reshape(-1)
    H = H.tocsr()
    Hpsi = H @ psi

    if weights is None:
        weights = np.ones(len(sym_ops_sparse))

    total = 0.0 + 0.0j
    for weight, S in zip(weights, sym_ops_sparse):
        if weight == 0:
            continue

        Spsi = S @ psi
        delta = 1j * ((H @ Spsi) - (S @ Hpsi))
        total += weight * np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp


def get_entropies_at_cuts(state, n_qubits, log_base='2'):
    """
    Get bi-partite entanglement across all partitions of qubits

    state: np.array - state
    log_base: str - '2' or 'e'
    """
    entropies = []
    for k in range(1, n_qubits):
        d = np.linalg.svd(
            np.reshape(state, (1 << k, 1 << (n_qubits - k))),
            compute_uv=False,
            full_matrices=False,
        )

        entropies.append(entropy(np.abs(d)**2, log_base=log_base))
    return entropies

def get_ent(
    symmetries,
    HQ,
    n_qubits,
    verbose=False,
    return_state=False,
    return_sparse_clifford=False,
    log_base='2',
    synthesis_basis="X",
    generator_mapping="row_reduced",
):
    """
    Get bi-partite entanglement across all partitions after diagonalizing symmetries and localizing them to qubits 0, 1, 2, ... in order

    """
    if len(symmetries) > 0:
        H_perm = permute_sym_to_start(
            HQ,
            symmetries,
            n_qubits,
            verbose=verbose,
            synthesis_basis=synthesis_basis,
            generator_mapping=generator_mapping,
        )
    else:
        if verbose: print("No symmetries passed, returning original bond entanglements.")
        H_perm = HQ
    e_p, gs = get_ground_state(get_sparse_operator(H_perm, n_qubits))

    ents = get_entropies_at_cuts(gs, n_qubits, log_base=log_base)
    if verbose:
        print("Entropy of cuts (bits):")
        for i, e in enumerate(ents):
            print("{} | {} : {}".format(i+1, i+2, e))
    
    if return_state:
        return ents, H_perm, gs
    else:
        return ents, H_perm
    
def int_to_binary_list(x: int, n: int, MSB_first=True) -> list[int]:
    """
    Convert a nonnegative integer x to a length-n list of binary digits.

    The most significant bit comes first.

    Example:
        int_to_binary_list(6, 4) -> [0, 1, 1, 0]
    """
    if x < 0:
        raise ValueError("x must be nonnegative")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if x >= (1 << n):
        raise ValueError(f"x={x} cannot be represented with {n} bits")

    b = [(x >> i) & 1 for i in reversed(range(n))]

    if MSB_first:
        return b
    else:
        return list(reversed(b))

def get_single_sector_energies(
    HQ,
    list_sym,
    n_qubits,
    verbose=False,
    synthesis_basis="X",
    generator_mapping="row_reduced",
):
    """
    Find ground state in symmetry sectors

    Rotates Hamiltonian and then freezes qubits, following with it solves for ground state energy

    """

    n_sym = len(list_sym)
    n_qubits_red = n_qubits - n_sym
    #all combinations

    H_perm = permute_sym_to_start(
        HQ,
        list_sym,
        n_qubits,
        False,
        synthesis_basis=synthesis_basis,
        generator_mapping=generator_mapping,
    )
    frozen_qubits = list(range(n_sym))

    gs_e_list = []
    for i in range(1<<n_sym):
        sec_label = int_to_binary_list(i, n_sym, MSB_first=False)
        sec_dict = {s: v for s, v in zip(frozen_qubits, sec_label)}
        H_red_sec = freeze_qubits(H_perm, sec_dict)

        gs_e, gs = get_ground_state(get_sparse_operator(H_red_sec, n_qubits_red))
        gs_e_list.append(gs_e)
    
    if verbose: print("Minimum single sector energy: ", np.min(gs_e_list))
    return gs_e_list

def get_bipartite_mps(HQ, n_qubits, target_energy=None, bd=100, n_sweeps=100, tol=1.6e-3):
    """
    Uses pyblock2 to solve and calculate the bipartite entanglement

    """
    np.random.seed(0)
    
    from .tn import QO_to_block2_MPO

    mpo, driver = QO_to_block2_MPO(HQ, n_qubits)
    ket = driver.get_random_mps(tag="KET", bond_dim=bd, nroots=1)

    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=n_sweeps,
        bond_dims=None,
        noises=[1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6] + [0.0]*(n_sweeps - 6),
        thrds=[1e-10] * n_sweeps,
        dav_max_iter=50,
        iprint=0
    )

    if target_energy is not None:

        if np.abs(energy - target_energy) < tol:
            print("Bipartite entanglement: Warning dmrg not converged to reference energy...")

    return driver.get_bipartite_entanglement(ket), ket

def get_permuted_bipartite_entanglement(
    symmetries,
    HQ,
    n_qubits,
    fci_energy=None,
    fci_gs=None,
    verbose=False,
    return_state=False,
    return_U=False,
    log_base='e',
    use_dmrg=False,
    return_clifford=False,
    synthesis_basis="X",
    generator_mapping="row_reduced",
):
    """
    Get bi-partite entanglement across all partitions after diagonalizing symmetries and localizing them to qubits 0, 1, 2, ... in order
    *Modified version of get_ent with dmrg calculations for speed.*

    ``synthesis_basis`` and ``generator_mapping`` are forwarded to
    ``Clifford.from_symmetries``. Their defaults preserve the historical
    row-reduced mapping.
    """
    #permute
    if len(symmetries) > 0:
        H_perm, clifford, perm = permute_sym_to_start(
            HQ,
            symmetries,
            n_qubits,
            verbose=verbose,
            return_clifford_perm=True,
            synthesis_basis=synthesis_basis,
            generator_mapping=generator_mapping,
        )
    else:
        if verbose: print("No symmetries passed, returning original bond entanglements.")
        H_perm = HQ
        clifford = Clifford(
            n_qubits,
            synthesis_basis=synthesis_basis,
            generator_mapping=generator_mapping,
        )
        perm = list(clifford.permutation)
    
    #solve
    if use_dmrg:
        ents, gs = get_bipartite_mps(H_perm, n_qubits, target_energy=fci_energy)

        if log_base == '2':
            ents = ents / np.log(2)
    else:
        if fci_gs is not None:
            #transform state directly
            gs = clifford.transform_state(fci_gs)
        else:
            e_p, gs = get_ground_state(get_sparse_operator(H_perm, n_qubits))
            if fci_energy is not None: assert np.isclose(fci_energy, e_p, atol=1e-5), "Permuted Hamiltonian ground state differs from fci by {}".format(fci_energy - e_p)
        ents = get_entropies_at_cuts(gs, n_qubits, log_base=log_base)
    
    if verbose:
        print("Entropy of cuts (log base = {}):".format(log_base))
        for i, e in enumerate(ents):
            print("{} | {} : {}".format(i+1, i+2, e))

    # Construct the requested return tuple without changing existing callers.
    if return_clifford:
        transform_object = clifford
    elif return_U:
        # Backward compatibility for callers that explicitly requested U.
        transform_object = clifford.sparse_matrix
    else:
        transform_object = None

    if transform_object is not None:
        if return_state:
            result = (ents, H_perm, transform_object, gs)
        else:
            result = (ents, H_perm, transform_object)
    else:
        if return_state:
            result = (ents, H_perm, gs)
        else:
            result = (ents, H_perm)

    return result
