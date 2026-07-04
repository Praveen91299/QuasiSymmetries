import itertools
import numpy as np
from scipy.optimize import linprog

from openfermion import FermionOperator, normal_ordered, jordan_wigner


def compress_fermion(op, tol=1e-12):
    op = normal_ordered(op)
    op.compress(abs_tol=tol)
    return op


def compress_qubit(op, tol=1e-12):
    op.compress(abs_tol=tol)
    return op


def infer_n_spin_orbitals(H):
    max_idx = -1
    for term in H.terms:
        for p, _ in term:
            max_idx = max(max_idx, p)
    return max_idx + 1


def number_operator(n_orb):
    N = FermionOperator()
    for p in range(n_orb):
        N += FermionOperator(((p, 1), (p, 0)), 1.0)
    return compress_fermion(N)


def real_one_body_generators(n_orb, include_diagonal=True):
    """
    Real Hermitian one-body generators F_ij.

    Diagonal:
        F_pp = a_p^ a_p

    Off-diagonal:
        F_pq = a_p^ a_q + a_q^ a_p, p < q

    This assumes a real electronic Hamiltonian.
    """
    F_list = []

    if include_diagonal:
        for p in range(n_orb):
            F_list.append(
                FermionOperator(((p, 1), (p, 0)), 1.0)
            )

    for p, q in itertools.combinations(range(n_orb), 2):
        F_pq = (
            FermionOperator(((p, 1), (q, 0)), 1.0)
            + FermionOperator(((q, 1), (p, 0)), 1.0)
        )
        F_list.append(F_pq)

    return [compress_fermion(F) for F in F_list]


def bliss_paper_killers_real(
    n_orb,
    n_electrons,
    include_diagonal_F=True,
    tol=1e-12,
):
    """
    Construct killers faithful to the LP-BLISS form:

        K(mu, xi)
        = mu1 * (N - Ne)
        + mu2 * (N^2 - Ne^2)
        + sum_ij xi_ij F_ij * (N - Ne)

    with real Hermitian one-body F_ij.

    Returns a list of FermionOperator columns:
        [K_mu1, K_mu2, K_xi_0, K_xi_1, ...]
    """
    I = FermionOperator((), 1.0)
    N = number_operator(n_orb)
    Ne = float(n_electrons)

    N_minus_Ne = compress_fermion(N - Ne * I, tol)
    N2_minus_Ne2 = compress_fermion(N * N - (Ne ** 2) * I, tol)

    killers = []

    # mu_1 term
    killers.append(N_minus_Ne)

    # mu_2 term
    killers.append(N2_minus_Ne2)

    # xi_ij F_ij (N - Ne) terms
    for F in real_one_body_generators(
        n_orb,
        include_diagonal=include_diagonal_F,
    ):
        K = compress_fermion(F * N_minus_Ne, tol)
        if len(K.terms) > 0:
            killers.append(K)

    return killers

def qubit_l1_norm(op):
    return float(sum(abs(complex(c)) for c in op.terms.values()))


def qubit_to_real_vector(op, term_list, tol=1e-10):
    vec = []
    max_imag = 0.0

    for term in term_list:
        c = complex(op.terms.get(term, 0.0))
        max_imag = max(max_imag, abs(c.imag))
        vec.append(c.real)

    if max_imag > tol:
        raise ValueError(
            f"Non-negligible imaginary Pauli coefficient encountered: {max_imag}"
        )

    return np.array(vec, dtype=float)


def lp_bliss_paper_real_pauli_1norm(
    H,
    n_electrons,
    n_orb=None,
    mapper=jordan_wigner,
    include_diagonal_F=True,
    tol=1e-10,
    lp_options=None,
):
    """
    LP-BLISS Hamiltonian modifier faithful to:

        K = mu1 (N - Ne)
          + mu2 (N^2 - Ne^2)
          + sum_ij xi_ij F_ij (N - Ne)

    for real electronic Hamiltonians.

    Minimizes the Pauli 1-norm after fermion-to-qubit mapping:

        min || mapper(H - K) ||_1
    """
    H = compress_fermion(H, tol)

    if n_orb is None:
        n_orb = infer_n_spin_orbitals(H)

    killers = bliss_paper_killers_real(
        n_orb=n_orb,
        n_electrons=n_electrons,
        include_diagonal_F=include_diagonal_F,
        tol=tol,
    )

    Q_H = compress_qubit(mapper(H), tol)
    Q_K = [compress_qubit(mapper(K), tol) for K in killers]

    all_pauli_terms = set(Q_H.terms.keys())
    for Q in Q_K:
        all_pauli_terms.update(Q.terms.keys())

    term_list = sorted(all_pauli_terms)

    h = qubit_to_real_vector(Q_H, term_list, tol)
    A = np.column_stack([
        qubit_to_real_vector(Q, term_list, tol)
        for Q in Q_K
    ])

    n_vars = A.shape[1]
    n_paulis = A.shape[0]

    # Variables:
    #   x = [mu1, mu2, xi_0, xi_1, ...]
    #   t = absolute-value slack variables
    #
    # Objective:
    #   min sum_i t_i
    #
    # Constraints:
    #   -t <= h - A x <= t
    c = np.concatenate([
        np.zeros(n_vars),
        np.ones(n_paulis),
    ])

    A_ub = np.vstack([
        np.hstack([-A, -np.eye(n_paulis)]),
        np.hstack([ A, -np.eye(n_paulis)]),
    ])

    b_ub = np.concatenate([-h, h])

    bounds = [(None, None)] * n_vars + [(0.0, None)] * n_paulis

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options=lp_options,
    )

    if not result.success:
        return H, {
            "success": False,
            "message": result.message,
            "initial_pauli_l1": qubit_l1_norm(Q_H),
            "final_pauli_l1": qubit_l1_norm(Q_H),
            "lp_result": result,
        }

    coeffs = result.x[:n_vars]

    K_opt = FermionOperator()
    for c_j, K_j in zip(coeffs, killers):
        if abs(c_j) > tol:
            K_opt += float(c_j) * K_j

    K_opt = compress_fermion(K_opt, tol)
    H_bliss = compress_fermion(H - K_opt, tol)

    Q_bliss = compress_qubit(mapper(H_bliss), tol)

    initial_l1 = qubit_l1_norm(Q_H)
    final_l1 = qubit_l1_norm(Q_bliss)

    info = {
        "success": True,
        "message": result.message,
        "coefficients": coeffs,
        "mu1": coeffs[0],
        "mu2": coeffs[1],
        "xi": coeffs[2:],
        "killers": killers,
        "K_opt": K_opt,
        "H_qubit_initial": Q_H,
        "H_qubit_final": Q_bliss,
        "initial_pauli_l1": initial_l1,
        "final_pauli_l1": final_l1,
        "pauli_l1_reduction": initial_l1 - final_l1,
        "relative_pauli_l1_reduction": (
            1.0 - final_l1 / initial_l1 if initial_l1 > 0 else 0.0
        ),
        "n_killers": len(killers),
        "n_pauli_terms_initial": len(Q_H.terms),
        "n_pauli_terms_final": len(Q_bliss.terms),
        "lp_result": result,
    }

    return H_bliss, info