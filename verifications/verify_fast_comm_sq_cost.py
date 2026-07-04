"""
Verify the faster orbital commutator-squared objective.

Run from QuasiSymmetries:
    python verifications/verify_fast_comm_sq_cost.py
"""
import time
import sys
from pathlib import Path

import numpy as np
from openfermion import QubitOperator, get_sparse_operator

# ``orb_rot_opt.py`` intentionally remains a repository script rather than part of
# the installable package, so make the repository root importable for this check.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orb_rot_opt import (
    GivensProductSparseSimilarityCostEvaluator,
    OrbitalCommSqCostEvaluator,
    RotatedSymmetryCostEvaluator,
    SimpleSparseSimilarityCostEvaluator,
    choose_fastest_comm_sq_evaluator,
    sparse_hamiltonian_from_rotated_tensors,
)
from quasisymmetries.ferm_utils import build_sparse_basis, rotate_chem_obt, rotate_chem_tbt
from quasisymmetries.metrics import comm_sq_exp_fast
from quasisymmetries.op_utils import prepare_pauli_actions, prepare_pauli_sum_action
from quasisymmetries.orbital_rotation import (
    RealOrbitalRotation,
    apply_givens_product_to_sparse_operator,
    apply_givens_product_to_state,
    givens_product_num_params,
    givens_product_params_from_mat,
)


def old_cached_commutator_cost(sym_ops, obt, tbt, state, params, n_qubits, basis_dict):
    """Original sparse-basis commutator-action pattern, kept for verification."""
    sym_sparse = [sym for sym in sym_ops]
    nc_basis = {
        key: [op @ sym - sym @ op for sym in sym_sparse]
        for key, op in basis_dict.items()
    }

    rot = RealOrbitalRotation(n_qubits, params)
    U = rot.get_mat_rep()
    obt_rot = rotate_chem_obt(obt, U)
    tbt_rot = rotate_chem_tbt(tbt, U)
    psi_rot = rot.get_exp_rep() @ state

    states = [np.zeros_like(psi_rot, dtype=complex) for _ in sym_ops]
    for idx, mats in nc_basis.items():
        if len(idx) == 2:
            coeff = obt_rot[idx]
        else:
            coeff = tbt_rot[idx]
        if abs(coeff) < 1e-12:
            continue
        for n, mat in enumerate(mats):
            states[n] += coeff * (mat @ psi_rot)

    return sum(np.vdot(vec, vec) for vec in states).real


def main():
    rng = np.random.default_rng(11)
    n_qubits = 4
    dim = 1 << n_qubits

    obt = rng.normal(scale=0.2, size=(n_qubits, n_qubits))
    obt = 0.5 * (obt + obt.T)
    tbt = rng.normal(scale=0.03, size=(n_qubits,) * 4)

    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    state /= np.linalg.norm(state)
    params = rng.normal(scale=0.15, size=RealOrbitalRotation.num_params(n_qubits))

    sym_ops = [QubitOperator("Z0"), QubitOperator("Z1 Z2")]
    basis_dict = build_sparse_basis(n_qubits, include_obt=True)
    pauli_actions = prepare_pauli_actions(sym_ops, n_qubits)
    for sym, action in zip(sym_ops, pauli_actions):
        sparse_sym = get_sparse_operator(sym, n_qubits) @ state
        assert np.allclose(sparse_sym, action.apply(state))

    evaluator = OrbitalCommSqCostEvaluator(
        sym_ops, obt, tbt, state, n_qubits, basis_dict=basis_dict
    )

    t0 = time.perf_counter()
    fast_cost = evaluator.cost(params)
    fast_time = time.perf_counter() - t0

    U = RealOrbitalRotation(n_qubits, params).get_mat_rep()
    H_rot = sparse_hamiltonian_from_rotated_tensors(
        rotate_chem_obt(obt, U), rotate_chem_tbt(tbt, U), basis_dict
    )
    psi_rot_slow = RealOrbitalRotation(n_qubits, params).get_exp_rep() @ state

    t0 = time.perf_counter()
    direct_cost = comm_sq_exp_fast(sym_ops, H_rot, psi_rot_slow, n_qubits)
    direct_time = time.perf_counter() - t0

    sym_sparse = [s for s in evaluator.sym_ops_sparse]
    t0 = time.perf_counter()
    old_cost = old_cached_commutator_cost(
        sym_sparse, obt, tbt, state, params, n_qubits, basis_dict
    )
    old_time = time.perf_counter() - t0

    print(f"fast evaluator: {fast_cost:.12g} ({fast_time:.4f}s)")
    print(f"direct metric : {direct_cost:.12g} ({direct_time:.4f}s)")
    print(f"old pattern   : {old_cost:.12g} ({old_time:.4f}s)")

    assert np.allclose(fast_cost, direct_cost, rtol=1e-10, atol=1e-10)
    assert np.allclose(fast_cost, old_cost, rtol=1e-10, atol=1e-10)

    chosen, timings, values = choose_fastest_comm_sq_evaluator(
        sym_ops,
        obt,
        tbt,
        state,
        n_qubits,
        basis_dict,
        params,
        H_sparse=sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis_dict),
        benchmark_repeats=1,
    )
    chosen_cost = chosen.cost(params)
    print("adaptive chose:", min(timings, key=timings.get))
    assert np.allclose(chosen_cost, direct_cost, rtol=1e-10, atol=1e-10)

    chosen_grad, grad_timings, _ = choose_fastest_comm_sq_evaluator(
        sym_ops,
        obt,
        tbt,
        state,
        n_qubits,
        basis_dict,
        params,
        H_sparse=sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis_dict),
        benchmark_repeats=1,
        select_for_gradient=True,
    )
    assert hasattr(chosen_grad, "gradient")

    simple_eval = SimpleSparseSimilarityCostEvaluator(
        sym_ops,
        sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis_dict),
        state,
        n_qubits,
        basis_dict=basis_dict,
    )

    t0 = time.perf_counter()
    analytic_grad = simple_eval.gradient(params)
    analytic_time = time.perf_counter() - t0

    eps = 1e-6
    t0 = time.perf_counter()
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = eps
        fd_grad[i] = (
            simple_eval.cost(params + step) - simple_eval.cost(params - step)
        ) / (2 * eps)
    fd_time = time.perf_counter() - t0

    grad_err = np.linalg.norm(analytic_grad - fd_grad)
    rel_grad_err = grad_err / max(1.0, np.linalg.norm(fd_grad))
    print(
        f"analytic grad time={analytic_time:.4f}s "
        f"finite-diff grad time={fd_time:.4f}s rel_error={rel_grad_err:.3e}"
    )
    assert rel_grad_err < 1e-5

    symmetry_eval = RotatedSymmetryCostEvaluator(
        sym_ops,
        sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis_dict),
        state,
        n_qubits,
        basis_dict=basis_dict,
    )
    symmetry_cost = symmetry_eval.cost(params)
    assert np.allclose(symmetry_cost, direct_cost, rtol=1e-10, atol=1e-10)

    t0 = time.perf_counter()
    symmetry_grad = symmetry_eval.gradient(params)
    symmetry_analytic_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    symmetry_fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = eps
        symmetry_fd_grad[i] = (
            symmetry_eval.cost(params + step) - symmetry_eval.cost(params - step)
        ) / (2 * eps)
    symmetry_fd_time = time.perf_counter() - t0

    symmetry_grad_err = np.linalg.norm(symmetry_grad - symmetry_fd_grad)
    symmetry_rel_grad_err = symmetry_grad_err / max(1.0, np.linalg.norm(symmetry_fd_grad))
    print(
        f"symmetry grad time={symmetry_analytic_time:.4f}s "
        f"symmetry finite-diff time={symmetry_fd_time:.4f}s "
        f"rel_error={symmetry_rel_grad_err:.3e}"
    )
    assert symmetry_rel_grad_err < 1e-5

    H0 = sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis_dict)
    gp_params = givens_product_params_from_mat(
        RealOrbitalRotation(n_qubits, params).get_mat_rep()
    )
    gp_eval = GivensProductSparseSimilarityCostEvaluator(
        sym_ops, H0, state, n_qubits, basis_dict=basis_dict
    )
    t0 = time.perf_counter()
    gp_cost = gp_eval.cost(gp_params)
    gp_time = time.perf_counter() - t0
    H_gp = apply_givens_product_to_sparse_operator(gp_params, H0, n_qubits)
    psi_gp = apply_givens_product_to_state(gp_params, state, n_qubits)
    gp_ref = comm_sq_exp_fast(sym_ops, H_gp, psi_gp, n_qubits)
    print(f"givens-product cost={gp_cost:.12g} ({gp_time:.4f}s)")
    assert np.allclose(gp_cost, gp_ref, rtol=1e-10, atol=1e-10)
    assert np.allclose(gp_cost, chosen_cost, rtol=1e-8, atol=1e-8)

    t0 = time.perf_counter()
    gp_grad = gp_eval.gradient(gp_params)
    gp_grad_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    gp_fd_grad = np.zeros_like(gp_params)
    for i in range(len(gp_params)):
        step = np.zeros_like(gp_params)
        step[i] = eps
        gp_fd_grad[i] = (
            gp_eval.cost(gp_params + step) - gp_eval.cost(gp_params - step)
        ) / (2 * eps)
    gp_fd_time = time.perf_counter() - t0
    gp_rel_grad_err = np.linalg.norm(gp_grad - gp_fd_grad) / max(
        1.0, np.linalg.norm(gp_fd_grad)
    )
    print(
        f"givens-product grad time={gp_grad_time:.4f}s "
        f"finite-diff time={gp_fd_time:.4f}s rel_error={gp_rel_grad_err:.3e}"
    )
    assert gp_rel_grad_err < 1e-5

    H_qubit = (
        QubitOperator("X0 Y1", 0.2)
        + QubitOperator("Z0 Z2", -0.7)
        + QubitOperator("X1 X2 Z3", 0.4)
        + QubitOperator("", 0.1)
    )
    H_action = prepare_pauli_sum_action(H_qubit, n_qubits)
    H_sparse = get_sparse_operator(H_qubit, n_qubits)
    sparse_ci_state = np.zeros(dim, dtype=complex)
    sparse_ci_state[[0, 3, 7]] = [0.5, -0.2j, 0.8]
    t0 = time.perf_counter()
    H_sparse_state = H_sparse @ sparse_ci_state
    sparse_matvec_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    H_action_state = H_action.apply(sparse_ci_state, sparse_input=True)
    pauli_sum_time = time.perf_counter() - t0
    print(
        f"H sparse matvec={sparse_matvec_time:.6f}s "
        f"H pauli-sum action={pauli_sum_time:.6f}s"
    )
    assert np.allclose(H_sparse_state, H_action_state)

    print("OK")


if __name__ == "__main__":
    main()
