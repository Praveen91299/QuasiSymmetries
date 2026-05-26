"""
Verify orbital-rotation optimization backends.

Run from QuasiSymmetries:
    python verifications/verify_orb_rot_opt_backends.py
"""
import sys
from pathlib import Path

import numpy as np
from openfermion import QubitOperator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orb_rot_opt import (
    GivensProductMatrixFreeCostEvaluator,
    GivensProductSparseSimilarityCostEvaluator,
    GivensSparseSimilarityCostEvaluator,
    OrbitalCommSqCostEvaluator,
    RotatedSymmetryCostEvaluator,
    SimpleSparseSimilarityCostEvaluator,
    minimize_cisd_comm_sq,
    sparse_hamiltonian_from_rotated_tensors,
)
from src.ferm_utils import build_sparse_basis, rotate_chem_obt, rotate_chem_tbt
from src.metrics import comm_sq_exp_fast
from src.orbital_rotation import (
    RealOrbitalRotation,
    apply_givens_product_adjoint_to_state,
    apply_givens_product_to_sparse_operator,
    apply_givens_product_to_state,
    givens_product_params_from_mat,
)


def finite_difference_gradient(evaluator, params, eps=1e-6):
    grad = np.zeros_like(params)
    for idx in range(len(params)):
        step = np.zeros_like(params)
        step[idx] = eps
        grad[idx] = (evaluator.cost(params + step) - evaluator.cost(params - step)) / (
            2 * eps
        )
    return grad


def random_problem(n_qubits, seed):
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits

    obt = rng.normal(size=(n_qubits, n_qubits))
    obt = 0.5 * (obt + obt.T)
    tbt = rng.normal(scale=0.02, size=(n_qubits,) * 4)
    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    state /= np.linalg.norm(state)
    params = rng.normal(
        scale=0.15, size=RealOrbitalRotation.num_params(n_qubits)
    )
    sym_ops = [
        QubitOperator("Z0", 1.0),
        QubitOperator("Z1", -1.0),
    ]
    return obt, tbt, state, params, sym_ops


def verify_tensor_rotation_convention():
    for n_qubits in (3, 4):
        obt, tbt, _, params, _ = random_problem(n_qubits, seed=100 + n_qubits)
        basis = build_sparse_basis(n_qubits, include_obt=True)
        H0 = sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis)

        rot = RealOrbitalRotation(n_qubits, params)
        U = rot.get_mat_rep()
        u = rot.get_exp_rep().tocsr()
        H_tensor = sparse_hamiltonian_from_rotated_tensors(
            rotate_chem_obt(obt, U),
            rotate_chem_tbt(tbt, U),
            basis,
        )
        H_similarity = (u @ H0 @ u.T.conjugate()).tocsr()
        err = np.linalg.norm((H_tensor - H_similarity).toarray())
        print(f"tensor convention n={n_qubits}: error={err:.3e}")
        assert err < 1e-10


def verify_backend_costs():
    for n_qubits in (3, 4, 5):
        obt, tbt, state, params, sym_ops = random_problem(
            n_qubits, seed=200 + n_qubits
        )
        basis = build_sparse_basis(n_qubits, include_obt=True)
        H0 = sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis)

        rot = RealOrbitalRotation(n_qubits, params)
        u = rot.get_exp_rep().tocsr()
        H_ref = (u @ H0 @ u.T.conjugate()).tocsr()
        psi_ref = u @ state
        ref = comm_sq_exp_fast(sym_ops, H_ref, psi_ref, n_qubits)

        evaluators = {
            "simple": SimpleSparseSimilarityCostEvaluator(
                sym_ops, H0, state, n_qubits, basis_dict=basis
            ),
            "givens": GivensSparseSimilarityCostEvaluator(
                sym_ops, H0, state, n_qubits, basis_dict=basis
            ),
            "symmetry": RotatedSymmetryCostEvaluator(
                sym_ops, H0, state, n_qubits, basis_dict=basis
            ),
            "tensor": OrbitalCommSqCostEvaluator(
                sym_ops, obt, tbt, state, n_qubits, basis_dict=basis
            ),
        }

        print(f"backend costs n={n_qubits}: ref={ref}")
        for name, evaluator in evaluators.items():
            value = evaluator.cost(params)
            diff = abs(value - ref)
            print(f"  {name}: value={value} diff={diff:.3e}")
            assert np.allclose(value, ref, rtol=1e-8, atol=1e-8)

        gp_params = givens_product_params_from_mat(rot.get_mat_rep())
        H_gp = apply_givens_product_to_sparse_operator(gp_params, H0, n_qubits)
        psi_gp = apply_givens_product_to_state(gp_params, state, n_qubits)
        gp_ref = comm_sq_exp_fast(sym_ops, H_gp, psi_gp, n_qubits)
        gp_eval = GivensProductSparseSimilarityCostEvaluator(
            sym_ops, H0, state, n_qubits, basis_dict=basis
        )
        gp_value = gp_eval.cost(gp_params)
        print(f"  givens_product: value={gp_value} diff={abs(gp_value - gp_ref):.3e}")
        assert np.allclose(gp_value, gp_ref, rtol=1e-8, atol=1e-8)

        gp_mf_eval = GivensProductMatrixFreeCostEvaluator(
            sym_ops, H0, state, n_qubits, basis_dict=basis
        )
        gp_mf_value = gp_mf_eval.cost(gp_params)
        print(
            "  givens_product_matrix_free: "
            f"value={gp_mf_value} diff={abs(gp_mf_value - gp_ref):.3e}"
        )
        assert np.allclose(gp_mf_value, gp_ref, rtol=1e-8, atol=1e-8)


def verify_gradients():
    for n_qubits in (3, 4):
        obt, tbt, state, params, sym_ops = random_problem(
            n_qubits, seed=300 + n_qubits
        )
        basis = build_sparse_basis(n_qubits, include_obt=True)
        H0 = sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis)

        gradient_cases = [
            (
                "simple",
                SimpleSparseSimilarityCostEvaluator(
                    sym_ops, H0, state, n_qubits, basis_dict=basis
                ),
                params,
            ),
            (
                "symmetry",
                RotatedSymmetryCostEvaluator(
                    sym_ops, H0, state, n_qubits, basis_dict=basis
                ),
                params,
            ),
        ]
        gp_params = givens_product_params_from_mat(
            RealOrbitalRotation(n_qubits, params).get_mat_rep()
        )
        gradient_cases.append(
            (
                "givens_product",
                GivensProductSparseSimilarityCostEvaluator(
                    sym_ops, H0, state, n_qubits, basis_dict=basis
                ),
                gp_params,
            )
        )
        gradient_cases.append(
            (
                "givens_product_matrix_free",
                GivensProductMatrixFreeCostEvaluator(
                    sym_ops, H0, state, n_qubits, basis_dict=basis
                ),
                gp_params,
            )
        )

        for name, evaluator, test_params in gradient_cases:
            analytic = evaluator.gradient(test_params)
            finite_diff = finite_difference_gradient(evaluator, test_params)
            rel_err = np.linalg.norm(analytic - finite_diff) / max(
                1.0, np.linalg.norm(finite_diff)
            )
            print(f"gradient n={n_qubits} {name}: rel_err={rel_err:.3e}")
            assert rel_err < 1e-5


def verify_givens_product_adjoint():
    for n_qubits in (3, 4, 5):
        _, _, state, params, _ = random_problem(n_qubits, seed=400 + n_qubits)
        gp_params = givens_product_params_from_mat(
            RealOrbitalRotation(n_qubits, params).get_mat_rep()
        )
        rotated = apply_givens_product_to_state(gp_params, state, n_qubits)
        restored = apply_givens_product_adjoint_to_state(
            gp_params, rotated, n_qubits
        )
        err = np.linalg.norm(restored - state)
        print(f"givens_product adjoint n={n_qubits}: error={err:.3e}")
        assert err < 1e-10


def verify_minimization_smoke():
    n_qubits = 3
    obt, tbt, state, _, sym_ops = random_problem(n_qubits, seed=500)
    basis = build_sparse_basis(n_qubits, include_obt=True)
    H0 = sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis)

    result_params = minimize_cisd_comm_sq(
        sym_ops,
        obt,
        tbt,
        constant=0.0,
        ref_gs=state,
        fci_gs=state,
        n_qubits=n_qubits,
        n_trials=0,
        parallel=False,
        basis_dict=basis,
        H_sparse=H0,
        objective_mode="givens",
        parameterization="expm",
        optimizer_options={"maxiter": 0},
        random_seed=123,
    )
    assert len(result_params) == RealOrbitalRotation.num_params(n_qubits)
    print("minimization smoke: OK")


def main():
    verify_tensor_rotation_convention()
    verify_givens_product_adjoint()
    verify_backend_costs()
    verify_gradients()
    verify_minimization_smoke()
    print("All orbital-rotation backend verifications passed.")


if __name__ == "__main__":
    main()
