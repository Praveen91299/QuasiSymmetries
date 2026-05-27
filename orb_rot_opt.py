### orbital rotations for cisd cost minimization
### May 22
from scipy.optimize import minimize
from scipy.linalg import expm, expm_frechet
import pickle
import os
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from src.orbital_rotation import (
    SpinRestrictedRealOrbitalRotation,
    RealOrbitalRotation,
    apply_real_orbital_rotation_to_state,
    apply_real_orbital_rotation_to_sparse_operator,
    apply_givens_product_to_state,
    apply_givens_product_adjoint_to_state,
    apply_givens_product_to_sparse_operator,
    givens_product_mat,
    givens_product_num_params,
)
from src.metrics import (
    l1norm,
    get_ent,
    find_commuting_paulis,
    get_permuted_bipartite_entanglement,
    get_entropies_at_cuts,
    prepare_sparse_symmetries,
    comm_sq_exp_sparse_syms,
    comm_sq_exp_pauli_actions,
    comm_sq_exp_fast
)
from src.op_utils import (
    prepare_pauli_actions,
    prepare_pauli_sum_action,
)
from src.ferm_utils import Eij, rotate_chem_tbt, rotate_chem_obt, get_chem_tensors, spatial_obt_to_spin_obt, spatial_tbt_to_spin_tbt, build_sparse_basis
from openfermion import MolecularData, get_sparse_operator, jordan_wigner, count_qubits, FermionOperator, QubitOperator, commutator
import numpy as np
from scipy.sparse import csr_matrix, identity as sparse_identity
from src.sym import hct_mod, get_seniority_symmetries
import time

def complex_to_jsonable(value, tol=1e-12):
    value = complex(value)
    if abs(value.imag) < tol:
        return float(value.real)
    return {"real": float(value.real), "imag": float(value.imag)}

def array_to_jsonable(array, tol=1e-12):
    array = np.asarray(array)
    return [
        [complex_to_jsonable(value, tol=tol) for value in row]
        for row in array
    ]

def qubit_operator_to_jsonable(op, tol=1e-12):
    return [
        {
            "term": [[int(qubit), pauli] for qubit, pauli in term],
            "coefficient": complex_to_jsonable(coeff, tol=tol),
        }
        for term, coeff in op.terms.items()
    ]

def save_optimized_orbital_data(
    output_dir,
    system,
    tag,
    orbital_matrix,
    sym_ops,
    optimization_info=None,
):
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, "{}_{}_orbitals.npy".format(system, tag))
    txt_path = os.path.join(output_dir, "{}_{}_orbitals.txt".format(system, tag))

    np.save(npy_path, orbital_matrix)

    payload = {
        "system": system,
        "tag": tag,
        "matrix_file": os.path.basename(npy_path),
        "orbital_rotation_matrix": array_to_jsonable(orbital_matrix),
        "symmetries": [qubit_operator_to_jsonable(sym) for sym in sym_ops],
    }
    if optimization_info is not None:
        payload["parameterization"] = optimization_info.get("parameterization")
        payload["objective_backend"] = optimization_info.get("objective_backend")
        payload["initial_cost"] = complex_to_jsonable(optimization_info.get("initial_cost"))
        payload["final_cost"] = complex_to_jsonable(optimization_info.get("final_cost"))

    with open(txt_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    return npy_path, txt_path

def real_orbital_rotation_generator_derivatives(n_qubits):
    """
    Derivatives dK/dtheta_a for RealOrbitalRotation's skew generator matrix K.
    """
    derivs = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            dK = np.zeros((n_qubits, n_qubits), dtype=complex)
            dK[i, j] = 1.0
            dK[j, i] = -1.0
            derivs.append(dK)
    return derivs

def one_body_sparse_from_matrix(mat, basis_dict, tol=1e-12):
    """
    Build sum_pq mat[p,q] a_p^ a_q from cached one-body sparse basis.
    """
    n_qubits = mat.shape[0]
    dim = 1 << n_qubits
    op = csr_matrix((dim, dim), dtype=complex)
    for p in range(n_qubits):
        for q in range(n_qubits):
            coeff = mat[p, q]
            if abs(coeff) > tol:
                op += coeff * basis_dict[(p, q)]
    return op.tocsr()

def orbital_rotation_outputs(params, n_qubits, parameterization):
    """
    Return one-particle U and Fock-space state-rotation function consistently.
    """
    if parameterization == "expm":
        rot = RealOrbitalRotation(n_qubits, params)
        U = rot.get_mat_rep()

        def rotate_state(state):
            return rot.get_exp_rep() @ state

        return U, rotate_state

    if parameterization == "givens_product":
        U = givens_product_mat(params, n_qubits)

        def rotate_state(state):
            return apply_givens_product_to_state(params, state, n_qubits)

        return U, rotate_state

    raise ValueError("Unknown parameterization {}".format(parameterization))

def build_nc_sparse_basis(basis_dict: dict, sym_ops_sparse):
    """
    collects commutators of basis_dict elements with symmetries

    """
    nc_basis_dict = {}
    for k, v in basis_dict.items():
        nc_basis_dict[k] = [v @ sym - sym @ v for sym in sym_ops_sparse]
        
    return nc_basis_dict

def chem_obt_tbt_to_chem_ferm(obt, tbt, constant):

    op = FermionOperator()
    op += constant
    a, b, c, d = np.shape(tbt)

    for i in range(a):
        for j in range(b):
            op += FermionOperator('{}^ {}'.format(i, j), obt[i, j])

            for k in range(c):
                for l in range(d):
                    op += FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l), tbt[i, j, k, l])

    return op

def sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis_dict, tol=1e-12):
    """
    Build sparse H from one- and two-body chemist tensors using a cached basis.

    This is usually faster than accumulating [basis, S]|psi> term-by-term for
    each symmetry because the expensive matvecs are done only once per symmetry
    in comm_sq_exp_sparse_syms.
    """
    n_qubits = obt.shape[0]
    dim = 1 << n_qubits
    H = csr_matrix((dim, dim), dtype=complex)

    for i in range(n_qubits):
        for j in range(n_qubits):
            coeff = obt[i, j]
            if abs(coeff) > tol:
                H += coeff * basis_dict[(i, j)]

    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(n_qubits):
                    coeff = tbt[i, j, k, l]
                    if abs(coeff) > tol:
                        H += coeff * basis_dict[(i, j, k, l)]

    return H.tocsr()

class OrbitalCommSqCostEvaluator:
    """
    Fast repeated objective evaluator for orbital-rotation commutator cost.

    Per call it:
      1. forms the small N x N orbital rotation,
      2. rotates the integral tensors,
      3. applies the Fock-space orbital rotation to the state with Givens
         amplitude updates instead of building exp(G) in 2^N space,
      4. evaluates sum_k || i[H_rot, S_k] |psi_rot> ||^2 using precomputed
         sparse symmetry matrices.
    """
    def __init__(
        self,
        sym_ops,
        obt,
        tbt,
        ref_state,
        n_qubits,
        basis_dict=None,
        coeff_tol=1e-12,
        givens_tol=1e-10,
        weights=None,
    ):
        self.sym_ops_sparse = prepare_sparse_symmetries(sym_ops, n_qubits)
        self.obt = np.asarray(obt)
        self.tbt = np.asarray(tbt)
        self.ref_state = np.asarray(ref_state).reshape(-1)
        self.n_qubits = n_qubits
        self.coeff_tol = coeff_tol
        self.givens_tol = givens_tol
        self.weights = weights

        if basis_dict is None:
            print("Preparing sparse basis...")
            basis_dict = build_sparse_basis(n_qubits, include_obt=True)
        self.basis_dict = basis_dict

    def rotated_sparse_hamiltonian(self, params):
        U = RealOrbitalRotation(self.n_qubits, params).get_mat_rep()
        obt_rot = rotate_chem_obt(self.obt, U)
        tbt_rot = rotate_chem_tbt(self.tbt, U)
        return sparse_hamiltonian_from_rotated_tensors(
            obt_rot, tbt_rot, self.basis_dict, tol=self.coeff_tol
        )

    def rotated_state(self, params):
        return apply_real_orbital_rotation_to_state(
            params, self.ref_state, self.n_qubits, tol=self.givens_tol
        )

    def cost(self, params):
        H_rot = self.rotated_sparse_hamiltonian(params)
        psi_rot = self.rotated_state(params)
        return comm_sq_exp_sparse_syms(
            self.sym_ops_sparse, H_rot, psi_rot, weights=self.weights
        )

class SimpleSparseSimilarityCostEvaluator:
    """
    Baseline objective:
        u = RO.get_exp_rep()
        cost = sum_k || i[u H u^dagger, S_k] u|psi> ||^2

    This keeps the efficient simple sparse path, while caching sparse S_k once.
    """
    def __init__(
        self,
        sym_ops,
        H_sparse,
        ref_state,
        n_qubits,
        sym_ops_sparse=None,
        pauli_actions=None,
        weights=None,
        basis_dict=None,
    ):
        self.sym_ops_sparse = (
            prepare_sparse_symmetries(sym_ops, n_qubits)
            if sym_ops_sparse is None
            else sym_ops_sparse
        )
        self.pauli_actions = (
            prepare_pauli_actions(sym_ops, n_qubits)
            if pauli_actions is None
            else pauli_actions
        )
        self.H_sparse = H_sparse.tocsr()
        self.ref_state = np.asarray(ref_state).reshape(-1)
        self.n_qubits = n_qubits
        self.weights = weights
        self.basis_dict = basis_dict
        self.ref_Hpsi = self.H_sparse @ self.ref_state
        self.generator_derivs = real_orbital_rotation_generator_derivatives(n_qubits)

    def cost(self, params):
        u = RealOrbitalRotation(self.n_qubits, params).get_exp_rep().tocsr()
        H_rot = (u @ self.H_sparse @ u.T.conjugate()).tocsr()
        psi_rot = u @ self.ref_state
        Hpsi_rot = u @ self.ref_Hpsi
        return comm_sq_exp_pauli_actions(
            self.pauli_actions, H_rot, psi_rot, weights=self.weights, Hpsi=Hpsi_rot
        )

    def _rotated_quantities(self, params):
        K = RealOrbitalRotation.build_param_mat(params, self.n_qubits)
        U = expm(K)
        u = RealOrbitalRotation(self.n_qubits, params).get_exp_rep().tocsr()
        H_rot = (u @ self.H_sparse @ u.T.conjugate()).tocsr()
        psi_rot = u @ self.ref_state
        Hpsi_rot = u @ self.ref_Hpsi
        return K, U, H_rot, psi_rot, Hpsi_rot

    def gradient(self, params):
        if self.basis_dict is None:
            raise ValueError("Analytic gradient requires a one-body sparse basis_dict.")

        K, U, H_rot, psi, Hpsi = self._rotated_quantities(params)

        weights = (
            np.ones(len(self.sym_ops_sparse))
            if self.weights is None
            else self.weights
        )

        sym_data = []
        for weight, S, action in zip(weights, self.sym_ops_sparse, self.pauli_actions):
            if weight == 0:
                continue
            Spsi = action.apply(psi)
            delta = 1j * ((H_rot @ Spsi) - action.apply(Hpsi))
            sym_data.append((weight, action, Spsi, delta))

        grad = np.zeros(len(params), dtype=float)
        U_dag = U.T.conjugate()

        for a, dK in enumerate(self.generator_derivs):
            dU = expm_frechet(K, dK, compute_expm=False)
            B = dU @ U_dag
            A = one_body_sparse_from_matrix(B, self.basis_dict)

            dpsi = A @ psi
            Hdpsi = H_rot @ dpsi
            dHpsi = A @ Hpsi - H_rot @ dpsi

            partial = 0.0
            for weight, action, Spsi, delta in sym_data:
                dHSpsi = A @ (H_rot @ Spsi) - H_rot @ (A @ Spsi)
                dCpsi = 1j * (dHSpsi - action.apply(dHpsi))
                Cdpsi = 1j * ((H_rot @ action.apply(dpsi)) - action.apply(Hdpsi))
                partial += weight * 2.0 * np.real(np.vdot(delta, dCpsi + Cdpsi))

            grad[a] = partial

        return grad

class RotatedSymmetryCostEvaluator:
    """
    Objective using unitary invariance:

        || i[V H V^dagger, S] V|psi> ||
      = || i[H, V^dagger S V] |psi> ||

    This avoids rotating H and the state.  It is attractive when the number of
    symmetries is small compared with the Hamiltonian sparsity/size.
    """
    def __init__(
        self,
        sym_ops,
        H_sparse,
        ref_state,
        n_qubits,
        sym_ops_sparse=None,
        weights=None,
        basis_dict=None,
        H_pauli_action=None,
        sparse_ref_tol=1e-12,
    ):
        self.sym_ops_sparse = (
            prepare_sparse_symmetries(sym_ops, n_qubits)
            if sym_ops_sparse is None
            else sym_ops_sparse
        )
        self.H_sparse = H_sparse.tocsr()
        self.ref_state = np.asarray(ref_state).reshape(-1)
        self.n_qubits = n_qubits
        self.weights = weights
        self.basis_dict = basis_dict
        self.generator_derivs = real_orbital_rotation_generator_derivatives(n_qubits)
        self.H_pauli_action = H_pauli_action
        if H_pauli_action is None:
            self.Hpsi = self.H_sparse @ self.ref_state
        else:
            self.Hpsi = H_pauli_action.apply(
                self.ref_state, sparse_input=True, tol=sparse_ref_tol
            )

    def rotated_symmetries(self, params):
        u = RealOrbitalRotation(self.n_qubits, params).get_exp_rep().tocsr()
        u_dag = u.T.conjugate()
        return [(u_dag @ S @ u).tocsr() for S in self.sym_ops_sparse]

    def cost(self, params):
        S_rot_list = self.rotated_symmetries(params)
        return comm_sq_exp_sparse_syms(
            S_rot_list, self.H_sparse, self.ref_state, weights=self.weights
        )

    def gradient(self, params):
        if self.basis_dict is None:
            raise ValueError("Analytic gradient requires a one-body sparse basis_dict.")

        K = RealOrbitalRotation.build_param_mat(params, self.n_qubits)
        U = expm(K)
        U_dag = U.T.conjugate()
        S_rot_list = self.rotated_symmetries(params)

        weights = (
            np.ones(len(S_rot_list))
            if self.weights is None
            else self.weights
        )

        sym_data = []
        for weight, S_rot in zip(weights, S_rot_list):
            if weight == 0:
                continue
            Spsi = S_rot @ self.ref_state
            delta = 1j * ((self.H_sparse @ Spsi) - (S_rot @ self.Hpsi))
            sym_data.append((weight, S_rot, delta))

        grad = np.zeros(len(params), dtype=float)
        for a, dK in enumerate(self.generator_derivs):
            dU = expm_frechet(K, dK, compute_expm=False)
            B = U_dag @ dU
            Bop = one_body_sparse_from_matrix(B, self.basis_dict)

            partial = 0.0
            for weight, S_rot, delta in sym_data:
                dS = S_rot @ Bop - Bop @ S_rot
                dSpsi = dS @ self.ref_state
                ddelta = 1j * ((self.H_sparse @ dSpsi) - (dS @ self.Hpsi))
                partial += weight * 2.0 * np.real(np.vdot(delta, ddelta))

            grad[a] = partial

        return grad

class GivensSparseSimilarityCostEvaluator(SimpleSparseSimilarityCostEvaluator):
    """
    Sparse-similarity objective using a sequence of sparse Givens updates.

    This is a deeper specialization of the Hamiltonian rotation: instead of
    constructing the full Fock-space orbital rotation exp(G), it decomposes the
    N x N orbital rotation into two-orbital rotations and applies H -> G H G^dagger
    step by step.  It can help when get_exp_rep() is expensive, but it can also
    lose because it performs many sparse matrix multiplications.
    """
    def __init__(
        self,
        sym_ops,
        H_sparse,
        ref_state,
        n_qubits,
        sym_ops_sparse=None,
        pauli_actions=None,
        weights=None,
        basis_dict=None,
        givens_tol=1e-10,
    ):
        super().__init__(
            sym_ops,
            H_sparse,
            ref_state,
            n_qubits,
            sym_ops_sparse=sym_ops_sparse,
            pauli_actions=pauli_actions,
            weights=weights,
            basis_dict=basis_dict,
        )
        self.givens_tol = givens_tol

    def cost(self, params):
        H_rot = apply_real_orbital_rotation_to_sparse_operator(
            params, self.H_sparse, self.n_qubits, tol=self.givens_tol
        )
        psi_rot = apply_real_orbital_rotation_to_state(
            params, self.ref_state, self.n_qubits, tol=self.givens_tol
        )
        Hpsi_rot = apply_real_orbital_rotation_to_state(
            params, self.ref_Hpsi, self.n_qubits, tol=self.givens_tol
        )
        return comm_sq_exp_pauli_actions(
            self.pauli_actions, H_rot, psi_rot, weights=self.weights, Hpsi=Hpsi_rot
        )

class GivensProductSparseSimilarityCostEvaluator(SimpleSparseSimilarityCostEvaluator):
    """
    Direct product-of-Givens orbital-rotation parameterization.

    Parameters are Givens angles in depth_eff_order_mf order.  This avoids the
    small-matrix expm, Fock-space expm, and SO decomposition used by the exp(K)
    parameterization.
    """
    def cost(self, params):
        H_rot = apply_givens_product_to_sparse_operator(
            params, self.H_sparse, self.n_qubits
        )
        psi_rot = apply_givens_product_to_state(params, self.ref_state, self.n_qubits)
        Hpsi_rot = apply_givens_product_to_state(params, self.ref_Hpsi, self.n_qubits)
        return comm_sq_exp_pauli_actions(
            self.pauli_actions, H_rot, psi_rot, weights=self.weights, Hpsi=Hpsi_rot
        )

    def _rotated_quantities(self, params):
        H_rot = apply_givens_product_to_sparse_operator(
            params, self.H_sparse, self.n_qubits
        )
        psi_rot = apply_givens_product_to_state(params, self.ref_state, self.n_qubits)
        Hpsi_rot = apply_givens_product_to_state(params, self.ref_Hpsi, self.n_qubits)
        return H_rot, psi_rot, Hpsi_rot

    def _tangent_generators(self, params):
        from src.orbital_rotation import givens_product_pairs, sparse_real_givens_unitary

        if self.basis_dict is None:
            raise ValueError("Analytic gradient requires a one-body sparse basis_dict.")

        pairs = givens_product_pairs(self.n_qubits)
        givens_ops = [
            sparse_real_givens_unitary(self.n_qubits, i, j, theta)
            for theta, (i, j) in zip(params, pairs)
        ]

        dim = 1 << self.n_qubits
        suffix = []
        current = sparse_identity(dim, dtype=complex, format="csr")
        for G in reversed(givens_ops):
            suffix.append(current)
            current = current @ G
        suffix.reverse()

        generators = []
        for (i, j), L in zip(pairs, suffix):
            Kij = self.basis_dict[(i, j)] - self.basis_dict[(j, i)]
            generators.append((L @ Kij @ L.T.conjugate()).tocsr())
        return generators

    def gradient(self, params):
        H_rot, psi, Hpsi = self._rotated_quantities(params)

        weights = (
            np.ones(len(self.pauli_actions))
            if self.weights is None
            else self.weights
        )

        sym_data = []
        for weight, action in zip(weights, self.pauli_actions):
            if weight == 0:
                continue
            Spsi = action.apply(psi)
            delta = 1j * ((H_rot @ Spsi) - action.apply(Hpsi))
            sym_data.append((weight, action, Spsi, delta))

        grad = np.zeros(len(params), dtype=float)
        for a, A in enumerate(self._tangent_generators(params)):
            dpsi = A @ psi
            Hdpsi = H_rot @ dpsi
            dHpsi = A @ Hpsi - H_rot @ dpsi

            partial = 0.0
            for weight, action, Spsi, delta in sym_data:
                dHSpsi = A @ (H_rot @ Spsi) - H_rot @ (A @ Spsi)
                dCpsi = 1j * (dHSpsi - action.apply(dHpsi))
                Cdpsi = 1j * ((H_rot @ action.apply(dpsi)) - action.apply(Hdpsi))
                partial += weight * 2.0 * np.real(np.vdot(delta, dCpsi + Cdpsi))

            grad[a] = partial
        return grad

class GivensProductMatrixFreeCostEvaluator(GivensProductSparseSimilarityCostEvaluator):
    """
    Direct Givens-product objective without forming H_rot.

    For each vector v this applies H_rot v = U H U^dagger v by unrotating v,
    applying the cached/base sparse Hamiltonian, then rotating back.  This avoids
    sparse matrix-matrix products in H -> U H U^dagger.
    """
    def _rotate(self, params, state):
        return apply_givens_product_to_state(params, state, self.n_qubits)

    def _unrotate(self, params, state):
        return apply_givens_product_adjoint_to_state(params, state, self.n_qubits)

    def _hrot_apply(self, params, state):
        return self._rotate(params, self.H_sparse @ self._unrotate(params, state))

    def cost(self, params):
        psi = self._rotate(params, self.ref_state)
        Hpsi = self._rotate(params, self.ref_Hpsi)

        weights = (
            np.ones(len(self.pauli_actions))
            if self.weights is None
            else self.weights
        )

        total = 0.0
        for weight, action in zip(weights, self.pauli_actions):
            if weight == 0:
                continue
            Spsi = action.apply(psi)
            delta = 1j * (self._hrot_apply(params, Spsi) - action.apply(Hpsi))
            total += weight * np.vdot(delta, delta).real
        return np.real_if_close(total)

    def _rotated_quantities(self, params):
        psi = self._rotate(params, self.ref_state)
        Hpsi = self._rotate(params, self.ref_Hpsi)
        return psi, Hpsi

    def gradient(self, params):
        psi, Hpsi = self._rotated_quantities(params)

        weights = (
            np.ones(len(self.pauli_actions))
            if self.weights is None
            else self.weights
        )

        sym_data = []
        for weight, action in zip(weights, self.pauli_actions):
            if weight == 0:
                continue
            Spsi = action.apply(psi)
            Hrot_Spsi = self._hrot_apply(params, Spsi)
            delta = 1j * (Hrot_Spsi - action.apply(Hpsi))
            sym_data.append((weight, action, Spsi, Hrot_Spsi, delta))

        grad = np.zeros(len(params), dtype=float)
        for a, A in enumerate(self._tangent_generators(params)):
            dpsi = A @ psi
            dHpsi = A @ Hpsi

            partial = 0.0
            for weight, action, Spsi, Hrot_Spsi, delta in sym_data:
                dSpsi = action.apply(dpsi)
                dHrot_Spsi = A @ Hrot_Spsi + self._hrot_apply(
                    params, dSpsi - A @ Spsi
                )
                ddelta = 1j * (dHrot_Spsi - action.apply(dHpsi))
                partial += weight * 2.0 * np.real(np.vdot(delta, ddelta))

            grad[a] = partial
        return grad

def time_cost_evaluator(evaluator, params, repeats=1):
    start = time.perf_counter()
    value = None
    for _ in range(repeats):
        value = evaluator.cost(params)
    return (time.perf_counter() - start) / repeats, value

def objective_method_parameterization(method):
    if method in ("givens_product", "givens_product_matrix_free"):
        return "givens_product"
    return "expm"

def choose_fastest_comm_sq_evaluator(
    sym_ops,
    obt,
    tbt,
    ref_state,
    n_qubits,
    basis_dict,
    x_probe,
    H_sparse=None,
    H_qubit=None,
    include_methods=("simple", "givens", "symmetry", "tensor"),
    benchmark_repeats=1,
    weights=None,
    select_for_gradient=False,
    verbose=True,
):
    """
    Build candidate objective evaluators, time them at x_probe, and return fastest.
    """
    sym_ops_sparse = prepare_sparse_symmetries(sym_ops, n_qubits)
    pauli_actions = prepare_pauli_actions(sym_ops, n_qubits)
    H_pauli_action = (
        prepare_pauli_sum_action(H_qubit, n_qubits)
        if H_qubit is not None
        else None
    )
    if H_sparse is None:
        H_sparse = sparse_hamiltonian_from_rotated_tensors(obt, tbt, basis_dict)

    candidates = {}
    if "simple" in include_methods:
        candidates["simple"] = SimpleSparseSimilarityCostEvaluator(
            sym_ops, H_sparse, ref_state, n_qubits,
            sym_ops_sparse=sym_ops_sparse, pauli_actions=pauli_actions,
            weights=weights, basis_dict=basis_dict
        )
    if "givens" in include_methods:
        candidates["givens"] = GivensSparseSimilarityCostEvaluator(
            sym_ops, H_sparse, ref_state, n_qubits,
            sym_ops_sparse=sym_ops_sparse, pauli_actions=pauli_actions, weights=weights,
            basis_dict=basis_dict,
        )
    if "givens_product" in include_methods:
        candidates["givens_product"] = GivensProductSparseSimilarityCostEvaluator(
            sym_ops, H_sparse, ref_state, n_qubits,
            sym_ops_sparse=sym_ops_sparse, pauli_actions=pauli_actions,
            weights=weights, basis_dict=basis_dict,
        )
    if "givens_product_matrix_free" in include_methods:
        candidates["givens_product_matrix_free"] = GivensProductMatrixFreeCostEvaluator(
            sym_ops, H_sparse, ref_state, n_qubits,
            sym_ops_sparse=sym_ops_sparse, pauli_actions=pauli_actions,
            weights=weights, basis_dict=basis_dict,
        )
    if "symmetry" in include_methods:
        candidates["symmetry"] = RotatedSymmetryCostEvaluator(
            sym_ops, H_sparse, ref_state, n_qubits,
            sym_ops_sparse=sym_ops_sparse, weights=weights,
            basis_dict=basis_dict, H_pauli_action=H_pauli_action,
        )
    if "tensor" in include_methods:
        candidates["tensor"] = OrbitalCommSqCostEvaluator(
            sym_ops, obt, tbt, ref_state, n_qubits,
            basis_dict=basis_dict, weights=weights
        )
        candidates["tensor"].sym_ops_sparse = sym_ops_sparse

    timings = {}
    grad_timings = {}
    selection_scores = {}
    values = {}
    if not candidates:
        raise ValueError("No compatible objective backends were requested.")
    for name, evaluator in candidates.items():
        elapsed, value = time_cost_evaluator(evaluator, x_probe, benchmark_repeats)
        timings[name] = elapsed
        values[name] = value
        gradient = getattr(evaluator, "gradient", None)
        if select_for_gradient and callable(gradient):
            start = time.perf_counter()
            gradient(x_probe)
            grad_timings[name] = time.perf_counter() - start
            selection_scores[name] = elapsed + grad_timings[name]
        elif select_for_gradient:
            selection_scores[name] = elapsed * (len(x_probe) + 1)
        else:
            selection_scores[name] = elapsed

    names = list(values)
    ref = values[names[0]]
    for name in names[1:]:
        if not np.allclose(ref, values[name], rtol=1e-8, atol=1e-8):
            print(
                "Warning: objective backend {} differs from {}: {} vs {}".format(
                    name, names[0], values[name], ref
                )
            )

    winner = min(selection_scores, key=selection_scores.get)
    candidates[winner].backend_name = winner
    if verbose:
        print("Objective backend timings at initial point:")
        for name, elapsed in sorted(timings.items(), key=lambda item: item[1]):
            if select_for_gradient:
                grad_note = (
                    " analytic_grad={:.6f}s".format(grad_timings[name])
                    if name in grad_timings
                    else " finite_diff_est={:.6f}s".format(selection_scores[name])
                )
                print(
                    "  {}: cost={:.6f}s{} selection_score={:.6f}s value={}".format(
                        name, elapsed, grad_note, selection_scores[name], values[name]
                    )
                )
            else:
                print("  {}: {:.6f}s value={}".format(name, elapsed, values[name]))
        print("Using objective backend: {}".format(winner))

    return candidates[winner], timings, values

def build_comm_sq_evaluator(
    sym_ops,
    obt,
    tbt,
    ref_state,
    n_qubits,
    basis_dict,
    x_probe,
    H_sparse=None,
    H_qubit=None,
    include_methods=("simple", "givens", "symmetry", "tensor"),
    benchmark_repeats=1,
    weights=None,
    use_analytic_gradient=True,
    verbose=True,
):
    evaluator, timings, values = choose_fastest_comm_sq_evaluator(
        sym_ops,
        obt,
        tbt,
        ref_state,
        n_qubits,
        basis_dict,
        x_probe,
        H_sparse=H_sparse,
        H_qubit=H_qubit,
        include_methods=include_methods,
        benchmark_repeats=benchmark_repeats,
        weights=weights,
        select_for_gradient=use_analytic_gradient,
        verbose=verbose,
    )
    gradient = None
    evaluator_gradient = getattr(evaluator, "gradient", None)
    if use_analytic_gradient and callable(evaluator_gradient):
        gradient = evaluator_gradient
    return evaluator, gradient, timings, values

def make_fast_cisd_comm_sq_cost(
    sym_ops,
    obt,
    tbt,
    ref_gs,
    n_qubits,
    basis_dict=None,
    **kwargs,
):
    evaluator = OrbitalCommSqCostEvaluator(
        sym_ops, obt, tbt, ref_gs, n_qubits, basis_dict=basis_dict, **kwargs
    )
    return evaluator.cost, evaluator

def _minimize_with_fd_control(
    cost,
    x0,
    callback,
    gradient=None,
    optimizer_method="L-BFGS-B",
    finite_diff_eps=1e-4,
    finite_diff_jac=None,
    optimizer_options=None,
):
    options = {} if optimizer_options is None else dict(optimizer_options)
    jac = gradient if gradient is not None else finite_diff_jac

    if gradient is not None:
        pass
    elif finite_diff_jac in ("2-point", "3-point", "cs"):
        options.setdefault("finite_diff_rel_step", finite_diff_eps)
    else:
        options.setdefault("eps", finite_diff_eps)

    return minimize(
        cost,
        x0,
        args=(),
        method=optimizer_method,
        jac=jac,
        callback=callback,
        options=options,
    )

def _run_minimize_trial_worker(payload):
    evaluator, gradient, _, _ = build_comm_sq_evaluator(
        payload["sym_ops"],
        payload["obt"],
        payload["tbt"],
        payload["ref_gs"],
        payload["n_qubits"],
        payload["basis_dict"],
        payload["x0"],
        H_sparse=payload["H_sparse"],
        H_qubit=payload["H_qubit"],
        include_methods=payload["include_methods"],
        benchmark_repeats=1,
        use_analytic_gradient=payload["use_analytic_gradient"],
        verbose=False,
    )

    return _minimize_with_fd_control(
        evaluator.cost,
        payload["x0"],
        callback=None,
        gradient=gradient,
        optimizer_method=payload["optimizer_method"],
        finite_diff_eps=payload["finite_diff_eps"],
        finite_diff_jac=payload["finite_diff_jac"],
        optimizer_options=payload["optimizer_options"],
    )

def num_orbital_rotation_params(n_qubits, parameterization):
    if parameterization == "expm":
        return RealOrbitalRotation.num_params(n_qubits)
    if parameterization == "givens_product":
        return givens_product_num_params(n_qubits)
    raise ValueError("Unknown parameterization {}".format(parameterization))

def time_optimizer_step(evaluator, params, use_analytic_gradient=True):
    cost_time, value = time_cost_evaluator(evaluator, params, repeats=1)
    gradient = getattr(evaluator, "gradient", None)
    if use_analytic_gradient and callable(gradient):
        start = time.perf_counter()
        gradient(params)
        grad_time = time.perf_counter() - start
        return cost_time + grad_time, cost_time, grad_time, value, True

    # Finite-difference estimate for one gradient evaluation.
    return cost_time * (len(params) + 1), cost_time, None, value, False

def choose_fastest_parameterization_by_runtime(
    sym_ops,
    obt,
    tbt,
    ref_state,
    n_qubits,
    basis_dict,
    H_sparse=None,
    H_qubit=None,
    objective_mode="auto",
    benchmark_repeats=1,
    weights=None,
    use_analytic_gradient=True,
):
    """
    Compare expm and direct Givens-product parameterizations by runtime.

    The comparison is done at the identity rotation in each coordinate system,
    so it compares optimizer-step cost rather than objective values at unrelated
    random coordinates.
    """
    candidates = []

    expm_methods = (
        ("simple", "givens", "symmetry", "tensor")
        if objective_mode == "auto"
        else tuple(
            m for m in (objective_mode,)
            if objective_method_parameterization(m) == "expm"
        )
    )
    if expm_methods:
        x_expm = np.zeros(num_orbital_rotation_params(n_qubits, "expm"))
        expm_evaluator, _, _ = choose_fastest_comm_sq_evaluator(
            sym_ops,
            obt,
            tbt,
            ref_state,
            n_qubits,
            basis_dict,
            x_expm,
            H_sparse=H_sparse,
            H_qubit=H_qubit,
            include_methods=expm_methods,
            benchmark_repeats=benchmark_repeats,
            weights=weights,
            select_for_gradient=use_analytic_gradient,
            verbose=True,
        )
        total, cost_t, grad_t, value, has_grad = time_optimizer_step(
            expm_evaluator, x_expm, use_analytic_gradient=use_analytic_gradient
        )
        candidates.append(("expm", expm_evaluator, total, cost_t, grad_t, value, has_grad))

    givens_methods = (
        ("givens_product", "givens_product_matrix_free")
        if objective_mode == "auto"
        else tuple(
            m for m in (objective_mode,)
            if objective_method_parameterization(m) == "givens_product"
        )
    )
    if givens_methods:
        x_givens = np.zeros(num_orbital_rotation_params(n_qubits, "givens_product"))
        givens_evaluator, _, _ = choose_fastest_comm_sq_evaluator(
            sym_ops,
            obt,
            tbt,
            ref_state,
            n_qubits,
            basis_dict,
            x_givens,
            H_sparse=H_sparse,
            H_qubit=H_qubit,
            include_methods=givens_methods,
            benchmark_repeats=benchmark_repeats,
            weights=weights,
            select_for_gradient=use_analytic_gradient,
            verbose=True,
        )
        total, cost_t, grad_t, value, has_grad = time_optimizer_step(
            givens_evaluator, x_givens, use_analytic_gradient=use_analytic_gradient
        )
        candidates.append(("givens_product", givens_evaluator, total, cost_t, grad_t, value, has_grad))

    if not candidates:
        raise ValueError("No compatible parameterizations were requested.")

    print("Parameterization runtime comparison at identity:")
    for name, _, total, cost_t, grad_t, value, has_grad in sorted(candidates, key=lambda item: item[2]):
        grad_label = (
            "grad={:.6f}s".format(grad_t)
            if has_grad
            else "finite_diff_est"
        )
        print(
            "  {}: step={:.6f}s cost={:.6f}s {} value={}".format(
                name, total, cost_t, grad_label, value
            )
        )

    winner = min(candidates, key=lambda item: item[2])
    print("Using parameterization: {}".format(winner[0]))
    return winner[0]

def minimize_cisd_comm_sq(
    sym_ops,
    obt,
    tbt,
    constant,
    ref_gs,
    fci_gs,
    n_qubits,
    n_trials=1,
    parallel=False,
    n_jobs=None,
    parallel_backend="process",
    random_seed=None,
    verify=True,
    basis_dict=None,
    H_sparse=None,
    H_qubit=None,
    objective_mode="auto",
    parameterization="expm",
    benchmark_repeats=1,
    optimizer_method="L-BFGS-B",
    finite_diff_eps=1e-4,
    finite_diff_jac=None,
    use_analytic_gradient=True,
    optimizer_options=None,
    return_info=False,
):
    """
    Minimize cisd comm squared expectation value at ref_gs
    

    """
    print("Symmetries: ", sym_ops)
    if basis_dict is None:
        print("Preparing sparse basis...")
        basis_dict = build_sparse_basis(n_qubits, include_obt=True)
    else:
        print("Loading provided sparse basis.")

    if parameterization == "auto_runtime":
        parameterization = choose_fastest_parameterization_by_runtime(
            sym_ops,
            obt,
            tbt,
            ref_gs,
            n_qubits,
            basis_dict,
            H_sparse=H_sparse,
            H_qubit=H_qubit,
            objective_mode=objective_mode,
            benchmark_repeats=benchmark_repeats,
            use_analytic_gradient=use_analytic_gradient,
        )

    num_params = num_orbital_rotation_params(n_qubits, parameterization)

    rng = np.random.default_rng(random_seed)
    x0 = np.zeros(num_params)
    include_methods = (
        (
            "simple",
            "givens",
            "givens_product",
            "givens_product_matrix_free",
            "symmetry",
            "tensor",
        )
        if objective_mode == "auto"
        else (objective_mode,)
    )
    include_methods = tuple(
        method for method in include_methods
        if objective_method_parameterization(method) == parameterization
    )
    evaluator, gradient, timings, values = build_comm_sq_evaluator(
        sym_ops,
        obt,
        tbt,
        ref_gs,
        n_qubits,
        basis_dict,
        x0,
        H_sparse=H_sparse,
        H_qubit=H_qubit,
        include_methods=include_methods,
        benchmark_repeats=benchmark_repeats,
        use_analytic_gradient=use_analytic_gradient,
        verbose=True,
    )
    cost = evaluator.cost
    if use_analytic_gradient:
        if gradient is None:
            print("Selected objective has no analytic gradient; using finite differences.")
        else:
            print("Using analytic gradient for selected objective.")

    print("Identity init objective value:", cost(x0))

    def call_back(params):
        print("Current objective value:", cost(params))

    x0_list = [x0] + [rng.random(num_params) for _ in range(n_trials)]

    if parallel and len(x0_list) > 1:
        backend_name = getattr(evaluator, "backend_name", None)
        if backend_name is None:
            raise RuntimeError("Could not identify selected objective backend for parallel trials.")

        if n_jobs is None:
            n_jobs = min(len(x0_list), os.cpu_count() or 1)
        else:
            n_jobs = min(n_jobs, len(x0_list))

        print("Running {} optimization starts in parallel with {} workers.".format(
            len(x0_list), n_jobs
        ))
        print("Parallel workers use backend: {}".format(backend_name))

        payload_base = {
            "sym_ops": sym_ops,
            "obt": obt,
            "tbt": tbt,
            "ref_gs": ref_gs,
            "n_qubits": n_qubits,
            "basis_dict": basis_dict,
            "H_sparse": H_sparse,
            "H_qubit": H_qubit,
            "include_methods": (backend_name,),
            "use_analytic_gradient": use_analytic_gradient,
            "optimizer_method": optimizer_method,
            "finite_diff_eps": finite_diff_eps,
            "finite_diff_jac": finite_diff_jac,
            "optimizer_options": optimizer_options,
        }

        results = []
        executor_cls = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
        try:
            executor_ctx = executor_cls(max_workers=n_jobs)
        except PermissionError:
            print("Process parallelism unavailable; falling back to thread workers.")
            executor_ctx = ThreadPoolExecutor(max_workers=n_jobs)

        with executor_ctx as executor:
            futures = []
            for trial_idx, trial_x0 in enumerate(x0_list):
                payload = dict(payload_base)
                payload["x0"] = trial_x0
                futures.append((trial_idx, executor.submit(_run_minimize_trial_worker, payload)))

            for trial_idx, future in futures:
                result = future.result()
                print("Parallel trial {} completed: fun={}".format(trial_idx, result.fun))
                results.append(result)

        min_result = min(results, key=lambda result: abs(result.fun))
    else:
        print("Identity init optimization")
        min_result = _minimize_with_fd_control(
            cost,
            x0,
            call_back,
            gradient=gradient,
            optimizer_method=optimizer_method,
            finite_diff_eps=finite_diff_eps,
            finite_diff_jac=finite_diff_jac,
            optimizer_options=optimizer_options,
        )

        for n, trial_x0 in enumerate(x0_list[1:]):
            print("Random Trial {}:".format(n))
            result = _minimize_with_fd_control(
                cost,
                trial_x0,
                call_back,
                gradient=gradient,
                optimizer_method=optimizer_method,
                finite_diff_eps=finite_diff_eps,
                finite_diff_jac=finite_diff_jac,
                optimizer_options=optimizer_options,
            )

            if abs(result.fun) < abs(min_result.fun):
                min_result = result
    
    print("Optimization completed, \nInitial cost: {}\nFinal cost: {}\nEntropies:".format(cost(x0), cost(min_result.x)))
    HQ = jordan_wigner(chem_obt_tbt_to_chem_ferm(obt, tbt, constant))
    ents_initial, HQ_perm = get_permuted_bipartite_entanglement(sym_ops, HQ, n_qubits, fci_gs=fci_gs, verbose=True, return_state=False, return_U=False, log_base='e')
    print("Initial FCI cut entropies:", ents_initial)
    ents_ref_initial, _ = get_permuted_bipartite_entanglement(
        sym_ops,
        HQ,
        n_qubits,
        fci_gs=ref_gs,
        verbose=False,
        return_state=False,
        return_U=False,
        log_base='e',
    )
    print("Initial ref cut entropies:", ents_ref_initial)

    U, rotate_state = orbital_rotation_outputs(
        min_result.x, n_qubits, parameterization
    )
    obt_rot = rotate_chem_obt(obt, U)
    tbt_rot = rotate_chem_tbt(tbt, U)
    HQ_rot = jordan_wigner(chem_obt_tbt_to_chem_ferm(obt_rot, tbt_rot, constant))

    ents_final, HQ_rot_perm = get_permuted_bipartite_entanglement(
        sym_ops,
        HQ_rot,
        n_qubits,
        fci_gs=rotate_state(fci_gs),
        verbose=True,
        return_state=False,
        return_U=False,
        log_base='e',
    )
    ents_ref_final, _ = get_permuted_bipartite_entanglement(
        sym_ops,
        HQ_rot,
        n_qubits,
        fci_gs=rotate_state(ref_gs),
        verbose=False,
        return_state=False,
        return_U=False,
        log_base='e',
    )

    print("Final FCI cut entropies:", ents_final)
    print("Final ref cut entropies:", ents_ref_final)

    if return_info:
        return {
            "params": min_result.x,
            "orbital_rotation_matrix": U,
            "parameterization": parameterization,
            "objective_backend": getattr(evaluator, "backend_name", None),
            "initial_cost": cost(x0),
            "final_cost": cost(min_result.x),
            "optimizer_result": min_result,
        }

    return min_result.x

if __name__ == "__main__":
    directory = './saved/hamiltonians/'
    system = 'LiH_corr'

    with open(directory+system+".pkl", "rb") as f:
        data = pickle.load(f)
    H, fci_e, fci_gs, cisd_e, cisd_gs = data
    molecule = MolecularData(filename=directory+system)
    HQ = jordan_wigner(H)

    n_qubits = count_qubits(HQ)
    Hs = get_sparse_operator(HQ, n_qubits)

    # make tbt for H
    constant, obt_spatial, tbt_chem_spatial = get_chem_tensors(molecule, True)
    obt, tbt = spatial_obt_to_spin_obt(obt_spatial), spatial_tbt_to_spin_tbt(tbt_chem_spatial)

    comm_sq_exp_cisd = lambda s_list: comm_sq_exp_fast(s_list, Hs, cisd_gs, n_qubits)
    comm_sq_exp_fci = lambda s_list: comm_sq_exp_fast(s_list, Hs, fci_gs, n_qubits)
    var_cisd = lambda s_list: variance(s_list, cisd_gs, n_qubits)
    var_fci = lambda s_list: variance(s_list, fci_gs, n_qubits)

    sym_group_score_func = lambda s_list: (-1)*comm_sq_exp_cisd(s_list) # BS score maximized
    sym_metric_func = lambda s: (-1)*sym_group_score_func([s]) # HCT minimized

    list_sym = get_seniority_symmetries(n_qubits)
    list_sym, _ = hct_mod(HQ, n_qubits//2, sym_metric_func=sym_metric_func, use_coeffs_eps=True)
    basis_dict = build_sparse_basis(n_qubits, True)
    res = minimize_cisd_comm_sq(
        list_sym,
        obt,
        tbt,
        constant,
        cisd_gs,
        fci_gs,
        n_qubits,
        basis_dict=basis_dict,
        H_sparse=Hs,
        H_qubit=HQ,
        objective_mode="auto",
        finite_diff_eps=1e-4,
        parameterization="auto_runtime",
        n_trials=2,
        random_seed=0,
        return_info=True,
    )

    optimized_orbital_dir = os.path.join("saved", "optimized_orbitals")
    orbital_tag = "nc_exp_cisd"
    orbital_matrix = np.real_if_close(res["orbital_rotation_matrix"])
    orbital_npy_path, orbital_txt_path = save_optimized_orbital_data(
        optimized_orbital_dir,
        system,
        orbital_tag,
        orbital_matrix,
        list_sym,
        optimization_info=res,
    )
    print("Saved optimized orbital rotation matrix to:", orbital_npy_path)
    print("Saved optimized orbital metadata and symmetries to:", orbital_txt_path)
