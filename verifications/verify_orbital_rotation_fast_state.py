"""
Verify fast real orbital-rotation state application.

Run from QuasiSymmetries:
    python verifications/verify_orbital_rotation_fast_state.py
"""
import time

import numpy as np


from openfermion import QubitOperator, get_sparse_operator

from quasisymmetries.orbital_rotation import (
    RealOrbitalRotation,
    apply_real_orbital_rotation_to_sparse_operator,
    apply_real_orbital_rotation_to_state,
)


def main():
    rng = np.random.default_rng(7)
    max_err = 0.0

    for n_qubits in range(2, 8):
        params = rng.normal(scale=0.2, size=RealOrbitalRotation.num_params(n_qubits))
        state = rng.normal(size=1 << n_qubits) + 1j * rng.normal(size=1 << n_qubits)
        state /= np.linalg.norm(state)

        t0 = time.perf_counter()
        slow = RealOrbitalRotation(n_qubits, params).get_exp_rep() @ state
        slow_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        fast = apply_real_orbital_rotation_to_state(params, state, n_qubits)
        fast_time = time.perf_counter() - t0

        err = np.linalg.norm(slow - fast)
        max_err = max(max_err, err)

        H = get_sparse_operator(
            QubitOperator("X0", 0.3) + QubitOperator("Z0 Z1", -0.7),
            n_qubits,
        )
        slow_H = RealOrbitalRotation(n_qubits, params).get_exp_rep()
        slow_H = slow_H @ H @ slow_H.T.conjugate()
        fast_H = apply_real_orbital_rotation_to_sparse_operator(params, H, n_qubits)
        op_err = np.linalg.norm((slow_H - fast_H).toarray())
        max_err = max(max_err, op_err)
        print(
            f"n={n_qubits:2d} error={err:.3e} "
            f"op_error={op_err:.3e} slow={slow_time:.4f}s fast={fast_time:.4f}s"
        )

    assert max_err < 1e-10, f"fast state rotation mismatch: {max_err}"
    print("OK")


if __name__ == "__main__":
    main()
