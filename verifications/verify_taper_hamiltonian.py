"""Verify local bra/ket qubit tapering against explicit dense projections.

Run from QuasiSymmetries:
    python verifications/verify_taper_hamiltonian.py
"""

import sys
from pathlib import Path

import numpy as np
from openfermion import QubitOperator, get_sparse_operator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.op_utils import (
    freeze_qubits,
    pauli_matrix_element_with_basis_state,
    permute_sym_to_start,
    taper_hamiltonian,
)


def dense_block(operator, n_qubits, bra_labels, ket_labels):
    """Extract the dense block for fixed leading computational-basis labels."""
    n_tapered = len(bra_labels)
    n_remaining = n_qubits - n_tapered
    block_size = 1 << n_remaining
    bra_index = int("".join(map(str, bra_labels)), 2) if bra_labels else 0
    ket_index = int("".join(map(str, ket_labels)), 2) if ket_labels else 0
    matrix = get_sparse_operator(operator, n_qubits=n_qubits).toarray()
    return matrix[
        bra_index * block_size : (bra_index + 1) * block_size,
        ket_index * block_size : (ket_index + 1) * block_size,
    ]


def verify_single_qubit_pauli_matrix_elements():
    expected = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    for pauli, matrix in expected.items():
        term = ((0, pauli),)
        for bra in (0, 1):
            for ket in (0, 1):
                actual = pauli_matrix_element_with_basis_state(
                    term, [bra], [ket]
                )
                assert np.allclose(actual, matrix[bra, ket])


def verify_all_dense_bra_ket_blocks():
    hamiltonian = (
        0.3 * QubitOperator(())
        + 0.7 * QubitOperator("Z0 X2")
        - 0.4 * QubitOperator("X0 Y1 Z3")
        + 0.2j * QubitOperator("Y0 X1 X2")
        - 0.6 * QubitOperator("Z1 Y2 Y3")
    )
    n_qubits = 4

    for bra_index in range(4):
        bra = [(bra_index >> 1) & 1, bra_index & 1]
        for ket_index in range(4):
            ket = [(ket_index >> 1) & 1, ket_index & 1]
            tapered = taper_hamiltonian(hamiltonian, bra, ket)
            tapered_matrix = get_sparse_operator(
                tapered, n_qubits=2
            ).toarray()
            expected = dense_block(hamiltonian, n_qubits, bra, ket)
            assert np.allclose(tapered_matrix, expected)


def verify_diagonal_taper_matches_freezing():
    hamiltonian = (
        QubitOperator("Z0", 0.5)
        + QubitOperator("Z1 X2", -0.7)
        + QubitOperator("X0 Z3", 0.2)
    )
    labels = [1, 0]
    tapered = taper_hamiltonian(hamiltonian, labels, labels)
    frozen = freeze_qubits(hamiltonian, {0: 1, 1: 0})
    assert tapered == frozen


def verify_after_symmetry_permutation():
    hamiltonian = (
        QubitOperator("X0 X1", 0.8)
        + QubitOperator("Z0 Z1", -0.4)
        + QubitOperator("Y0 Y1 Z2", 0.3)
        + QubitOperator("X2", 0.2)
    )
    symmetry = [QubitOperator("X0 X1")]
    permuted = permute_sym_to_start(
        hamiltonian, symmetry, n_qubits=3
    )

    for bra in ([0], [1]):
        for ket in ([0], [1]):
            tapered = taper_hamiltonian(permuted, bra, ket)
            actual = get_sparse_operator(tapered, n_qubits=2).toarray()
            expected = dense_block(permuted, 3, bra, ket)
            assert np.allclose(actual, expected)


def verify_input_validation():
    hamiltonian = QubitOperator("Z0")
    for bad_labels in ([2], [-1]):
        try:
            taper_hamiltonian(hamiltonian, bad_labels, [0])
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid basis labels should raise ValueError.")

    try:
        taper_hamiltonian(hamiltonian, [0, 1], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("Unequal bra/ket label lengths should raise ValueError.")


def main():
    verify_single_qubit_pauli_matrix_elements()
    verify_all_dense_bra_ket_blocks()
    verify_diagonal_taper_matches_freezing()
    verify_after_symmetry_permutation()
    verify_input_validation()
    print("All taper_hamiltonian verifications passed.")


if __name__ == "__main__":
    main()
