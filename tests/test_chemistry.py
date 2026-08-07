import numpy as np
from pyscf.ci import cisd
from pyscf.fci import cistring

from quasisymmetries.chemistry import (
    _pyscf_to_interleaved_jw_phases,
    restricted_cisd_vector_to_sparse_qubit_state,
)


def _fci_matrix_to_interleaved_dense(fcivec, nmo, nocc):
    addresses = np.arange(fcivec.shape[0])
    strings = cistring.addrs2str(nmo, nocc, addresses)
    output = np.zeros(1 << (2 * nmo), dtype=complex)
    for alpha_address, alpha_string in enumerate(strings):
        for beta_address, beta_string in enumerate(strings):
            index = 0
            for orbital in range(nmo):
                index |= (
                    (int(alpha_string) >> orbital) & 1
                ) << (2 * nmo - 1 - 2 * orbital)
                index |= (
                    (int(beta_string) >> orbital) & 1
                ) << (2 * nmo - 2 - 2 * orbital)
            phase = _pyscf_to_interleaved_jw_phases(
                [alpha_string], [beta_string], nmo
            )[0]
            output[index] = phase * fcivec[alpha_address, beta_address]
    return output


def test_sparse_cisd_expansion_matches_pyscf_fcivec():
    nmo = 4
    nocc = 2
    size = 1 + nocc * (nmo - nocc) + nocc**2 * (nmo - nocc) ** 2
    vector = np.random.default_rng(7).normal(size=size)

    expected = _fci_matrix_to_interleaved_dense(
        cisd.to_fcivec(vector, nmo, (nocc, nocc)), nmo, nocc
    )
    actual = restricted_cisd_vector_to_sparse_qubit_state(
        vector, nmo, nocc
    ).to_dense()

    np.testing.assert_allclose(actual, expected)


def test_sparse_cisd_expansion_applies_coefficient_tolerance():
    nmo = 3
    nocc = 1
    size = 1 + nocc * (nmo - nocc) + nocc**2 * (nmo - nocc) ** 2
    vector = np.zeros(size)
    vector[0] = 1.0
    vector[1] = 1e-8

    state = restricted_cisd_vector_to_sparse_qubit_state(
        vector, nmo, nocc, coefficient_tolerance=1e-7
    )

    assert state.nnz == 1
    assert state.coeffs[0] == 1.0


def test_pyscf_to_interleaved_phase_counts_alpha_beta_crossings():
    # alpha orbital 1 crosses occupied beta orbital 0: one minus sign.
    phases = _pyscf_to_interleaved_jw_phases(
        alpha_strings=[0b10, 0b01, 0b11],
        beta_strings=[0b01, 0b10, 0b11],
        n_spatial_orbitals=2,
    )

    np.testing.assert_array_equal(phases, [-1.0, 1.0, -1.0])
