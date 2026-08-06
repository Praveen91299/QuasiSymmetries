from pathlib import Path

import numpy as np
from openfermion import QubitOperator

from quasisymmetries.save import (
    load_json,
    load_qubit_operator,
    load_sparse_qubit_state,
    load_trusted_pickle,
    read_csv,
    save_json,
    save_pickle,
    save_qubit_operator,
    save_sparse_qubit_state,
    write_csv,
)
from quasisymmetries.state_utils import SparseQubitState


def test_json_and_csv_helpers(tmp_path):
    json_path = save_json(
        tmp_path / "data.json",
        {
            "path": Path("somewhere"),
            "array": np.asarray([1, 2]),
            "complex": 2.0 + 3.0j,
        },
    )
    payload = load_json(json_path)
    assert payload == {
        "path": "somewhere",
        "array": [1, 2],
        "complex": {"real": 2.0, "imag": 3.0},
    }

    csv_path = write_csv(
        tmp_path / "rows.csv",
        [{"frame": "raw", "sweeps": [0.1, 0.2]}],
    )
    assert read_csv(csv_path) == [
        {"frame": "raw", "sweeps": "[0.1, 0.2]"}
    ]


def test_qubit_operator_round_trip(tmp_path):
    operator = (
        1.25 * QubitOperator(())
        + (0.5 - 0.75j) * QubitOperator("X0 Y2")
    )
    path = save_qubit_operator(tmp_path / "operator.json", operator)
    assert load_qubit_operator(path) == operator


def test_sparse_state_and_pickle_round_trip(tmp_path):
    state = SparseQubitState(
        [1, 6],
        [1 / np.sqrt(2), 1j / np.sqrt(2)],
        n_qubits=3,
    )
    state_path = save_sparse_qubit_state(tmp_path / "state.npz", state)
    loaded = load_sparse_qubit_state(state_path)
    assert loaded.n_qubits == state.n_qubits
    assert np.array_equal(loaded.indices, state.indices)
    assert np.allclose(loaded.coeffs, state.coeffs)

    pickle_path = save_pickle(tmp_path / "trusted.pkl", {"value": 7})
    assert load_trusted_pickle(pickle_path) == {"value": 7}
