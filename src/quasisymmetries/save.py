"""Reusable persistence helpers for benchmark inputs and results.

JSON and NPZ are used for portable data. Pickle helpers are intentionally
named ``load_trusted_pickle`` because pickle files must never be loaded from
an untrusted source.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from openfermion import QubitOperator


def to_jsonable(value):
    """Recursively convert common scientific-Python values to JSON data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, complex):
        if abs(value.imag) < 1e-14:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: str | Path, payload, *, atomic: bool = True) -> Path:
    """Write JSON, atomically by default, and return the resolved output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = to_jsonable(payload)
    if not atomic:
        path.write_text(
            json.dumps(encoded, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_obj:
            temporary_path = Path(file_obj.name)
            json.dump(encoded, file_obj, indent=2, allow_nan=False)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def load_json(path: str | Path):
    """Load a JSON document."""
    with Path(path).open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_csv(path: str | Path, rows: Iterable[Mapping]) -> Path | None:
    """Write heterogeneous result rows, JSON-encoding nested cell values."""
    rows = list(rows)
    if not rows:
        return None
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(to_jsonable(value))
                        if isinstance(
                            value, (dict, list, tuple, np.ndarray)
                        )
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows as strings, matching :class:`csv.DictReader` semantics."""
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def encode_qubit_operator(operator: QubitOperator) -> dict:
    """Encode an OpenFermion QubitOperator as portable JSON data."""
    return {
        "terms": [
            {
                "pauli": [[int(index), pauli] for index, pauli in term],
                "coefficient": {
                    "real": float(complex(coefficient).real),
                    "imag": float(complex(coefficient).imag),
                },
            }
            for term, coefficient in operator.terms.items()
        ]
    }


def decode_qubit_operator(payload: Mapping) -> QubitOperator:
    """Decode data produced by :func:`encode_qubit_operator`."""
    operator = QubitOperator()
    for encoded_term in payload["terms"]:
        term = tuple(
            (int(index), str(pauli))
            for index, pauli in encoded_term["pauli"]
        )
        encoded_coefficient = encoded_term["coefficient"]
        coefficient = complex(
            encoded_coefficient["real"],
            encoded_coefficient["imag"],
        )
        operator += QubitOperator(term, coefficient)
    return operator


def save_qubit_operator(
    path: str | Path, operator: QubitOperator
) -> Path:
    """Save a QubitOperator as portable JSON."""
    return save_json(path, encode_qubit_operator(operator))


def load_qubit_operator(path: str | Path) -> QubitOperator:
    """Load a QubitOperator saved by :func:`save_qubit_operator`."""
    return decode_qubit_operator(load_json(path))


def encode_pauli_term_stream(stream) -> dict:
    """Encode a packed Pauli Hamiltonian as portable JSON-compatible data.

    Parameters
    ----------
    stream
        ``PauliTermStream`` or compatible OpenFermion ``QubitOperator``.

    Returns
    -------
    payload
        Dictionary containing the qubit count, integer X/Z masks, and complex
        coefficients split into real and imaginary components.
    """
    from .bs.utils import as_pauli_term_stream

    stream = as_pauli_term_stream(stream)
    return {
        "format": "pauli_mask_coefficients_v1",
        "n_qubits": stream.n_qubits,
        "terms": [
            {
                "x_mask": int(item.mask[0]),
                "z_mask": int(item.mask[1]),
                "coefficient": {
                    "real": float(item.signed_coefficient.real),
                    "imag": float(item.signed_coefficient.imag),
                },
            }
            for item in stream.terms
        ],
    }


def decode_pauli_term_stream(payload: Mapping, n_qubits: int | None = None):
    """Decode packed-mask or legacy QubitOperator JSON into a Pauli stream.

    Parameters
    ----------
    payload
        Mapping produced by ``encode_pauli_term_stream`` or the repository's
        legacy QubitOperator JSON encoder.
    n_qubits
        Optional explicit qubit count, required to preserve unused trailing
        qubits when reading legacy data.

    Returns
    -------
    stream
        Decoded ``PauliTermStream`` with combined coefficients.
    """
    from .bs.utils import PauliTermStream, term_to_masks

    if payload.get("format") == "pauli_mask_coefficients_v1":
        stored_n_qubits = int(payload["n_qubits"])
        if n_qubits is not None and int(n_qubits) != stored_n_qubits:
            raise ValueError("stored and requested qubit counts differ")
        return PauliTermStream.from_terms(
            stored_n_qubits,
            (
                (
                    (int(item["x_mask"]), int(item["z_mask"])),
                    complex(
                        item["coefficient"]["real"],
                        item["coefficient"]["imag"],
                    ),
                )
                for item in payload["terms"]
            ),
        )

    # Read the repository's older QubitOperator JSON format directly into
    # masks, without first constructing an OpenFermion operator dictionary.
    if n_qubits is None:
        n_qubits = 0
        for item in payload["terms"]:
            for index, _pauli in item["pauli"]:
                n_qubits = max(n_qubits, int(index) + 1)
    return PauliTermStream.from_terms(
        int(n_qubits),
        (
            (
                term_to_masks(
                    tuple(
                        (int(index), str(pauli))
                        for index, pauli in item["pauli"]
                    ),
                    int(n_qubits),
                ),
                complex(
                    item["coefficient"]["real"],
                    item["coefficient"]["imag"],
                ),
            )
            for item in payload["terms"]
        ),
    )


def save_pauli_term_stream(path: str | Path, stream) -> Path:
    """Save a packed Pauli stream and return the written path.

    Parameters
    ----------
    path
        Destination JSON path.
    stream
        ``PauliTermStream`` or compatible OpenFermion ``QubitOperator``.

    Returns
    -------
    path
        Destination path returned by ``save_json``.
    """
    return save_json(path, encode_pauli_term_stream(stream))


def load_pauli_term_stream(path: str | Path, n_qubits: int | None = None):
    """Load packed-mask or legacy Hamiltonian JSON as a Pauli stream.

    Parameters
    ----------
    path
        Source JSON path.
    n_qubits
        Optional explicit qubit count for legacy Hamiltonian JSON.

    Returns
    -------
    stream
        Decoded ``PauliTermStream``.
    """
    return decode_pauli_term_stream(load_json(path), n_qubits=n_qubits)


def save_sparse_qubit_state(path: str | Path, state) -> Path:
    """Save a SparseQubitState as compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_qubits=np.asarray(state.n_qubits, dtype=np.int64),
        indices=np.asarray(state.indices, dtype=np.int64),
        coeffs=np.asarray(state.coeffs, dtype=np.complex128),
    )
    return path


def load_sparse_qubit_state(path: str | Path):
    """Load a compressed SparseQubitState without constructing a dense vector."""
    from .state_utils import SparseQubitState

    with np.load(Path(path), allow_pickle=False) as data:
        return SparseQubitState(
            data["indices"],
            data["coeffs"],
            n_qubits=int(data["n_qubits"].item()),
        )


def save_pickle(path: str | Path, value) -> Path:
    """Save an exact Python representation when no portable format exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(value, file_obj)
    return path


def load_trusted_pickle(path: str | Path):
    """Load a trusted pickle file. Never use this on untrusted input."""
    with Path(path).open("rb") as file_obj:
        return pickle.load(file_obj)
