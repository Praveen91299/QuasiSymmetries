# %%
# %%
from __future__ import annotations

import json
import os
import pickle
import tempfile
import warnings
import numpy as np
from dataclasses import dataclass, field, fields
from pathlib import Path

from openfermion import QubitOperator


_JSON_SCHEMA = "quasisymmetries.BenchmarkData"
_JSON_VERSION = 1


def _encode_qubit_operator(operator: QubitOperator):
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


def _decode_qubit_operator(data):
    operator = QubitOperator()
    for encoded_term in data["terms"]:
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


@dataclass
class BenchmarkData:
    tag: str = ''
    symmetries: list[QubitOperator] = field(default_factory=list)
    non_commuting_l1: float = 0
    num_commuting_terms: int = 0
    sym_entropy: float = 0
    cut_entropies: list[float] = field(default_factory=list)
    dmrg_bd: int = 0
    single_sector_e: float = 0

    @staticmethod
    def _with_suffix(filename, suffix):
        path = Path(filename)
        return path if path.suffix == suffix else path.with_suffix(path.suffix + suffix)

    @classmethod
    def _resolve_load_path(cls, filename):
        path = Path(filename)
        if path.suffix == ".json":
            return path
        if path.suffix == ".pkl":
            if path.exists():
                return path
            json_path = path.with_suffix(".json")
            return json_path if json_path.exists() else path

        json_path = cls._with_suffix(path, ".json")
        if json_path.exists():
            return json_path

        pickle_path = cls._with_suffix(path, ".pkl")
        if pickle_path.exists():
            return pickle_path

        return json_path

    def to_dict(self):
        """
        Return all dataclass fields as a plain dictionary.
        """
        return {data_field.name: getattr(self, data_field.name) for data_field in fields(self)}

    @classmethod
    def from_dict(cls, data):
        """
        Build a BenchmarkData object from a saved attributes dictionary.
        """
        field_names = {data_field.name for data_field in fields(cls)}
        return cls(**{name: data[name] for name in field_names if name in data})

    def _to_json_dict(self):
        return {
            "tag": self.tag,
            "symmetries": [
                _encode_qubit_operator(symmetry)
                for symmetry in self.symmetries
            ],
            "non_commuting_l1": float(self.non_commuting_l1),
            "num_commuting_terms": int(self.num_commuting_terms),
            "sym_entropy": float(self.sym_entropy),
            "cut_entropies": [
                float(entropy) for entropy in self.cut_entropies
            ],
            "dmrg_bd": int(self.dmrg_bd),
            "single_sector_e": float(self.single_sector_e),
        }

    @classmethod
    def _from_json_dict(cls, data):
        decoded = dict(data)
        decoded["symmetries"] = [
            _decode_qubit_operator(symmetry)
            for symmetry in data.get("symmetries", [])
        ]
        return cls.from_dict(decoded)

    @classmethod
    def _from_saved_payload(cls, payload):
        if isinstance(payload, cls):
            return payload

        if isinstance(payload, dict) and payload.get("__type__") == cls.__name__:
            return cls.from_dict(payload["attributes"])

        if isinstance(payload, dict):
            field_names = {data_field.name for data_field in fields(cls)}
            if field_names.intersection(payload):
                return cls.from_dict(payload)

        raise TypeError(f"Cannot convert saved payload of type {type(payload).__name__} to {cls.__name__}")

    @classmethod
    def _pickle_load(cls, file_obj):
        class BenchmarkDataUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__" and name == cls.__name__:
                    return cls
                return super().find_class(module, name)

        return BenchmarkDataUnpickler(file_obj).load()

    def __str__(self):
        lines = [f"{self.__class__.__name__}:"]
        for data_field in fields(self):
            value = getattr(self, data_field.name)
            if isinstance(value, list):
                lines.append(f"{data_field.name}:")
                if value:
                    lines.extend(f"  {item}" for item in value)
                else:
                    lines.append("  []")
            else:
                lines.append(f"{data_field.name}: {value}")
        return "\n".join(lines)

    def write_to_file(self, filename):
        ent_str = "\n".join([val.__str__() for val in self.cut_entropies])
        with open(filename, 'a') as f:
            print(self.tag, file=f)
            print("Symmetries:", file=f)
            print("\n".join([sym.__str__() for sym in self.symmetries]), file=f)
            print("Non-commutator L1: ", self.non_commuting_l1, file=f)
            print("Entropy: ", self.sym_entropy, file=f)
            print("Commuting terms: ", self.num_commuting_terms, file=f)
            print("Cut entropies:\n", ent_str, file=f)
            print("DMRG conv BD: ", self.dmrg_bd, file=f)
            print("Single sector energy: ", self.single_sector_e, file=f)
    
    @staticmethod
    def _write_json(payload, filename):
        path = Path(filename)
        if path.suffix == ".pkl":
            path = path.with_suffix(".json")
        elif path.suffix != ".json":
            path = path.with_suffix(path.suffix + ".json")

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
                json.dump(payload, file_obj, indent=2, allow_nan=False)
                file_obj.write("\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def _read_json(cls, filename):
        with Path(filename).open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        if payload.get("schema") != _JSON_SCHEMA:
            raise ValueError(
                f"Unsupported benchmark schema: {payload.get('schema')!r}"
            )
        if payload.get("version") != _JSON_VERSION:
            raise ValueError(
                f"Unsupported benchmark schema version: {payload.get('version')!r}"
            )
        return payload

    @classmethod
    def _read_legacy_pickle(cls, filename):
        warnings.warn(
            "Loading legacy pickle data. Only load pickle files you trust; "
            "re-save the result as JSON when possible.",
            UserWarning,
            stacklevel=3,
        )
        with Path(filename).open("rb") as file_obj:
            return cls._pickle_load(file_obj)

    def save(self, filename):
        """
        Save this benchmark as portable, versioned JSON.

        ``.json`` is appended when it is not already present.
        """
        payload = {
            "schema": _JSON_SCHEMA,
            "version": _JSON_VERSION,
            "kind": "single",
            "attributes": self._to_json_dict(),
        }
        return self._write_json(payload, filename)

    @classmethod
    def load(cls, filename):
        """
        Load one JSON benchmark, or a trusted legacy pickle.
        """
        path = cls._resolve_load_path(filename)
        if path.suffix == ".json":
            payload = cls._read_json(path)
            if payload.get("kind") != "single":
                raise ValueError(f"{path} contains a benchmark dataset collection")
            return cls._from_json_dict(payload["attributes"])

        payload = cls._read_legacy_pickle(path)
        return cls._from_saved_payload(payload)
    
    @classmethod
    def save_datasets(cls, datasets: list[BenchmarkData], filename):
        """
        Save datasets as portable, versioned JSON.

        A list is used so entries with duplicate tags remain distinct.
        """
        payload = {
            "schema": _JSON_SCHEMA,
            "version": _JSON_VERSION,
            "kind": "collection",
            "datasets": [data._to_json_dict() for data in datasets],
        }
        return cls._write_json(payload, filename)

    @classmethod
    def load_datasets(cls, filename):
        """
        Load a JSON dataset collection, or a trusted legacy pickle.

        This also accepts the older tag-keyed dictionary format, but returns a
        list so duplicate tags are not lost in newly saved files.
        """
        path = cls._resolve_load_path(filename)
        if path.suffix == ".json":
            payload = cls._read_json(path)
            if payload.get("kind") == "single":
                return [cls._from_json_dict(payload["attributes"])]
            if payload.get("kind") != "collection":
                raise ValueError(f"Unsupported benchmark payload kind in {path}")
            return [
                cls._from_json_dict(data)
                for data in payload["datasets"]
            ]

        payload = cls._read_legacy_pickle(path)

        if isinstance(payload, dict) and payload.get("__type__") == f"{cls.__name__}.datasets":
            return [cls.from_dict(data) for data in payload["datasets"]]

        if isinstance(payload, dict) and payload.get("__type__") == cls.__name__:
            return [cls._from_saved_payload(payload)]

        if isinstance(payload, list):
            return [cls._from_saved_payload(data) for data in payload]

        if isinstance(payload, dict):
            return [cls._from_saved_payload(data) for data in payload.values()]

        return [cls._from_saved_payload(payload)]

    @classmethod
    def view_saved_files(cls, filenames):
        """
        Import and print saved BenchmarkData file(s).
        """
        if isinstance(filenames, (str, Path)):
            filenames = [filenames]

        outputs = []
        for filename in filenames:
            datasets = cls.load_datasets(filename)
            outputs.append(f"{filename}:")
            outputs.extend(str(data) for data in datasets)

        view = "\n\n".join(outputs)
        print(view)
        return view

    @classmethod
    def view_saved_file(cls, filename):
        """
        Import and print one saved BenchmarkData file.
        """
        return cls.view_saved_files([filename])


    @classmethod
    def plot_cut_entropies(cls, datasets: list[BenchmarkData], gs=None, filename: str = None):
        """
        Save and return plot for entropies.
        """
        import matplotlib.pyplot as plt
        from .metrics import get_entropies_at_cuts

        fig, ax = plt.subplots()

        for data in datasets:
            n_qubits = len(data.cut_entropies) + 1
            x = range(1, n_qubits)
            ax.plot(x, data.cut_entropies, label=data.tag)
        
        if gs is not None:
            #reference state
            n_qubits = int(np.log2(len(gs)))
            gs_ent = get_entropies_at_cuts(gs, n_qubits)
            x = range(1, n_qubits)
            ax.plot(x, gs_ent, label="Reference")

        ax.legend()
        ax.set_xlabel("MPS Bond Index", fontsize=14)
        ax.set_ylabel(r"Bipartite entanglement $S_{vN}$", fontsize=14)
        ax.set_xticks(x, x)

        if filename is not None: fig.savefig(filename, dpi=300, bbox_inches="tight")

        return fig

def benchmark_syms(list_syms, HQ, fci_gs, fci_e, n_qubits, N_2_sym=False, verbose=True, print_to_file=None, tag="",
                   compress_cutoff = 1e-10, return_processed_data =False, log_base=np.e):
    """
    Run all benchmarks for symmetries

    """
    import quimb.tensor as qtn
    from .metrics import (
        entropy_pauli_syms,
        find_commuting_paulis,
        get_permuted_bipartite_entanglement,
        get_single_sector_energies,
        universal_grading,
    )
    from .tn import find_dmrg_conv_bd_quimb

    print(tag)
    nc_l1 = universal_grading(list_syms, HQ, verbose=verbose)
    c = len(find_commuting_paulis(HQ, list_syms, verbose=verbose))

    ent, H_perm, clifford, gs_rot = get_permuted_bipartite_entanglement(
        list_syms,
        HQ,
        n_qubits,
        fci_energy=fci_e,
        fci_gs=fci_gs,
        verbose=verbose,
        return_state=True,
        return_clifford=True,
        log_base=log_base,
        use_dmrg=False,
    )
    
    gs_rot_mps = qtn.MatrixProductState.from_dense(gs_rot, cutoff = 1e-20)     
    dmrg_bd, _, dmrg_data = find_dmrg_conv_bd_quimb(H_perm, n_qubits, fci_e, tol=1.6e-3, n_sweeps=100, 
                        reps=1, verbose=False, compress_cutoff = 1e-20, sweep_tol = 1e-6,
                        noise = 1e0, bsz=2, guess_mps = gs_rot_mps, seed=0, return_data=True)

    #ent and dmrg
    if N_2_sym:
        ent_N_2 = entropy_pauli_syms(list_syms, fci_gs, n_qubits, verbose=verbose)
        ss_energies = get_single_sector_energies(HQ, list_syms, n_qubits, verbose=verbose)
        ss_e = np.min(ss_energies)
        #N/2 syms, single sector, BO energies TODO K and BO energies, but they are not really relevant here

        data = BenchmarkData(tag=tag, symmetries=list_syms, non_commuting_l1 = nc_l1, num_commuting_terms=c,  sym_entropy=ent_N_2, cut_entropies=ent, dmrg_bd=dmrg_bd, single_sector_e=ss_e)
    else:
        data = BenchmarkData(tag=tag, symmetries=list_syms, non_commuting_l1 = nc_l1, num_commuting_terms=c, cut_entropies=ent, dmrg_bd=dmrg_bd)
    
    if print_to_file is not None:
        data.write_to_file(print_to_file)

    if return_processed_data:
        processed_data = {
            "H_perm": H_perm,
            "clifford": clifford,
            "gs_rot": gs_rot,
            "mpo": dmrg_data["mpo"],
        }
        return data, processed_data
    else:
        return data
