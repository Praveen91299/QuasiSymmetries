## In search of greater ~purpose~ Pauli Quasi Symmetries...  

See `hct_bs_sample.py` for example script to find symmetries and test various metrics.

Notes:  
- HCT_mod should give the same symmetries as found in the HCT paper, but the diagonalizing Clifford is not unique, hence need not match.  
- BS-HCT has been observed to not improve much upon HCT, hence redundant.  
- Beam search (with HCT symmetries added) currently performs best (lowest entanglement/bond dimension for DMRG convergence).  
- DO NOT MODIFY ARCHIVED DATA IN ./saved/results/thesis_data

### Requirements

Python 3.9 or newer. Core dependencies are installed automatically;
tensor-network, chemistry, circuit, and development dependencies are available
as optional extras below.

### Installation

Install the core package in editable mode while developing:

```bash
python -m pip install -e .
```

Optional features can be installed with extras:

```bash
python -m pip install -e ".[tensor-network,chemistry,circuits,dev]"
```

The import name is `quasisymmetries`:

```python
from quasisymmetries import (
    BeamSearch_Symmetries,
    Clifford,
    permute_sym_to_start,
    taper_hamiltonian,
)
```

Workflow and benchmarking scripts remain at the repository root. Reusable
benchmark and MPO helpers are available from `quasisymmetries.benchmark` and
`quasisymmetries.mpo`.

`BenchmarkData.save()` and `BenchmarkData.save_datasets()` use versioned JSON
files. Existing pickle benchmark files remain readable for migration, but
pickle files should only be loaded when their source is trusted.

Clifford synthesis defaults to the historical X-string elimination route.
For Z-native elimination, which can shorten circuits for Z-heavy symmetries:

```python
clifford = Clifford.from_symmetries(
    symmetries,
    n_qubits=n_qubits,
    synthesis_basis="Z",
    generator_mapping="positive_z",
)
```

`generator_mapping="positive_z"` maps each original signed symmetry to
`+Z0`, `+Z1`, ... in input-list order. The default remains
`"row_reduced"` for backward compatibility. `taper_symmetries()` defaults to
the positive-Z mapping so its bra/ket labels refer directly to the original
symmetry list.

To compare both routes using the saved MAY27 H2O/N2 beam symmetries:

```bash
python benchmark_clifford_routes.py
```

Results are written to `saved/results/JUL04/clifford_routes/`.

PyBlock2 Pauli-MPO/DMRG helpers use unique system temporary directories by
default instead of writing to `tmp_block2_pauli/`. Internally owned scratch is
removed when calculations finish or the driver is released. If a persistent
scratch path is supplied explicitly, the caller retains ownership and it is
never deleted automatically. Long-lived manual driver workflows can call:

```python
from quasisymmetries.tn import cleanup_block2_driver

cleanup_block2_driver(driver)
```
