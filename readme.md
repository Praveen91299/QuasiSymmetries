## In search of greater ~purpose~ Pauli Quasi Symmetries...  

See `scripts/hct_bs_sample.py` for example script to find symmetries and test various metrics.

Notes:  
- `HCT` follows the tapering paper by applying symplectic Gram--Schmidt to each threshold kernel; `hct_mod` retains the original greedy implementation.
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

Workflow and benchmarking scripts live in `scripts/`. Reusable
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
python scripts/benchmark_clifford_routes.py
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

### N2/6-31G scalable benchmark

Generate the full 18-spatial-orbital, 36-qubit Hamiltonian and test its raw
pyblock2 Pauli MPO without running FCI or DMRG:

```bash
python -u scripts/probe_n2_631g_qubit_mpo.py
```

The larger-basis benchmark is checkpointed by stage. A cautious first run is:

```bash
python -u scripts/benchmark_n2_631g_pyblock2.py --stage cisd
python -u scripts/benchmark_n2_631g_pyblock2.py --stage symmetries
python -u scripts/benchmark_n2_631g_pyblock2.py --stage reference
python -u scripts/benchmark_n2_631g_pyblock2.py --stage frames
python -u scripts/benchmark_n2_631g_pyblock2.py --stage dmrg
```

Block2 and MKL/BLAS thread counts are controlled independently. For example,
to use four Block2 threads while avoiding nested BLAS parallelism:

```bash
python -u scripts/benchmark_n2_631g_pyblock2.py \
  --stage reference \
  --n-threads 4 \
  --n-mkl-threads 1
```

Thread/process counts, stack-memory allocation, and verbosity are execution
settings: changing them does not invalidate completed CISD, reference,
symmetry, or frame checkpoints. They are still recorded in ``settings.json``.
Completed DMRG outputs are also reused; use ``--force-stage`` or a new output
directory only when those calculations themselves should be repeated.

Each command reuses completed prerequisites. The reference stage currently
uses fixed bond dimension 100; dimension 150 remains in the requested grid but
is skipped by default because its memory requirement is impractical for this
N2/6-31G MPO. Consequently, the M=100 energy is a converged fixed-bond
variational reference but is not independently validated against a larger
bond dimension. The DMRG stage processes and releases one fermionic or qubit
frame at a time. To pilot only the raw representations before transformed
frames, use:

```bash
python -u scripts/benchmark_n2_631g_pyblock2.py \
  --stage dmrg \
  --output-dir saved/results/n2_631g_raw_pilot \
  --frames raw_fermionic_su2 raw_qubit \
  --bond-dims 10 20 30 40 60 80 100
```

Changing a result-defining saved-run setting requires a new ``--output-dir``.
Use ``--force-stage`` only when the selected stage and all prerequisites should
be recomputed deliberately.
