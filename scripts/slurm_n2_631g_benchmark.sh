#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --job-name=n2_631g_dmrg
#SBATCH --output=/scratch/jpraveen/slurm_logs/n2_631g_dmrg_%j.out

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.12 scipy-stack

# Trillium allocates all 192 CPU cores on a node. Select a small, explicit
# Block2 thread team rather than inheriting that allocation size. Override at
# submission with --export=ALL,BLOCK2_THREADS=<n> when testing thread scaling.
BLOCK2_THREADS="${BLOCK2_THREADS:-4}"
export OMP_NUM_THREADS="$BLOCK2_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_SOURCE="$HOME/QuasiSymmetries"
VENV_DIR="$PROJECT/qsenv"
WORK_ROOT="${SLURM_TMPDIR:-$SCRATCH/work_$SLURM_JOB_ID}"
WORK_REPO="$WORK_ROOT/QuasiSymmetries"

# These paths persist between jobs, so rerunning the submission resumes all
# completed Hamiltonian, CISD, reference, symmetry, frame, and DMRG stages.
RESULTS_ROOT="$SCRATCH/results/n2_631g_pyblock2"
PROBE_DIR="$RESULTS_ROOT/probe"
BENCHMARK_DIR="$RESULTS_ROOT/benchmark"

export NUMBA_CACHE_DIR="$WORK_ROOT/numba_cache"
export XDG_CACHE_HOME="$WORK_ROOT/cache"
export MPLCONFIGDIR="$WORK_ROOT/matplotlib_cache"
export IPYTHONDIR="$WORK_ROOT/ipython"
export JUPYTER_CONFIG_DIR="$WORK_ROOT/jupyter_config"
export JUPYTER_RUNTIME_DIR="$WORK_ROOT/jupyter_runtime"
export TMPDIR="$WORK_ROOT/tmp"

mkdir -p "$WORK_ROOT" "$RESULTS_ROOT" "$PROBE_DIR" "$BENCHMARK_DIR"
mkdir -p "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
         "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" \
         "$TMPDIR"

# Do not copy the virtual environment: its executables and activation scripts
# contain absolute paths. Use the persistent environment in $PROJECT instead.
source "$VENV_DIR/bin/activate"

cp -a "$REPO_SOURCE" "$WORK_ROOT/"
cd "$WORK_REPO"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -c "import sys; print('Python:', sys.executable, sys.version)"
python -c "import block2, pyblock2; print('block2:', block2.__file__); print('pyblock2:', pyblock2.__file__)"
python -c "import pyscf, openfermion, openfermionpyscf; print('chemistry dependencies ok')"
python -c "import quasisymmetries; print('quasisymmetries:', quasisymmetries.__file__)"

# Generate or reload the MolecularData and streamed Jordan--Wigner Hamiltonian.
# The benchmark builds its own MPOs, so the separate probe MPO is unnecessary.
python -u scripts/probe_n2_631g_qubit_mpo.py \
  --output-dir "$PROBE_DIR" \
  --hamiltonian-only

# Override at submission time, for example:
#   sbatch --export=ALL,QS_STAGE=reference scripts/slurm_n2_631g_benchmark.sh
# The default "all" executes every remaining stage and reuses checkpoints.
QS_STAGE="${QS_STAGE:-all}"

python -u scripts/benchmark_n2_631g_pyblock2.py \
  --probe-dir "$PROBE_DIR" \
  --output-dir "$BENCHMARK_DIR" \
  --stage "$QS_STAGE" \
  --skip-reference-bond-dims 150 \
  --energy-reference-bond-dim 200 \
  --energy-reference-bond-increment 10 \
  --require-reference-validation \
  --n-threads "$BLOCK2_THREADS" \
  --n-mkl-threads 1 \
  --stack-mem-gb 32.0 \
  --verbose

echo "Benchmark results: $BENCHMARK_DIR"
