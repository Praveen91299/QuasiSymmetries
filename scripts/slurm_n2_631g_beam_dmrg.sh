#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=22:00:00
#SBATCH --job-name=n2_631g_beam
#SBATCH --output=/scratch/jpraveen/slurm_logs/n2_631g_beam_%j.out

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.12 scipy-stack

# Block2 was faster with four threads than with larger thread teams in the
# N2/6-31G timing tests. Override only when deliberately retesting scaling:
#   sbatch --export=ALL,BLOCK2_THREADS=2 scripts/slurm_n2_631g_beam_dmrg.sh
BLOCK2_THREADS="${BLOCK2_THREADS:-4}"
if (( BLOCK2_THREADS < 1 || BLOCK2_THREADS > 4 )); then
  echo "BLOCK2_THREADS must lie between 1 and 4" >&2
  exit 2
fi

export OMP_NUM_THREADS="$BLOCK2_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# These locations can be changed at submission without editing this file:
#   sbatch --export=ALL,QS_SYSTEMS="eqm",QS_RESULTS_DIR=/scratch/... script.sh
REPO_SOURCE="${QS_REPO_SOURCE:-$HOME/QuasiSymmetries}"
VENV_DIR="${QS_VENV_DIR:-$PROJECT/qsenv}"
RESULTS_DIR="${QS_RESULTS_DIR:-$SCRATCH/results/n2_631g_beam_dmrg}"
SYSTEMS_TEXT="${QS_SYSTEMS:-eqm corr diss}"
STACK_MEM_GB="${STACK_MEM_GB:-32.0}"
MAX_BOND_DIM="${MAX_BOND_DIM:-300}"

WORK_ROOT="${SLURM_TMPDIR:-$SCRATCH/work_$SLURM_JOB_ID}"
WORK_REPO="$WORK_ROOT/QuasiSymmetries"

export NUMBA_CACHE_DIR="$WORK_ROOT/numba_cache"
export XDG_CACHE_HOME="$WORK_ROOT/cache"
export MPLCONFIGDIR="$WORK_ROOT/matplotlib_cache"
export IPYTHONDIR="$WORK_ROOT/ipython"
export JUPYTER_CONFIG_DIR="$WORK_ROOT/jupyter_config"
export JUPYTER_RUNTIME_DIR="$WORK_ROOT/jupyter_runtime"
export TMPDIR="$WORK_ROOT/tmp"

mkdir -p "$WORK_ROOT" "$RESULTS_DIR"
mkdir -p "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
mkdir -p "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR"
mkdir -p "$TMPDIR" "$TMPDIR/block2"

if [[ ! -d "$REPO_SOURCE" ]]; then
  echo "Repository not found: $REPO_SOURCE" >&2
  exit 2
fi
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 2
fi

# Keep the environment at its persistent $PROJECT path. Copying a Python venv
# breaks the absolute paths embedded in its activation scripts and executables.
source "$VENV_DIR/bin/activate"
cp -a "$REPO_SOURCE" "$WORK_ROOT/"
cd "$WORK_REPO"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

read -r -a SYSTEMS <<< "$SYSTEMS_TEXT"
if (( ${#SYSTEMS[@]} < 1 )); then
  echo "QS_SYSTEMS must contain at least one of: eqm corr diss" >&2
  exit 2
fi

python -c "import sys; print('Python:', sys.executable, sys.version)"
python -c "import block2, pyblock2; print('block2:', block2.__file__); print('pyblock2:', pyblock2.__file__)"
python -c "import pyscf, openfermion, openfermionpyscf; print('chemistry dependencies ok')"
python -c "import quasisymmetries; print('quasisymmetries:', quasisymmetries.__file__)"

echo "Geometries: ${SYSTEMS[*]}"
echo "Persistent results: $RESULTS_DIR"
echo "Node-local Block2 scratch: $TMPDIR/block2"
echo "Block2 threads: $BLOCK2_THREADS; stack memory: $STACK_MEM_GB GiB"
echo "Maximum candidate MPS bond dimension: $MAX_BOND_DIM"

# Validate every requested corrected CISD/Hamiltonian/symmetry fingerprint
# before spending hours on the first reference calculation.
python -u scripts/benchmark_n2_631g_beam_dmrg.py \
  --systems "${SYSTEMS[@]}" \
  --output-dir "$RESULTS_DIR" \
  --max-bond-dim "$MAX_BOND_DIM" \
  --dry-run

# Results and completed-bond checkpoints are written directly to persistent
# scratch. Resubmitting the job therefore resumes the reference pair and the
# adaptive qubit search instead of starting over after a time limit.
python -u scripts/benchmark_n2_631g_beam_dmrg.py \
  --systems "${SYSTEMS[@]}" \
  --output-dir "$RESULTS_DIR" \
  --max-bond-dim "$MAX_BOND_DIM" \
  --n-threads "$BLOCK2_THREADS" \
  --n-mkl-threads 1 \
  --stack-mem-gb "$STACK_MEM_GB" \
  --scratch "$TMPDIR/block2" \
  --verbose

echo "Benchmark results: $RESULTS_DIR"
