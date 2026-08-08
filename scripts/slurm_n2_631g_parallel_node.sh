#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=22:00:00
#SBATCH --job-name=n2_631g_frames
#SBATCH --output=/scratch/jpraveen/slurm_logs/n2_631g_frames_%j.out

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.12 scipy-stack

# Each DMRG frame is an independent Python/Block2 process. Four processes
# with four Block2 threads each are a conservative starting point on a
# 192-core, 745-GiB Trillium node. Override these at submission with
# --export=ALL,FRAME_WORKERS=<n>,BLOCK2_THREADS=<n>.
FRAME_WORKERS="${FRAME_WORKERS:-4}"
BLOCK2_THREADS="${BLOCK2_THREADS:-4}"
if (( FRAME_WORKERS < 1 || BLOCK2_THREADS < 1 )); then
  echo "FRAME_WORKERS and BLOCK2_THREADS must both be positive" >&2
  exit 2
fi
if (( FRAME_WORKERS * BLOCK2_THREADS > SLURM_CPUS_PER_TASK )); then
  echo "FRAME_WORKERS * BLOCK2_THREADS exceeds allocated CPUs" >&2
  exit 2
fi

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
RESULTS_ROOT="$SCRATCH/results/n2_631g_pyblock2"
PROBE_DIR="$RESULTS_ROOT/probe"
BENCHMARK_DIR="$RESULTS_ROOT/benchmark"
WORKER_LOG_DIR="$BENCHMARK_DIR/worker_logs"

export NUMBA_CACHE_DIR="$WORK_ROOT/numba_cache"
export XDG_CACHE_HOME="$WORK_ROOT/cache"
export MPLCONFIGDIR="$WORK_ROOT/matplotlib_cache"
export IPYTHONDIR="$WORK_ROOT/ipython"
export JUPYTER_CONFIG_DIR="$WORK_ROOT/jupyter_config"
export JUPYTER_RUNTIME_DIR="$WORK_ROOT/jupyter_runtime"
export TMPDIR="$WORK_ROOT/tmp"

mkdir -p "$WORK_ROOT" "$RESULTS_ROOT" "$PROBE_DIR" "$BENCHMARK_DIR"
mkdir -p "$WORKER_LOG_DIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"
mkdir -p "$MPLCONFIGDIR" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$TMPDIR"

source "$VENV_DIR/bin/activate"
cp -a "$REPO_SOURCE" "$WORK_ROOT/"
cd "$WORK_REPO"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -c "import sys; print('Python:', sys.executable, sys.version)"
python -c "import block2, pyblock2; print('block2:', block2.__file__); print('pyblock2:', pyblock2.__file__)"
python -c "import pyscf, openfermion, openfermionpyscf; print('chemistry dependencies ok')"
python -c "import quasisymmetries; print('quasisymmetries:', quasisymmetries.__file__)"

COMMON_ARGS=(
  --probe-dir "$PROBE_DIR"
  --output-dir "$BENCHMARK_DIR"
  --skip-reference-bond-dims 150
  --energy-reference-bond-dim 200
  --energy-reference-bond-increment 10
  --no-require-reference-validation
  --n-threads "$BLOCK2_THREADS"
  --n-mkl-threads 1
  --stack-mem-gb 32.0
  --verbose
)

# All shared files are prepared by one process before frame workers start.
python -u scripts/probe_n2_631g_qubit_mpo.py \
  --output-dir "$PROBE_DIR" \
  --hamiltonian-only
python -u scripts/benchmark_n2_631g_pyblock2.py \
  "${COMMON_ARGS[@]}" \
  --stage frames

FRAMES=(
  raw_fermionic_su2
  raw_qubit
  seniority_Nover2
  HCT_Nover2_CommL1
  HCT_N_CommL1
  Beam_Nover2_CommL1
  Beam_N_CommL1
  HCT_N_CommL1_Fiedler
  Beam_N_CommL1_Fiedler
  BLISS_HCT_N_CommL1
  BLISS_Beam_N_CommL1
)

run_frame() {
  local frame="$1"
  local safe_frame="${frame//[^[:alnum:]_]/_}"
  local frame_root="$WORK_ROOT/frame_workers/$safe_frame"
  local frame_log="$WORKER_LOG_DIR/${safe_frame}_${SLURM_JOB_ID}.log"
  mkdir -p "$frame_root/tmp" "$frame_root/cache" "$frame_root/numba_cache"
  mkdir -p "$frame_root/matplotlib_cache"
  echo "Starting $frame; log=$frame_log"
  (
    export TMPDIR="$frame_root/tmp"
    export XDG_CACHE_HOME="$frame_root/cache"
    export NUMBA_CACHE_DIR="$frame_root/numba_cache"
    export MPLCONFIGDIR="$frame_root/matplotlib_cache"
    srun --exclusive --nodes=1 --ntasks=1 \
      --cpus-per-task="$BLOCK2_THREADS" --cpu-bind=cores \
      python -u scripts/benchmark_n2_631g_pyblock2.py \
        "${COMMON_ARGS[@]}" \
        --stage dmrg \
        --frames "$frame" \
        --worker-mode
  ) >"$frame_log" 2>&1
}

# Maintain a bounded same-node process pool. wait -n releases a slot whenever
# any worker finishes, so a slow frame does not hold up launching later ones.
running=0
worker_failed=0
for frame in "${FRAMES[@]}"; do
  run_frame "$frame" &
  running=$((running + 1))
  if (( running >= FRAME_WORKERS )); then
    if ! wait -n; then
      worker_failed=1
    fi
    running=$((running - 1))
  fi
done
while (( running > 0 )); do
  if ! wait -n; then
    worker_failed=1
  fi
  running=$((running - 1))
done

# Aggregate every completed frame even if one worker failed. The aggregate
# records any incomplete frame names and can be regenerated after resubmission.
python -u scripts/benchmark_n2_631g_pyblock2.py \
  "${COMMON_ARGS[@]}" \
  --stage aggregate

echo "Benchmark results: $BENCHMARK_DIR"
echo "Per-frame logs: $WORKER_LOG_DIR"
if (( worker_failed != 0 )); then
  echo "At least one frame worker failed; inspect the per-frame logs." >&2
  exit 1
fi
