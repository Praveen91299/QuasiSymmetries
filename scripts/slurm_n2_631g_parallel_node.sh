#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=22:00:00
#SBATCH --job-name=n2_631g_frames
#SBATCH --output=/scratch/jpraveen/slurm_logs/n2_631g_frames_%j.out

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.12 scipy-stack

# Each DMRG frame is an independent Python/Block2 process. Eight processes
# with twenty Block2 threads each use 160 of the 192 CPU cores while keeping
# the aggregate Block2 stack allowance well below node memory on a
# 192-core, 745-GiB Trillium node. Override these at submission with
# --export=ALL,FRAME_WORKERS=<n>,BLOCK2_THREADS=<n>.
FRAME_WORKERS="${FRAME_WORKERS:-8}"
BLOCK2_THREADS="${BLOCK2_THREADS:-20}"
if (( FRAME_WORKERS < 1 || BLOCK2_THREADS < 1 )); then
  echo "FRAME_WORKERS and BLOCK2_THREADS must both be positive" >&2
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

if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required to pin same-node frame workers" >&2
  exit 2
fi
# Query the cpuset actually granted to the batch process rather than assuming
# Linux CPU identifiers are 0..191. Worker slots receive disjoint subsets.
mapfile -t ALLOWED_CPUS < <(
  python -c 'import os; print(*sorted(os.sched_getaffinity(0)), sep="\n")'
)
REQUIRED_CPUS=$((FRAME_WORKERS * BLOCK2_THREADS))
if (( REQUIRED_CPUS > ${#ALLOWED_CPUS[@]} )); then
  echo "Requested $REQUIRED_CPUS worker CPUs but only ${#ALLOWED_CPUS[@]} are available" >&2
  exit 2
fi

# Alternate worker slots between CPU sockets to avoid placing every Block2
# process on the first 96-core socket. Linux topology data is used, so this
# remains correct if CPU identifiers are not contiguous.
mapfile -t WORKER_CPU_SETS < <(
  python -c '
import collections
import os
import sys

n_workers = int(sys.argv[1])
n_threads = int(sys.argv[2])
by_socket = collections.defaultdict(list)
for cpu in sorted(os.sched_getaffinity(0)):
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
    with open(path, encoding="utf-8") as handle:
        socket = int(handle.read())
    by_socket[socket].append(cpu)
sockets = sorted(by_socket)
offsets = {socket: 0 for socket in sockets}
for worker in range(n_workers):
    preferred = sockets[worker % len(sockets)]
    choices = [preferred] + [s for s in sockets if s != preferred]
    socket = next(
        s for s in choices
        if offsets[s] + n_threads <= len(by_socket[s])
    )
    start = offsets[socket]
    cpus = by_socket[socket][start : start + n_threads]
    offsets[socket] += n_threads
    print(",".join(map(str, cpus)))
' "$FRAME_WORKERS" "$BLOCK2_THREADS"
)
if (( ${#WORKER_CPU_SETS[@]} != FRAME_WORKERS )); then
  echo "Failed to construct one disjoint CPU set per worker" >&2
  exit 2
fi

python -c "import sys; print('Python:', sys.executable, sys.version)"
python -c "import block2, pyblock2; print('block2:', block2.__file__); print('pyblock2:', pyblock2.__file__)"
python -c "import pyscf, openfermion, openfermionpyscf; print('chemistry dependencies ok')"
python -c "import quasisymmetries; print('quasisymmetries:', quasisymmetries.__file__)"
echo "Available CPUs: ${#ALLOWED_CPUS[@]}"
echo "Frame workers: $FRAME_WORKERS; Block2 threads per worker: $BLOCK2_THREADS"
for ((slot = 0; slot < FRAME_WORKERS; slot++)); do
  echo "Worker slot $slot CPU set: ${WORKER_CPU_SETS[$slot]}"
done

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
  local slot="$2"
  local cpu_list="$3"
  local safe_frame="${frame//[^[:alnum:]_]/_}"
  local frame_root="$WORK_ROOT/frame_workers/$safe_frame"
  local frame_log="$WORKER_LOG_DIR/${safe_frame}_${SLURM_JOB_ID}.log"
  mkdir -p "$frame_root/tmp" "$frame_root/cache" "$frame_root/numba_cache"
  mkdir -p "$frame_root/matplotlib_cache"
  echo "Starting $frame in slot $slot on CPUs $cpu_list; log=$frame_log"
  (
    export TMPDIR="$frame_root/tmp"
    export XDG_CACHE_HOME="$frame_root/cache"
    export NUMBA_CACHE_DIR="$frame_root/numba_cache"
    export MPLCONFIGDIR="$frame_root/matplotlib_cache"
    taskset --cpu-list "$cpu_list" \
    python -u scripts/benchmark_n2_631g_pyblock2.py \
      "${COMMON_ARGS[@]}" \
      --stage dmrg \
      --frames "$frame" \
      --worker-mode
  ) >"$frame_log" 2>&1
}

# Atomic mkdir operations on node-local storage provide dynamic load balancing
# without nested srun steps or shared-file writes. Exactly one worker can
# successfully create the claim directory for a given frame index.
QUEUE_CLAIMS="$WORK_ROOT/frame_queue_claims"
mkdir -p "$QUEUE_CLAIMS"

claim_next_frame() {
  local index
  for ((index = 0; index < ${#FRAMES[@]}; index++)); do
    if mkdir "$QUEUE_CLAIMS/$index" 2>/dev/null; then
      printf '%s' "${FRAMES[$index]}"
      return 0
    fi
  done
  return 1
}

run_worker_slot() {
  local slot="$1"
  local cpu_list="${WORKER_CPU_SETS[$slot]}"
  local frame
  local slot_failed=0
  while frame=$(claim_next_frame); do
    if ! run_frame "$frame" "$slot" "$cpu_list"; then
      echo "Frame $frame failed in worker slot $slot" >&2
      slot_failed=1
    fi
  done
  return "$slot_failed"
}

worker_pids=()
for ((slot = 0; slot < FRAME_WORKERS; slot++)); do
  run_worker_slot "$slot" &
  worker_pids+=("$!")
done

worker_failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    worker_failed=1
  fi
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
