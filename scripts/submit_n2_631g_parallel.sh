#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="${1:-$HOME/QuasiSymmetries/scripts}"

# One Trillium allocation prepares shared inputs, runs a bounded pool of
# independent frame processes on that node, and then aggregates the results.
JOB_ID=$(sbatch --parsable "$SCRIPT_ROOT/slurm_n2_631g_parallel_node.sh")

echo "N2/6-31G same-node benchmark job: $JOB_ID"
