#!/usr/bin/env bash
# Recovery as a function of initializer perturbation strength.
#
# The synthetic benchmarks perturb the empirical initializer before running STLS
# so the starting point is not the answer. How hard that perturbation is chosen
# to be is a free parameter, and recovery depends on it, so we report the
# dependence instead of a single value. The perturbation is lognormal,
# rate * exp(sigma * randn()), which cannot change the sign of a rate.
#
# At sigma = 1.0 a bad draw can place L-BFGS in a basin from which the objective
# is roughly an order of magnitude above the good one, and no thresholding rule
# recovers from that. The sweep locates where this begins.
#
#   SIGMAS   perturbation strengths   (default 0 0.25 0.5 0.75 1.0)
#   SEEDS    seeds per strength       (default 1..6)
#   SYSTEMS  which benchmarks         (default mm schlogl)
#   MODE     relative | significance  (default relative)
set -u

SIGMAS=${SIGMAS:-"0.0 0.25 0.5 0.75 1.0"}
SEEDS=${SEEDS:-"1 2 3 4 5 6"}
SYSTEMS=${SYSTEMS:-"mm schlogl"}
MODE=${STLS_MODE:-relative}
OUTDIR=${OUTDIR:-/tmp/init_robustness}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$OUTDIR"
echo "sigmas=[$SIGMAS] seeds=[$SEEDS] systems=[$SYSTEMS] mode=$MODE"

for sys in $SYSTEMS; do
  for sig in $SIGMAS; do
    for s in $SEEDS; do
      log="$OUTDIR/${sys}_sig${sig}_seed${s}.log"
      if [ -s "$log" ] && grep -q "PARAMETER RECOVERY" "$log"; then continue; fi
      RUN_SEED="$s" INIT_NOISE_SCALE="$sig" STLS_MODE="$MODE" \
        timeout 5400 julia --project="$ROOT/experiments" "$ROOT/experiments/${sys}_run.jl" \
        > "$log" 2>&1
      nnz=$(grep -oE "Final NNZ: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+$")
      echo "  ${sys} sigma=${sig} seed=${s}  nnz=${nnz:-NA}"
    done
  done
done
echo "ROBUSTNESS_DONE"
