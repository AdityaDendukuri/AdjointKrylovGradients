#!/usr/bin/env bash
# Seed ensemble over the four synthetic CME benchmarks.
#
# The published table reported one run per system. With the old initializer that
# was a roughly 1-in-5 favourable draw for the three systems that used
# empirical_initialization, because rate*(1 + 5*randn()) is negative 42% of the
# time and the subsequent max(., 1e-8) deleted the direction outright. With the
# lognormal perturbation nothing is deleted, so a seed ensemble measures what the
# method actually does rather than which draw was lucky.
#
#   SEEDS         space-separated seeds        (default 1..10)
#   STLS_MODE     relative | significance      (default relative)
#   STLS_SIGLEVEL significance level           (default 5.0)
#   OUTDIR        where per-run logs are kept
set -u

SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
MODE=${STLS_MODE:-relative}
SIGLEVEL=${STLS_SIGLEVEL:-5.0}
NOISE_MODEL=${INIT_NOISE_MODEL:-lognormal}
NOISE_SCALE=${INIT_NOISE_SCALE:-1.0}
OUTDIR=${OUTDIR:-/tmp/seed_ensemble}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$OUTDIR"
echo "seeds=[$SEEDS] mode=$MODE siglevel=$SIGLEVEL noise=$NOISE_MODEL/$NOISE_SCALE"
echo "logs -> $OUTDIR"

for sys in mm schlogl brusselator lv; do
  for s in $SEEDS; do
    log="$OUTDIR/${sys}_seed${s}.log"
    if [ -s "$log" ] && grep -q "PARAMETER RECOVERY" "$log"; then
      echo "  skip ${sys} seed ${s} (already done)"
      continue
    fi
    echo "  running ${sys} seed ${s}"
    RUN_SEED="$s" \
    STLS_MODE="$MODE" STLS_SIGLEVEL="$SIGLEVEL" \
    INIT_NOISE_MODEL="$NOISE_MODEL" INIT_NOISE_SCALE="$NOISE_SCALE" \
    timeout 5400 julia --project="$ROOT/experiments" "$ROOT/experiments/${sys}_run.jl" \
      > "$log" 2>&1
  done
done
echo "ENSEMBLE_DONE"
