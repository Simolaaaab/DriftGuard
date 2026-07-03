#!/usr/bin/env bash
# launch_lar_sweep.sh — supervisor's LAR triggers for the cost-of-delay table.
#
# Adds the four LAR variants to tab:al-delay, holding everything else fixed at
# the canonical drift_anchored + prompt-B + seed-2025 config so the new rows are
# directly comparable to the existing kl_only / periodic / or rows:
#
#   random_lar_w100     window 100, 10 random probes (10%)
#   random_lar_w200     window 200, 10 random probes (5%)
#   kl_guided_lar_w100  window 100, 10 KL-divergence-selected probes
#   kl_guided_lar_w200  window 200, 10 KL-divergence-selected probes
#
# Δ (lar_drop_delta) defaults to the preset's 0.10; override with LAR_DELTA.
# Calibrate first:  python3 reboot/step3_al/calibrate_lar.py --window 200 \
#                       --probe-k 10 --select random
#
# snapshot_every=1 so eval_stream_cumulative.py can resolve the live model at
# every stream row (required for the cost-of-delay replay).

set -e
cd "$(dirname "$0")/../.."

mkdir -p reboot/runs/al/_logs

if [ -z "$AZURE_API_KEY" ] || [ -z "$AZURE_BASE_URL" ]; then
  echo "ERROR: AZURE_API_KEY or AZURE_BASE_URL not set. Aborting."
  exit 1
fi
: "${AZURE_MODEL:=DeepSeek-V4-Flash}"
: "${LAR_DELTA:=0.10}"
export AZURE_MODEL
echo "[oracle] AZURE_MODEL=${AZURE_MODEL}  LAR_DELTA=${LAR_DELTA}"

# 0) Materialise the complete per-stream LLM-label table (+backfill the ~410
#    rows with no cached B label) so LAR is computed on 100% coverage.
echo "[labels] building complete stream LLM-label table (with backfill)..."
python3 reboot/step3_al/lar_labels.py --backfill --model "${AZURE_MODEL}"

# 1) The four LAR runs.
for PRESET in random_lar_w100 random_lar_w200 kl_guided_lar_w100 kl_guided_lar_w200; do
  RUN_ID="drift_anchored_${PRESET}_B_seed2025_full_snap"
  LOG="reboot/runs/al/_logs/${RUN_ID}.log"

  if [ -f "reboot/runs/al/${RUN_ID}/summary.json" ]; then
    echo "[skip] ${RUN_ID} already has summary.json"
    continue
  fi

  echo "=========================================="
  echo "[launch] ${RUN_ID}  preset=${PRESET}  Δ=${LAR_DELTA}"
  echo "=========================================="

  python3 reboot/step3_al/run_al.py \
    --strategy drift_anchored \
    --oracle llm \
    --oracle-prompt B \
    --threshold-preset "${PRESET}" \
    --lar-drop-delta "${LAR_DELTA}" \
    --snapshot-every 1 \
    --max-rounds 30 \
    --seed 2025 \
    --run-id "${RUN_ID}" \
    2>&1 | tee "${LOG}"

  echo "[done]  ${RUN_ID}"
done

# 2) Cost-of-delay replay over ALL trigger runs (existing + new LAR), so the
#    comparison print includes the probe / total-LLM cost columns.
echo
echo "All LAR runs complete. Generating cost-of-delay eval for the full set..."
python3 reboot/step3_al/eval_stream_cumulative.py \
  --runs-glob "drift_anchored_*_B_seed2025_full_snap"
