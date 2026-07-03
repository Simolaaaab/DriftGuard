#!/usr/bin/env bash
# launch_trigger_sweep.sh — runs 3 trigger variants on drift_anchored+B+seed2025.
#
# Selector = drift_anchored (the paper's hero/novelty + canonical best).
# Trigger ∈ {periodic, kl_only, or} = the actual ablation knob.
# Same oracle (DeepSeek-V4-Flash), same step2_cache, same seed as the
# canonical drift_anchored_B_seed2025 → the trigger-comparison headline
# numbers ARE directly comparable to the existing Main Results.
# In particular, drift_anchored_periodic_B_seed2025_full_snap should
# reproduce drift_anchored_B_seed2025 (different snapshot cadence only).
#
# All runs use snapshot_every=1 so eval_stream_cumulative.py can resolve
# the exact live model at every stream row.

set -e
cd "$(dirname "$0")/../.."

mkdir -p reboot/runs/al/_logs

# Sanity check on credentials.
if [ -z "$AZURE_API_KEY" ] || [ -z "$AZURE_BASE_URL" ]; then
  echo "ERROR: AZURE_API_KEY or AZURE_BASE_URL not set. Aborting."
  exit 1
fi
: "${AZURE_MODEL:=DeepSeek-V4-Flash}"
export AZURE_MODEL
echo "[oracle] AZURE_MODEL=${AZURE_MODEL}"

# Three triggers (down from 5 — boss's call):
#   periodic  → dumb timer baseline (Section X.0 in the paper)
#   kl_only   → input-based, robust to model confidence saturation
#   or        → composite (delegate OR KL), most operationally sensitive
# Dropped:
#   AND       → would-fire analysis already shows it fires ~8/25 on natural
#               drift (delegate band saturates). Don't waste an LLM run.
#   random_b. → timing-ablation; previous data already supports the finding
#               that random_batching ≈ periodic at fixed cadence.
declare -A PRESETS=(
  ["periodic"]="periodic_audit"
  ["kl_only"]="kl_only"
  ["or"]="or_audit"
)

for TRIG in periodic kl_only or; do
  PRESET="${PRESETS[$TRIG]}"
  RUN_ID="drift_anchored_${TRIG}_B_seed2025_full_snap"
  LOG="reboot/runs/al/_logs/${RUN_ID}.log"

  if [ -f "reboot/runs/al/${RUN_ID}/summary.json" ]; then
    echo "[skip] ${RUN_ID} already has summary.json"
    continue
  fi

  echo "=========================================="
  echo "[launch] ${RUN_ID}  preset=${PRESET}  model=${AZURE_MODEL}"
  echo "  log → ${LOG}"
  echo "=========================================="

  python3 reboot/step3_al/run_al.py \
    --strategy drift_anchored \
    --oracle llm \
    --oracle-prompt B \
    --threshold-preset "${PRESET}" \
    --snapshot-every 1 \
    --max-rounds 30 \
    --seed 2025 \
    --run-id "${RUN_ID}" \
    2>&1 | tee "${LOG}"

  echo "[done]  ${RUN_ID}"
done

echo
echo "All trigger runs complete. Generating stream-cumulative eval..."
python3 reboot/step3_al/eval_stream_cumulative.py \
  --runs-glob "drift_anchored_*_B_seed2025_full_snap"
