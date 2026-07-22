#!/usr/bin/env bash
set -euo pipefail

# Evaluate phase-boundary handoffs on the held-out unseen RL suite.
#
# Usage:
#   bash evaluation/evaluate_hybrid_transition.sh
#   bash evaluation/evaluate_hybrid_transition.sh composition-worst
#   bash evaluation/evaluate_hybrid_transition.sh composition
#
# Optional overrides:
#   DEVICE=cuda EXECUTE_ACTIONS=10 LIMIT=2 PYTHON=... \
#     bash evaluation/evaluate_hybrid_transition.sh composition-worst

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-composition-worst}"
DEVICE="${DEVICE:-cuda}"
EXECUTE_ACTIONS="${EXECUTE_ACTIONS:-10}"
PUSH_MAX_ACTIONS="${PUSH_MAX_ACTIONS:-70}"
PNP_MAX_ACTIONS="${PNP_MAX_ACTIONS:-140}"
EPISODE_DIR="${EPISODE_DIR:-datasets/aloha2-role-composition/raw_npz/primitive_test/unseen_rl}"
RESULT_ROOT="${RESULT_ROOT:-results/hybrid_transition}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "./.venv/Scripts/python.exe" ]]; then
    PYTHON="./.venv/Scripts/python.exe"
  elif [[ -x "./.venv/bin/python" ]]; then
    PYTHON="./.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

case "${MODEL}" in
  composition)
    CHECKPOINT="checkpoints/language_act_composition_balanced_250/best_prior.pt"
    OUTPUT_NAME="composition_best_prior"
    ;;
  composition-worst)
    CHECKPOINT="checkpoints/language_act_composition_balanced_250/best_worst_prior.pt"
    OUTPUT_NAME="composition_best_worst"
    ;;
  *)
    echo "Usage: bash $0 [composition|composition-worst]" >&2
    exit 2
    ;;
esac

[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
}
[[ -d "${EPISODE_DIR}" ]] || {
  echo "Missing held-out suite: ${EPISODE_DIR}" >&2
  exit 1
}

episode_count=$(find "${EPISODE_DIR}" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l)
episode_count="${episode_count//[[:space:]]/}"
if [[ "${episode_count}" -lt 1 ]]; then
  echo "No episode_*.npz files in ${EPISODE_DIR}" >&2
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

MODES=(expert-expert expert-act act-expert)
for mode in "${MODES[@]}"; do
  output_dir="${RESULT_ROOT}/${OUTPUT_NAME}/${mode}"
  echo
  echo "=== model=${MODEL} hybrid=${mode} episodes=${episode_count} ==="
  "${PYTHON}" -u evaluation/evaluate_hybrid_transition.py \
    --mode "${mode}" \
    --checkpoint "${CHECKPOINT}" \
    --episode-dir "${EPISODE_DIR}" \
    --output "${output_dir}" \
    --execute-actions "${EXECUTE_ACTIONS}" \
    --push-max-actions "${PUSH_MAX_ACTIONS}" \
    --pnp-max-actions "${PNP_MAX_ACTIONS}" \
    --device "${DEVICE}" \
    "${LIMIT_ARGS[@]}"
done

echo
echo "Hybrid evaluations completed: ${RESULT_ROOT}/${OUTPUT_NAME}"
