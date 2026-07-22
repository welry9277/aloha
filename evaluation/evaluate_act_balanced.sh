#!/usr/bin/env bash
set -euo pipefail

# Evaluate the balanced ACT experiments on the fixed main/primitive test suites.
#
# Usage:
#   bash evaluation/evaluate_act_balanced.sh
#   bash evaluation/evaluate_act_balanced.sh seen
#   bash evaluation/evaluate_act_balanced.sh composition
#   bash evaluation/evaluate_act_balanced.sh composition-worst
#
# Optional environment overrides:
#   DEVICE=cuda EXECUTE_ACTIONS=10 MAX_ACTIONS=220 PYTHON=... \
#     bash evaluation/evaluate_act_balanced.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-all}"
DEVICE="${DEVICE:-cuda}"
EXECUTE_ACTIONS="${EXECUTE_ACTIONS:-10}"
MAX_ACTIONS="${MAX_ACTIONS:-220}"
RESULT_ROOT="${RESULT_ROOT:-results/act_balanced}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "./.venv/Scripts/python.exe" ]]; then
    PYTHON="./.venv/Scripts/python.exe"
  elif [[ -x "./.venv/bin/python" ]]; then
    PYTHON="./.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

declare -A CHECKPOINTS=(
  [seen]="checkpoints/language_act_seen_only_50/best_prior.pt"
  [composition]="checkpoints/language_act_composition_balanced_250/best_prior.pt"
  [composition-worst]="checkpoints/language_act_composition_balanced_250/best_worst_prior.pt"
)

declare -A OUTPUT_NAMES=(
  [seen]="seen_best_prior"
  [composition]="composition_best_prior"
  [composition-worst]="composition_best_worst"
)

declare -A TASK_DIRS=(
  [seen_lr]="datasets/aloha2-role-composition/raw_npz/primitive_test/seen_lr"
  [unseen_rl]="datasets/aloha2-role-composition/raw_npz/primitive_test/unseen_rl"
  [left_tray_push]="datasets/aloha2-role-composition/raw_npz/primitive_test/left_tray_push"
  [right_tray_push]="datasets/aloha2-role-composition/raw_npz/primitive_test/right_tray_push"
  [left_pick_place]="datasets/aloha2-role-composition/raw_npz/primitive_test/left_pick_place"
  [right_pick_place]="datasets/aloha2-role-composition/raw_npz/primitive_test/right_pick_place"
)

declare -A EXPECTED_EPISODES=(
  [seen_lr]=50
  [unseen_rl]=50
  [left_tray_push]=20
  [right_tray_push]=20
  [left_pick_place]=20
  [right_pick_place]=20
)

ALL_MODELS=(seen composition composition-worst)
ALL_TASKS=(
  seen_lr
  unseen_rl
  left_tray_push
  right_tray_push
  left_pick_place
  right_pick_place
)

case "${MODEL}" in
  all) MODELS=("${ALL_MODELS[@]}") ;;
  seen|composition|composition-worst) MODELS=("${MODEL}") ;;
  *)
    echo "Usage: bash $0 [all|seen|composition|composition-worst]" >&2
    exit 2
    ;;
esac

[[ -f "evaluation/evaluate_language_act_suite.py" ]] || {
  echo "Missing evaluator: ${PROJECT_ROOT}/evaluation/evaluate_language_act_suite.py" >&2
  exit 1
}

echo "Checking test suites..."
for task in "${ALL_TASKS[@]}"; do
  episode_dir="${TASK_DIRS[$task]}"
  expected="${EXPECTED_EPISODES[$task]}"
  [[ -d "${episode_dir}" ]] || {
    echo "Missing test directory: ${episode_dir}" >&2
    exit 1
  }

  count=$(find "${episode_dir}" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l)
  count="${count//[[:space:]]/}"
  if [[ "${count}" -ne "${expected}" ]]; then
    echo "Expected ${expected} episodes for ${task}, found ${count}: ${episode_dir}" >&2
    exit 1
  fi
  echo "  ${task}: ${count} episodes"
done

for model in "${MODELS[@]}"; do
  checkpoint="${CHECKPOINTS[$model]}"
  output_name="${OUTPUT_NAMES[$model]}"
  [[ -f "${checkpoint}" ]] || {
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  }

  for task in "${ALL_TASKS[@]}"; do
    output_dir="${RESULT_ROOT}/${output_name}/${task}"
    echo
    echo "=== model=${model} task=${task} ==="
    "${PYTHON}" -u evaluation/evaluate_language_act_suite.py \
      --checkpoint "${checkpoint}" \
      --episode-dir "${TASK_DIRS[$task]}" \
      --output "${output_dir}" \
      --execute-actions "${EXECUTE_ACTIONS}" \
      --max-actions "${MAX_ACTIONS}" \
      --device "${DEVICE}"
  done
done

echo
echo "All ACT evaluations completed. Results: ${RESULT_ROOT}"
