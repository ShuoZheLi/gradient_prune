#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/shuozhe/gradient_prune"
MODEL_PATH="/data/shuozhe/saved_model/Qwen3-8B-Base"
CALIBRATION_PATH="$REPO_ROOT/saved_calibration_dataset/qwen2.5-1.5b-instruct_math7500_correct"
OUTPUT_DIR="$REPO_ROOT/results/06_30_2026/qwen3_8b_base_wanda_scores"
LOG_DIR="$REPO_ROOT/exp_track/06_30_2026/logs"
LOG_FILE="$LOG_DIR/qwen3_8b_base_wanda_scores.log"
PID_FILE="$LOG_DIR/qwen3_8b_base_wanda_scores.pid"
WORKER="$REPO_ROOT/exp_track/06_30_2026/worker_qwen3_8b_base_wanda_scores.sh"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
DTYPE="${DTYPE:-bf16}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ -f "$PID_FILE" ]]; then
  old_pids="$(tr '\n' ' ' < "$PID_FILE")"
  for old_pid in $old_pids; do
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Existing WANDA scoring job is still running with PID $old_pid" >&2
      exit 1
    fi
  done
fi

existing_workers="$(pgrep -f "scripts/score_wanda.py --model $MODEL_PATH --calibration $CALIBRATION_PATH --output-dir $OUTPUT_DIR" || true)"
if [[ -n "$existing_workers" ]]; then
  echo "Existing WANDA scoring workers are still running:" >&2
  echo "$existing_workers" >&2
  printf '%s\n' $existing_workers > "$PID_FILE"
  exit 1
fi

{
  echo "[$(date -Is)] Starting Qwen3-8B-Base WANDA scoring"
  echo "Repo: $REPO_ROOT"
  echo "Model: $MODEL_PATH"
  echo "Calibration: $CALIBRATION_PATH"
  echo "Output: $OUTPUT_DIR"
  echo "NUM_GPUS=$NUM_GPUS MAX_LENGTH=$MAX_LENGTH MICROBATCH_SIZE=$MICROBATCH_SIZE DTYPE=$DTYPE"
  echo "Worker: $WORKER"
} >> "$LOG_FILE"

NUM_GPUS="$NUM_GPUS" MAX_LENGTH="$MAX_LENGTH" MICROBATCH_SIZE="$MICROBATCH_SIZE" DTYPE="$DTYPE" \
  nohup bash "$WORKER" >> "$LOG_FILE" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "Started WANDA scoring in background: PID $pid"
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT_DIR"
