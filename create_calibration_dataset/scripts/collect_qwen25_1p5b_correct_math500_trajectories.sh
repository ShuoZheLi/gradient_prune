#!/bin/sh
set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-/data/shuozhe/saved_model/Qwen2.5-1.5B-Instruct}"
dataset_path="${DATASET_PATH:-/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
output_dir="${OUTPUT_DIR:-$repo_root/saved_calibration_dataset/qwen2.5-1.5b-instruct_math500_correct}"
raw_jsonl="${RAW_JSONL:-$output_dir/raw_actor_responses.jsonl}"
shard_dir="${SHARD_DIR:-$output_dir/shards}"
all_trajectories_jsonl="${ALL_TRAJECTORIES_JSONL:-$output_dir/all_actor_trajectories.jsonl}"
all_trajectories_parquet="${ALL_TRAJECTORIES_PARQUET:-$output_dir/all_actor_trajectories.parquet}"
correct_jsonl="${CORRECT_JSONL:-$output_dir/correct_actor_responses.jsonl}"
calib_parquet="${CALIB_PARQUET:-$output_dir/qwen2.5-1.5b-instruct_math500_correct.parquet}"
metrics_json="${METRICS_JSON:-$output_dir/metrics.json}"

max_examples="${MAX_EXAMPLES:-500}"
start_index="${START_INDEX:-0}"
seed="${SEED:-42}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_new_tokens="${MAX_NEW_TOKENS:-2048}"
generation_backend="${GENERATION_BACKEND:-vllm}"
batch_size="${BATCH_SIZE:-32}"
generation_max_batch_tokens="${GENERATION_MAX_BATCH_TOKENS:-0}"
response_log_max="${RESPONSE_LOG_MAX:--1}"
use_cache="${USE_CACHE:-0}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:-0}"
dtype="${DTYPE:-auto}"
devices="${DEVICES:-1 2 3}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.7}"
enforce_eager="${ENFORCE_EAGER:-1}"
response_key="${RESPONSE_KEY:-}"
reward_score_dir="${REWARD_SCORE_DIR:-}"
trust_remote_code="${TRUST_REMOTE_CODE:-0}"
skip_merge="${SKIP_MERGE:-1}"
progress_interval="${PROGRESS_INTERVAL:-5}"
dry_run="${DRY_RUN:-0}"

mkdir -p "$output_dir" "$shard_dir"

echo "[collect] model_path=$model_path"
echo "[collect] dataset_path=$dataset_path"
echo "[collect] raw_jsonl=$raw_jsonl"
echo "[collect] generation_backend=$generation_backend batch_size=$batch_size generation_max_batch_tokens=$generation_max_batch_tokens use_cache=$use_cache"
echo "[collect] vllm tensor_parallel_size=$tensor_parallel_size gpu_memory_utilization=$gpu_memory_utilization enforce_eager=$enforce_eager"

run_generation_shard() {
    shard_id="$1"
    shard_start="$2"
    shard_count="$3"
    shard_device="$4"
    shard_output="$5"

    if [ "$generation_backend" = "vllm" ]; then
        actor_dir="$model_path"
        if [ "$skip_merge" != "1" ]; then
            actor_dir=$("$python_bin" - "$model_path" <<'PYRESOLVE'
import sys
from create_calibration_dataset.generate_actor_responses_minimal import resolve_actor_hf_dir
print(resolve_actor_hf_dir(sys.argv[1], skip_merge=False))
PYRESOLVE
)
        fi
        shard_metrics="$shard_dir/metrics_shard_${shard_id}.json"
        set -- create_calibration_dataset/vllm_accuracy_runner.py \
            --model_path "$actor_dir" \
            --dataset_path "$dataset_path" \
            --output_path "$shard_output" \
            --metrics_path "$shard_metrics" \
            --prompt_key prompt \
            --start_index "$shard_start" \
            --max_examples "$shard_count" \
            --seed "$seed" \
            --max_prompt_length "$max_prompt_length" \
            --max_new_tokens "$max_new_tokens" \
            --batch_size "$batch_size" \
            --generation_max_batch_tokens "$generation_max_batch_tokens" \
            --response_log_max "$response_log_max" \
            --temperature "$temperature" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --tensor_parallel_size "$tensor_parallel_size" \
            --gpu_memory_utilization "$gpu_memory_utilization" \
            --dtype "$dtype"
        if [ "$enforce_eager" = "1" ]; then
            set -- "$@" --enforce_eager
        else
            set -- "$@" --no_enforce_eager
        fi
    elif [ "$generation_backend" = "transformers" ]; then
        set -- create_calibration_dataset/generate_actor_responses_minimal.py \
            --checkpoint_dir "$model_path" \
            --dataset_path "$dataset_path" \
            --output_path "$shard_output" \
            --prompt_key prompt \
            --start_index "$shard_start" \
            --max_examples "$shard_count" \
            --seed "$seed" \
            --max_prompt_length "$max_prompt_length" \
            --max_new_tokens "$max_new_tokens" \
            --batch_size "$batch_size" \
            --generation_max_batch_tokens "$generation_max_batch_tokens" \
            --response_log_max "$response_log_max" \
            --temperature "$temperature" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --dtype "${TRANSFORMERS_DTYPE:-bf16}" \
            --device cuda:0
        if [ "$use_cache" = "1" ]; then
            set -- "$@" --use_cache
        fi
        if [ "$trust_remote_code" = "1" ]; then
            set -- "$@" --trust_remote_code
        fi
        if [ "$skip_merge" = "1" ]; then
            set -- "$@" --skip_merge
        fi
    else
        echo "GENERATION_BACKEND must be 'vllm' or 'transformers'; got $generation_backend" >&2
        exit 2
    fi

    if [ -n "$response_key" ]; then
        set -- "$@" --response_key "$response_key"
    fi
    if [ -n "$reward_score_dir" ]; then
        set -- "$@" --reward_score_dir "$reward_score_dir"
    fi

    echo "[collect][shard $shard_id] backend=$generation_backend gpu=$shard_device start=$shard_start count=$shard_count output=$shard_output"
    if [ "$dry_run" = "1" ]; then
        printf '[collect][shard %s] command:' "$shard_id"
        printf ' %s' "CUDA_VISIBLE_DEVICES=$shard_device" "$python_bin" "$@"
        printf '\n'
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$shard_device" "$python_bin" "$@"
}

set -- $devices
num_shards=$#
if [ "$num_shards" -lt 1 ]; then
    echo "DEVICES must contain at least one GPU id." >&2
    exit 2
fi
if [ "$max_examples" -lt 0 ]; then
    echo "Parallel sharding requires MAX_EXAMPLES >= 0; got $max_examples" >&2
    exit 2
fi
if [ "$batch_size" -lt 1 ]; then
    echo "BATCH_SIZE must be >= 1; got $batch_size" >&2
    exit 2
fi

rm -f "$raw_jsonl" "$all_trajectories_jsonl" "$all_trajectories_parquet" "$shard_dir"/raw_actor_responses_shard_*.jsonl "$shard_dir"/metrics_shard_*.json "$shard_dir"/shard_*.log
base_count=$((max_examples / num_shards))
remainder=$((max_examples % num_shards))
shard_index=0
pids=""

for gpu_id in $devices; do
    shard_count=$base_count
    if [ "$shard_index" -lt "$remainder" ]; then
        shard_count=$((shard_count + 1))
    fi

    if [ "$shard_count" -gt 0 ]; then
        shard_start=$((start_index + shard_index * base_count + shard_index))
        if [ "$shard_index" -ge "$remainder" ]; then
            shard_start=$((start_index + remainder * (base_count + 1) + (shard_index - remainder) * base_count))
        fi
        shard_output="$shard_dir/raw_actor_responses_shard_${shard_index}.jsonl"
        shard_log="$shard_dir/shard_${shard_index}.log"
        if [ "$dry_run" = "1" ]; then
            run_generation_shard "$shard_index" "$shard_start" "$shard_count" "$gpu_id" "$shard_output"
        else
            run_generation_shard "$shard_index" "$shard_start" "$shard_count" "$gpu_id" "$shard_output" >"$shard_log" 2>&1 &
            pids="$pids $!"
        fi
    fi
    shard_index=$((shard_index + 1))
done

if [ "$dry_run" = "1" ]; then
    echo "[collect] dry run complete; no generation launched."
    exit 0
fi

progress_bar() {
    current="$1"
    total="$2"
    width=40
    if [ "$total" -le 0 ]; then
        percent=100
        filled=$width
    else
        percent=$((current * 100 / total))
        filled=$((current * width / total))
    fi
    empty=$((width - filled))
    bar=""
    while [ "${#bar}" -lt "$filled" ]; do bar="${bar}#"; done
    while [ "$empty" -gt 0 ]; do bar="${bar}-"; empty=$((empty - 1)); done
    printf '\r[collect] progress [%s] %s/%s (%s%%)' "$bar" "$current" "$total" "$percent"
}

progress_count() {
    completed=0
    for shard_output in "$shard_dir"/raw_actor_responses_shard_*.jsonl; do
        if [ -f "$shard_output" ]; then
            shard_lines=$(wc -l < "$shard_output" | tr -d ' ')
            completed=$((completed + shard_lines))
        fi
    done
    if [ "$completed" -gt "$max_examples" ]; then
        completed="$max_examples"
    fi
    printf '%s' "$completed"
}

progress_monitor() {
    while :; do
        progress_bar "$(progress_count)" "$max_examples"
        sleep "$progress_interval"
    done
}

progress_monitor &
progress_pid=$!

failed=0
for pid in $pids; do
    if ! wait "$pid"; then
        failed=1
    fi
done
kill "$progress_pid" 2>/dev/null || true
wait "$progress_pid" 2>/dev/null || true
progress_bar "$(progress_count)" "$max_examples"
printf '
'

if [ "$failed" -ne 0 ]; then
    echo "One or more generation shards failed. Logs:" >&2
    ls -1 "$shard_dir"/shard_*.log >&2 || true
    exit 1
fi

: > "$raw_jsonl"
shard_index=0
for gpu_id in $devices; do
    shard_output="$shard_dir/raw_actor_responses_shard_${shard_index}.jsonl"
    if [ -f "$shard_output" ]; then
        cat "$shard_output" >> "$raw_jsonl"
    fi
    shard_index=$((shard_index + 1))
done

raw_count=$(wc -l < "$raw_jsonl" | tr -d ' ')
if [ "$raw_count" -ne "$max_examples" ]; then
    echo "Merged raw response count mismatch: expected $max_examples, got $raw_count" >&2
    echo "Shard logs are in: $shard_dir" >&2
    exit 1
fi
echo "[collect] merged $raw_count responses from $num_shards shard(s)"

echo "[postprocess] filtering correct responses and writing calibration parquet"
"$python_bin" - \
    --model_path "$model_path" \
    --skip_merge "$skip_merge" \
    --trust_remote_code "$trust_remote_code" \
    --raw_jsonl "$raw_jsonl" \
    --all_trajectories_jsonl "$all_trajectories_jsonl" \
    --all_trajectories_parquet "$all_trajectories_parquet" \
    --correct_jsonl "$correct_jsonl" \
    --calib_parquet "$calib_parquet" \
    --metrics_json "$metrics_json" <<'PY'
import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from create_calibration_dataset.generate_actor_responses_minimal import resolve_actor_hf_dir

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--skip_merge", required=True)
parser.add_argument("--trust_remote_code", required=True)
parser.add_argument("--raw_jsonl", required=True)
parser.add_argument("--all_trajectories_jsonl", required=True)
parser.add_argument("--all_trajectories_parquet", required=True)
parser.add_argument("--correct_jsonl", required=True)
parser.add_argument("--calib_parquet", required=True)
parser.add_argument("--metrics_json", required=True)
args = parser.parse_args()

raw_path = Path(args.raw_jsonl).expanduser()
all_jsonl_path = Path(args.all_trajectories_jsonl).expanduser()
all_parquet_path = Path(args.all_trajectories_parquet).expanduser()
correct_path = Path(args.correct_jsonl).expanduser()
parquet_path = Path(args.calib_parquet).expanduser()
metrics_path = Path(args.metrics_json).expanduser()
all_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
all_parquet_path.parent.mkdir(parents=True, exist_ok=True)
correct_path.parent.mkdir(parents=True, exist_ok=True)
parquet_path.parent.mkdir(parents=True, exist_ok=True)
metrics_path.parent.mkdir(parents=True, exist_ok=True)

actor_dir = resolve_actor_hf_dir(args.model_path, skip_merge=args.skip_merge.lower() in {"1", "true", "yes"})
tokenizer = AutoTokenizer.from_pretrained(
    actor_dir,
    trust_remote_code=args.trust_remote_code.lower() in {"1", "true", "yes"},
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

all_rows = []
correct_rows = []
num_total = 0
num_scored = 0
with raw_path.open("r", encoding="utf-8") as input_file, all_jsonl_path.open("w", encoding="utf-8") as all_file, correct_path.open("w", encoding="utf-8") as correct_file:
    for line in input_file:
        if not line.strip():
            continue
        num_total += 1
        row = json.loads(line)
        if "is_correct" in row:
            num_scored += 1

        prompt = row.get("prompt", "")
        response = row.get("response", "")
        trajectory = f"{prompt}{response}"
        trajectory_ids = tokenizer(
            trajectory,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        out_row = {
            "example_id": row.get("example_id"),
            "prompt": prompt,
            "response": response,
            "task_score": row.get("task_score"),
            "is_correct": bool(row.get("is_correct", False)),
            "prompt_generated_trajectory": trajectory,
            "prompt_generated_trajectory_ids": trajectory_ids,
        }
        all_rows.append(out_row)
        all_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        if out_row["is_correct"]:
            correct_rows.append(out_row)
            correct_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")

all_df = pd.DataFrame(all_rows)
all_df.to_parquet(all_parquet_path, index=False)
correct_df = pd.DataFrame(correct_rows)
correct_df.to_parquet(parquet_path, index=False)
metrics = {
    "num_total": num_total,
    "num_scored": num_scored,
    "num_correct": len(correct_rows),
    "accuracy": (len(correct_rows) / num_scored) if num_scored else None,
    "raw_jsonl": str(raw_path),
    "all_trajectories_jsonl": str(all_jsonl_path),
    "all_trajectories_parquet": str(all_parquet_path),
    "correct_jsonl": str(correct_path),
    "calib_parquet": str(parquet_path),
}
metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2))
if not correct_rows:
    raise SystemExit("No correct trajectories were collected; try sampling or inspect raw responses.")
PY

echo "[done] all trajectories parquet: $all_trajectories_parquet"
echo "[done] correct trajectories parquet: $calib_parquet"
echo "[done] use with PUNE as calib_data=$calib_parquet"
