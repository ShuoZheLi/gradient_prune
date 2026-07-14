source /data/shuozhe/miniconda3/etc/profile.d/conda.sh && conda activate verl
export PYTHONPATH=/data/shuozhe/gradient_prune:$PYTHONPATH
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.portkey.ai/v1}"
export OPENAI_EVALUATOR_MODEL="${OPENAI_EVALUATOR_MODEL:-@caee-zw7893-ope-815f99/gpt-5-mini}"
NO_API="${NO_API:-1}"

judge_args=()
if [[ "$NO_API" == "1" ]]; then
  judge_args+=(--skip_semantic_judge)
fi

python -m response_analysis.process_existing_results \
  --input_dir /data/shuozhe/gradient_prune/results/07_13_2026 \
  --output_dir /data/shuozhe/gradient_prune/results/07_13_2026/processed \
  --surface_workers 32 \
  --judge_workers 32 \
  --request_timeout 180 \
  --max_retries 5 \
  "${judge_args[@]}"
