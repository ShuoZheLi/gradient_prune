from pathlib import Path

from config import ExperimentConfig
from experiment_runner import (
    _enumerate_conditions,
    _select_condition_shard,
    _write_condition_result,
    merge_condition_results,
)


def test_condition_sharding_round_robin():
    config = ExperimentConfig()
    config.methods = ["wanda"]
    config.pruning.sparsities = [0.0, 0.1, 0.2, 0.3]

    conditions = _enumerate_conditions(config, config.methods)

    assert _select_condition_shard(conditions, condition_shard_id=0, num_condition_shards=2) == [
        ("wanda", 0.0, None),
        ("wanda", 0.2, None),
    ]
    assert _select_condition_shard(conditions, condition_shard_id=1, num_condition_shards=2) == [
        ("wanda", 0.1, None),
        ("wanda", 0.3, None),
    ]


def test_merge_condition_results_orders_and_fills_accuracy_drop(tmp_path: Path):
    config = ExperimentConfig()
    config.methods = ["wanda"]
    config.pruning.sparsities = [0.0, 0.1]

    results_dir = tmp_path / "tables"
    dense_like = {
        "model_name": "m",
        "calibration_type": "prompt_response",
        "calibration_path": "cal",
        "loss_on": "response_only",
        "method": "wanda",
        "sparsity": 0.0,
        "lambda_value": None,
        "calibration_ce": None,
        "heldout_ce": None,
        "wikitext_ppl": None,
        "task_accuracy": 0.8,
        "accuracy_drop": None,
        "generalization_gap": None,
        "num_pruned_weights": 0,
        "num_total_prunable_weights": 10,
        "actual_sparsity": 0.0,
        "seed": 42,
        "notes": "",
    }
    sparse = dict(dense_like, sparsity=0.1, task_accuracy=0.5, actual_sparsity=0.1)

    _write_condition_result(sparse, results_dir)
    _write_condition_result(dense_like, results_dir)

    rows = merge_condition_results(results_dir, config)

    assert [row["sparsity"] for row in rows] == [0.0, 0.1]
    assert rows[0]["accuracy_drop"] == 0.0
    assert rows[1]["accuracy_drop"] == 0.30000000000000004
    assert (results_dir / "main_results.csv").is_file()
    assert (results_dir / "main_results.json").is_file()


def test_sharded_worker_writes_only_condition_result(monkeypatch, tmp_path: Path):
    import experiment_runner

    config = ExperimentConfig()
    config.methods = ["wanda"]
    config.pruning.sparsities = [0.0, 0.1]
    config.output.root_dir = str(tmp_path)
    config.output.save_plots = False

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  model_name_or_path: dummy\n"
        "methods: [wanda]\n"
        "pruning:\n"
        "  sparsities: [0.0, 0.1]\n"
        "  load_scores: true\n"
        f"  score_root: {tmp_path / 'scores'}\n"
        "output:\n"
        f"  root_dir: {tmp_path}\n"
        "  save_plots: false\n"
    )

    monkeypatch.setattr(experiment_runner, "load_model_and_tokenizer", lambda *args, **kwargs: (object(), object()))
    (tmp_path / "scores").mkdir()
    monkeypatch.setattr(experiment_runner, "_validate_pruning_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_runner, "_save_representative_scores", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment_runner,
        "_run_condition",
        lambda *args, **kwargs: {
            "model_name": "m",
            "calibration_type": "prompt_response",
            "calibration_path": "cal",
            "loss_on": "response_only",
            "method": "wanda",
            "sparsity": 0.0,
            "lambda_value": None,
            "calibration_ce": None,
            "heldout_ce": None,
            "wikitext_ppl": None,
            "task_accuracy": 0.8,
            "accuracy_drop": None,
            "generalization_gap": None,
            "num_pruned_weights": 0,
            "num_total_prunable_weights": 10,
            "actual_sparsity": 0.0,
            "seed": 42,
            "notes": "",
        },
    )

    rows = experiment_runner.run_experiment(config_path, condition_shard_id=0, num_condition_shards=2)

    assert len(rows) == 1
    assert (tmp_path / "tables" / "conditions").is_dir()
    assert not (tmp_path / "tables" / "main_results.csv").exists()
