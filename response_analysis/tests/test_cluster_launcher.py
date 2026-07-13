from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "response_analysis" / "scripts" / "qwen3_8b_wanda_response_analysis_multi_node.sh"


def run_launcher_dry(tmp_path: Path, extra_env: dict[str, str]) -> Path:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "WORK_DIR": str(REPO_ROOT),
            "RESULTS_BASE": str(tmp_path / "results"),
            "MODEL_PATH": "/data/shuozhe/saved_model/Qwen3-0.6B",
            "DATASET_PATH": "/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet",
            "RUN_NAME": "launcher_test",
            "RUN_TIMESTAMP": "fixed",
        }
    )
    env.update(extra_env)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, cwd=REPO_ROOT)
    subprocess.run(["bash", str(SCRIPT)], check=True, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    config_paths = list((tmp_path / "results").glob("response_analysis/launcher_test/runs/*/logs/config.env"))
    assert len(config_paths) == 1
    return config_paths[0]


def parse_config(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_pruning_sparsity_default_is_defined_once_for_model_id(tmp_path: Path):
    config = parse_config(run_launcher_dry(tmp_path, {"PRUNING_SPARSITY": "0.375"}))
    assert config["PRUNING_SPARSITY"] == "0.375"
    assert config["PRUNED_MODEL_ID"] == "qwen3_8b_wanda_s0.375"


def test_parallel_auto_disabled_without_slurm(tmp_path: Path):
    config = parse_config(run_launcher_dry(tmp_path, {"PARALLEL_GENERATION": "auto", "PARALLEL_ENTROPY": "auto"}))
    assert config["PARALLEL_GENERATION"] == "0"
    assert config["PARALLEL_ENTROPY"] == "0"


def test_parallel_auto_enabled_with_fake_multinode_slurm(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_scontrol = fake_bin / "scontrol"
    fake_scontrol.write_text("#!/bin/bash\nprintf 'node-a\\nnode-b\\n'\n", encoding="utf-8")
    fake_scontrol.chmod(0o755)
    config = parse_config(
        run_launcher_dry(
            tmp_path,
            {
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "SLURM_JOB_ID": "123",
                "SLURM_JOB_NODELIST": "node-[a-b]",
                "MAX_EXAMPLES": "8",
                "PARALLEL_GENERATION": "auto",
                "PARALLEL_ENTROPY": "auto",
            },
        )
    )
    assert config["PARALLEL_GENERATION"] == "1"
    assert config["PARALLEL_ENTROPY"] == "1"
    assert config["NODES"] == "node-a node-b"
