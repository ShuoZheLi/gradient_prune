from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "bf16"
    device: str = "cuda:0"
    trust_remote_code: bool = False


@dataclass
class PruningConfig:
    prune_ops: list[str] | None = None
    sparsities: list[float] = field(default_factory=lambda: [0.0, 0.5])
    granularity: str = "rowwise"
    save_pruned_models: bool = False
    load_masks: bool = False
    mask_root: str | None = None


@dataclass
class HybridConfig:
    lambda_values: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])


@dataclass
class CalibrationConfig:
    type: str = "prompt_response"
    path: str | None = None
    only_correct: bool = False
    loss_on: str = "response_only"
    max_samples: int | None = None
    microbatch_size: int = 1
    fisher_estimator: str = "microbatch"
    max_length: int = 4096
    text_key: str | None = None
    prompt_key: str = "prompt"
    response_key: str | None = "response"
    shuffle: bool = False


@dataclass
class CalibrationCEConfig:
    enabled: bool = True
    backend: str = "transformers"
    path: str | None = None
    type: str | None = None
    only_correct: bool | None = None
    loss_on: str | None = None
    max_samples: int | None = None
    batch_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    enforce_eager: bool = True
    trust_remote_code: bool = False
    max_length: int | None = None
    text_key: str | None = None
    prompt_key: str | None = None
    response_key: str | None = None


@dataclass
class HeldoutCEConfig:
    enabled: bool = True
    backend: str = "transformers"
    path: str | None = None
    loss_on: str = "response_only"
    max_samples: int | None = None
    batch_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    enforce_eager: bool = True
    trust_remote_code: bool = False
    max_length: int = 4096
    text_key: str | None = None
    prompt_key: str = "prompt"
    response_key: str | None = "response"


@dataclass
class TextPPLConfig:
    enabled: bool = True
    backend: str = "transformers"
    dataset_name: str = "wikitext"
    dataset_config: str | None = "wikitext-2-raw-v1"
    split: str = "validation"
    text_key: str = "text"
    max_samples: int | None = 256
    batch_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    enforce_eager: bool = True
    trust_remote_code: bool = False
    max_length: int = 2048


@dataclass
class TaskAccuracyConfig:
    enabled: bool = False
    dataset_path: str | None = None
    backend: str = "transformers"
    max_examples: int | None = None
    prompt_key: str = "prompt"
    response_key: str | None = None
    reward_score_dir: str | None = None
    max_prompt_length: int = 2048
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    batch_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    enforce_eager: bool = True
    trust_remote_code: bool = False


@dataclass
class OutputConfig:
    root_dir: str = "results/debug"
    save_stats: bool = True
    save_masks: bool = True
    save_plots: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "debug"
    seed: int = 42
    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_name_or_path=""))
    pruning: PruningConfig = field(default_factory=PruningConfig)
    methods: list[str] = field(default_factory=lambda: ["dense", "magnitude", "signed_taylor"])
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    calibration_ce: CalibrationCEConfig = field(default_factory=CalibrationCEConfig)
    heldout_ce: HeldoutCEConfig = field(default_factory=HeldoutCEConfig)
    text_ppl: TextPPLConfig = field(default_factory=TextPPLConfig)
    task_accuracy: TaskAccuracyConfig = field(default_factory=TaskAccuracyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _construct_dataclass(cls: type, data: dict[str, Any]):
    kwargs = {}
    field_map = {f.name: f for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in field_map:
            kwargs[key] = value
            continue
        field_info = field_map[key]
        default_obj = None
        if field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            default_obj = field_info.default_factory()  # type: ignore[misc]
        elif field_info.default is not MISSING:
            default_obj = field_info.default
        if default_obj is not None and is_dataclass(default_obj) and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(type(default_obj), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "text_ppl" not in raw and "wikitext" in raw:
        raw["text_ppl"] = raw.pop("wikitext")
    return _construct_dataclass(ExperimentConfig, raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_plain(config), handle, sort_keys=False)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_plain(value) for key, value in obj.items()}
    return obj
