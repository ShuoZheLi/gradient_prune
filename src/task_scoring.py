from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in normal eval envs
    np = None

MATH_DATA_SOURCES = {"lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500", "math_500"}
MATH_DAPO_DATA_SOURCES = {"math_dapo", "math", "math_dapo_reasoning"}
VERL_SCORER_BACKENDS = {"verl_default", "verl_math_reward", "verl_math_verify", "legacy_modules", "fallback"}


@dataclass(frozen=True)
class TaskExample:
    example_id: int
    prompt: str
    data_source: str
    ground_truth: Any

    @property
    def prompt_text(self) -> str:
        return self.prompt


def is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False
    bool_types = (bool,)
    if np is not None:
        bool_types = (bool, np.bool_)
    return bool(result) if isinstance(result, bool_types) else False


def normalize_enable_thinking(value: Any) -> str:
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    aliases = {"1": "true", "yes": "true", "y": "true", "on": "true", "0": "false", "no": "false", "n": "false", "off": "false", "none": "auto", "default": "auto"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "true", "false"}:
        raise ValueError(f"enable_thinking must be one of auto, true, false; got {value!r}")
    return normalized


def apply_chat_template_with_optional_thinking(tokenizer, messages, *, enable_thinking: str = "auto", add_generation_prompt: bool = True) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    thinking_mode = normalize_enable_thinking(enable_thinking)
    if thinking_mode != "auto":
        kwargs["enable_thinking"] = thinking_mode == "true"
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)



def looks_like_rendered_chat_prompt(text: str) -> bool:
    return any(marker in text for marker in ("<|im_start|>", "<|start_header_id|>", "[INST]", "<s>[INST]"))

def normalize_prompt(prompt: Any, tokenizer=None, *, enable_thinking: str = "auto") -> str:
    if isinstance(prompt, str):
        stripped = prompt.strip()
        if stripped.startswith(("[", "{")):
            try:
                parsed = ast.literal_eval(prompt)
            except (SyntaxError, ValueError):
                parsed = None
            if parsed is not None:
                return normalize_prompt(parsed, tokenizer, enable_thinking=enable_thinking)
        if normalize_enable_thinking(enable_thinking) != "auto" and tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and not looks_like_rendered_chat_prompt(prompt):
            try:
                return apply_chat_template_with_optional_thinking(tokenizer, [{"role": "user", "content": prompt}], enable_thinking=enable_thinking)
            except Exception:
                pass
    if hasattr(prompt, "tolist"):
        try:
            return normalize_prompt(prompt.tolist(), tokenizer, enable_thinking=enable_thinking)
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(prompt, dict):
        if "messages" in prompt:
            return normalize_prompt(prompt["messages"], tokenizer, enable_thinking=enable_thinking)
        for key in ("prompt", "text", "content"):
            if key in prompt:
                return str(prompt[key])
        return json.dumps(prompt, ensure_ascii=False, default=str)
    if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes, bytearray)):
        values = list(prompt)
        if not values:
            return ""
        if all(isinstance(item, dict) for item in values):
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                try:
                    return apply_chat_template_with_optional_thinking(tokenizer, values, enable_thinking=enable_thinking)
                except Exception:
                    pass
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in values)
        if all(isinstance(item, str) for item in values):
            return "\n".join(values)
        return "\n".join(str(item) for item in values)
    return str(prompt)


def extract_prompt_value(row, prompt_key: str) -> Any:
    for key in (prompt_key, "prompt", "query", "question", "problem", "original_question", "messages"):
        if key in row and not is_missing(row[key]):
            return row[key]
    available = list(row.index) if hasattr(row, "index") else []
    raise KeyError(f"Cannot find prompt column. Requested {prompt_key!r}; available columns: {available}")


def extract_prompt(row, prompt_key: str, tokenizer=None, *, enable_thinking: str = "auto") -> str:
    return normalize_prompt(extract_prompt_value(row, prompt_key), tokenizer, enable_thinking=enable_thinking)


def extract_data_source(row, dataset_path: str | Path | None = None) -> str:
    for key in ("data_source", "source", "dataset", "dataset_source"):
        if key in row and not is_missing(row[key]):
            return str(row[key])
    if dataset_path is not None:
        path_str = str(dataset_path)
        path_name = Path(path_str).name.lower()
        path_parts = {part.lower() for part in Path(path_str).parts}
        if path_str in {"ShuoZheLi/MetaMathQA-math-500", "MetaMathQA-math-500", "metamathqa_math_500", "math_500"}:
            return "math_500"
        if "metamathqa-math-500" in path_parts and path_name in {"test.parquet", "math7500.parquet"}:
            return "math_500"
    return ""


def _extract_nested(value: Any, keys: Sequence[str]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def extract_ground_truth(row, response_key: str | None = None) -> Any:
    if response_key and response_key in row and not is_missing(row[response_key]):
        return row[response_key]

    for key in ("ground_truth", "answer", "solution", "response", "target"):
        if key in row and not is_missing(row[key]):
            return row[key]

    reward_model = row.get("reward_model") if hasattr(row, "get") else None
    ground_truth = _extract_nested(reward_model, ("ground_truth",))
    if ground_truth is not None and not is_missing(ground_truth):
        return ground_truth

    extra_info = row.get("extra_info") if hasattr(row, "get") else None
    for key in ("answer", "ground_truth", "solution", "target"):
        ground_truth = _extract_nested(extra_info, (key,))
        if ground_truth is not None and not is_missing(ground_truth):
            return ground_truth
    return None


def dataframe_to_task_examples(df, prompt_key: str, response_key: str | None = None, tokenizer=None, dataset_path: str | Path | None = None) -> list[TaskExample]:
    examples: list[TaskExample] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        example_id = int(row["example_id"] if "example_id" in row and not is_missing(row["example_id"]) else row["id"] if "id" in row and not is_missing(row["id"]) else idx)
        examples.append(
            TaskExample(
                example_id=example_id,
                prompt=extract_prompt(row, prompt_key, tokenizer),
                data_source=extract_data_source(row, dataset_path),
                ground_truth=extract_ground_truth(row, response_key),
            )
        )
    return examples


def task_example_to_dict(example: TaskExample) -> dict[str, Any]:
    return {"example_id": example.example_id, "prompt": example.prompt, "data_source": example.data_source, "ground_truth": example.ground_truth}


def _last_braced_content(text: str, command: str) -> str | None:
    command_name = command.lstrip("\\")
    starts = [match.end() for match in re.finditer(r"\\+" + re.escape(command_name) + r"\{", text)]
    if not starts:
        return None
    result = None
    for start in starts:
        depth = 1
        chars = []
        index = start
        while index < len(text):
            char = text[index]
            if char == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
                chars.append(char)
            elif char == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    result = "".join(chars)
                    break
                chars.append(char)
            else:
                chars.append(char)
            index += 1
    return result


def extract_math_answer(text: Any) -> str | None:
    if text is None:
        return None
    text = str(text)
    boxed = _last_braced_content(text, "\\boxed")
    if boxed is not None:
        return normalize_math_answer(boxed, extract_answer=False)
    hashes = re.findall(r"####\s*([^\n]+)", text)
    if hashes:
        return normalize_math_answer(hashes[-1], extract_answer=False)
    last_line = text.splitlines()[-1] if text else ""
    prefixed = re.findall(r"(?:final answer|answer is|answer:)\s*(.+)$", last_line, flags=re.IGNORECASE)
    candidate = prefixed[-1] if prefixed else last_line
    search_text = candidate if prefixed else text
    if any(token in candidate for token in ("\\in", "[", "]", "(", ")", "\\cup", "\\infty", "∞")):
        return normalize_math_answer(candidate, extract_answer=False)
    if re.search(r"\\+(?:d?frac|sqrt|pm|cdot|times|div|leq?|geq?|neq?|approx)", candidate) or any(token in candidate for token in ("^", "_")):
        return normalize_math_answer(candidate, extract_answer=False)
    numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.?\d+(?:/\d+)?)", search_text)
    numbers = [number.replace(",", "") for number in numbers]
    if numbers:
        return normalize_math_answer(numbers[-1], extract_answer=False)
    return normalize_math_answer(last_line, extract_answer=False)


def normalize_math_answer(answer: Any, *, extract_answer: bool = True) -> str | None:
    if answer is None:
        return None
    if extract_answer:
        extracted = extract_math_answer(answer)
        if extracted is None:
            return None
        answer = extracted
    answer = str(answer).strip().strip("$").strip()
    answer = re.sub(r"\\+left", "", answer)
    answer = re.sub(r"\\+right", "", answer)
    answer = answer.replace("\\(", "(").replace("\\)", ")").replace("\\[", "[").replace("\\]", "]")
    answer = re.sub(r"\\+[!,;]", "", answer)
    answer = re.sub(r"\\+(?:text|mathrm)\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = answer.strip(".。")
    return answer.lower()


def fallback_math_score(response_text: str, ground_truth: Any) -> float:
    prediction = extract_math_answer(response_text)
    target = extract_math_answer(ground_truth)
    return float(bool(prediction) and target is not None and prediction == target)


def score_math_response(response_text: str, ground_truth: Any) -> tuple[float, bool]:
    score = fallback_math_score(response_text, ground_truth)
    return score, bool(score == 1.0)


def reward_module_path(module_name: str, reward_score_dir: str | Path | None = None) -> Path:
    if reward_score_dir is not None:
        return Path(reward_score_dir).expanduser() / f"{module_name}.py"
    if os.environ.get("VERL_REWARD_SCORE_DIR"):
        return Path(os.environ["VERL_REWARD_SCORE_DIR"]).expanduser() / f"{module_name}.py"
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "verl" / "utils" / "reward_score" / f"{module_name}.py",
        here.parents[2] / "verl" / "utils" / "reward_score" / f"{module_name}.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def load_reward_module(module_name: str, reward_score_dir: str | Path | None = None):
    module_path = reward_module_path(module_name, reward_score_dir)
    if not module_path.is_file():
        raise FileNotFoundError(f"Reward module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"_task_scoring_reward_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scorer_backend() -> str:
    backend = os.environ.get("TASK_SCORER_BACKEND") or os.environ.get("MATH_SCORER") or "verl_math_reward"
    backend = backend.strip().lower()
    if backend == "verl_default":
        backend = "verl_math_reward"
    if backend not in VERL_SCORER_BACKENDS:
        choices = ", ".join(sorted(VERL_SCORER_BACKENDS))
        raise ValueError(f"Unsupported scorer backend {backend!r}. Choose one of: {choices}")
    return backend


def _import_verl_default_compute_score():
    try:
        module = importlib.import_module("verl.utils.reward_score")
    except ImportError as exc:
        raise ImportError(
            "Unable to import verl.utils.reward_score. Activate the verl environment before collecting/scoring, "
            "for example: source /data/shuozhe/miniconda3/etc/profile.d/conda.sh && conda activate verl. "
            "Set TASK_SCORER_BACKEND=legacy_modules or fallback only if you intentionally do not want verl scoring."
        ) from exc
    return module.default_compute_score


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _compute_score_with_verl_default(data_source: str, response_text: str, ground_truth: Any) -> Any:
    return _import_verl_default_compute_score()(
        data_source,
        response_text,
        ground_truth,
        math_dapo_binary_reward=_env_flag("MATH_DAPO_BINARY_REWARD", default=True),
    )


def _compute_score_with_verl_math_verify(response_text: str, ground_truth: Any) -> Any:
    try:
        module = importlib.import_module("verl.utils.reward_score.math_verify")
    except ImportError as exc:
        raise ImportError(
            "Unable to import verl.utils.reward_score.math_verify. Activate the verl environment and ensure "
            "math-verify is installed, or use TASK_SCORER_BACKEND=verl_default."
        ) from exc
    return module.compute_score(model_output=response_text, ground_truth=str(ground_truth))


def scalarize_score(score: Any) -> float:
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)


def compute_score_with_legacy_reward_module(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> Any:
    if data_source == "openai/gsm8k":
        return load_reward_module("gsm8k", reward_score_dir).compute_score(response_text, ground_truth)
    if data_source in MATH_DATA_SOURCES:
        try:
            return load_reward_module("math_reward", reward_score_dir).compute_score(response_text, ground_truth)
        except FileNotFoundError:
            return fallback_math_score(response_text, ground_truth)
    if data_source in MATH_DAPO_DATA_SOURCES or data_source.startswith("aime"):
        try:
            return load_reward_module("math_dapo", reward_score_dir).compute_score(response_text, ground_truth, incorrect_reward=0.0)
        except FileNotFoundError:
            return fallback_math_score(response_text, ground_truth)
    return fallback_math_score(response_text, ground_truth)


def compute_score_with_reward_module(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> Any:
    if reward_score_dir is not None:
        return compute_score_with_legacy_reward_module(data_source, response_text, ground_truth, reward_score_dir=reward_score_dir)

    backend = scorer_backend()
    if backend == "verl_math_reward":
        return _compute_score_with_verl_default(data_source, response_text, ground_truth)
    if backend == "verl_math_verify":
        if data_source in MATH_DATA_SOURCES or data_source in MATH_DAPO_DATA_SOURCES or data_source.startswith("aime"):
            return _compute_score_with_verl_math_verify(response_text, ground_truth)
        return _compute_score_with_verl_default(data_source, response_text, ground_truth)
    if backend == "legacy_modules":
        return compute_score_with_legacy_reward_module(data_source, response_text, ground_truth)
    return fallback_math_score(response_text, ground_truth)


def score_response(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> float:
    if ground_truth is None:
        return float("nan")
    return scalarize_score(compute_score_with_reward_module(data_source, response_text, ground_truth, reward_score_dir=reward_score_dir))


def score_task_response(response_text: str, ground_truth: Any, data_source: str = "", reward_score_dir: str | Path | None = None) -> tuple[float, bool]:
    score = score_response(data_source, response_text, ground_truth, reward_score_dir=reward_score_dir)
    return score, bool(score == 1.0)
