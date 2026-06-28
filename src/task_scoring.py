from __future__ import annotations

import ast
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


def normalize_prompt(prompt: Any, tokenizer=None) -> str:
    if isinstance(prompt, str) and prompt.strip().startswith(("[", "{")):
        try:
            parsed = ast.literal_eval(prompt)
        except (SyntaxError, ValueError):
            parsed = None
        if parsed is not None:
            return normalize_prompt(parsed, tokenizer)
    if hasattr(prompt, "tolist"):
        try:
            return normalize_prompt(prompt.tolist(), tokenizer)
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(prompt, dict):
        if "messages" in prompt:
            return normalize_prompt(prompt["messages"], tokenizer)
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
                    return tokenizer.apply_chat_template(values, tokenize=False, add_generation_prompt=True)
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


def extract_prompt(row, prompt_key: str, tokenizer=None) -> str:
    return normalize_prompt(extract_prompt_value(row, prompt_key), tokenizer)


def extract_data_source(row, dataset_path: str | Path | None = None) -> str:
    for key in ("data_source", "source", "dataset", "dataset_source"):
        if key in row and not is_missing(row[key]):
            return str(row[key])
    if dataset_path is not None and str(dataset_path) in {"ShuoZheLi/MetaMathQA-math-500", "MetaMathQA-math-500", "metamathqa_math_500", "math_500"}:
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


def scalarize_score(score: Any) -> float:
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)


def compute_score_with_reward_module(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> Any:
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


def score_response(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> float:
    if ground_truth is None:
        return float("nan")
    return scalarize_score(compute_score_with_reward_module(data_source, response_text, ground_truth, reward_score_dir=reward_score_dir))


def score_task_response(response_text: str, ground_truth: Any, data_source: str = "", reward_score_dir: str | Path | None = None) -> tuple[float, bool]:
    score = score_response(data_source, response_text, ground_truth, reward_score_dir=reward_score_dir)
    return score, bool(score == 1.0)
