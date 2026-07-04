import json

import torch
import torch.nn as nn

from config import CalibrationConfig, ExperimentConfig, ModelConfig, PruningConfig
from apply_pruning import build_masks_for_model
from experiment_runner import _build_masks_from_saved_scores, _save_representative_scores
from pruning_scores import gradient_norm_score, hybrid_wanda_signed_taylor_score, magnitude_score, signed_first_order_score, signed_taylor_score, wanda_score


def test_signed_taylor_formula_exact_values():
    w = torch.tensor([1.0, -2.0])
    g = torch.tensor([0.5, -0.25])
    h = torch.tensor([2.0, 0.5])
    expected = -g * w + 0.5 * h * w.pow(2)
    actual = signed_taylor_score(w, g, h)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([0.5, 0.5]))


def test_gradient_norm_and_wanda_broadcasting():
    w = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    h = torch.tensor([[4.0, 9.0, 16.0], [1.0, 0.25, 0.0]])
    activation_norm = torch.tensor([10.0, 100.0, 1000.0])
    torch.testing.assert_close(gradient_norm_score(w, h), w.abs() * h.sqrt())
    torch.testing.assert_close(wanda_score(w, activation_norm), w.abs() * activation_norm.view(1, -1))


def test_hybrid_keeps_negative_signed_taylor_term():
    w = torch.tensor([[1.0]])
    g = torch.tensor([[2.0]])
    h = torch.tensor([[0.0]])
    activation_norm = torch.tensor([0.1])
    score = hybrid_wanda_signed_taylor_score(w, activation_norm, g, h, lambda_value=1.0)
    torch.testing.assert_close(score, torch.tensor([[-1.9]]))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.self_attn.q_proj.weight.copy_(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))


class TinyModel:
    def __init__(self):
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock()])


def test_runner_saves_wanda_scores_in_standalone_format(tmp_path):
    model = TinyModel()
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    activation_stats = {"model.layers.0.self_attn.q_proj": torch.tensor([10.0, 100.0, 1000.0])}

    config = ExperimentConfig(
        seed=123,
        model=ModelConfig(model_name_or_path="tiny-model", dtype="bf16", device="cuda:0", trust_remote_code=True),
        pruning=PruningConfig(prune_ops=["q_proj"]),
        methods=["wanda"],
        calibration=CalibrationConfig(
            type="prompt_response",
            path="tiny-calibration",
            only_correct=True,
            max_samples=7,
            microbatch_size=2,
            max_length=128,
            shuffle=True,
        ),
    )

    _save_representative_scores(model, gradient_stats=None, activation_stats=activation_stats, scores_dir=scores_dir, config=config)

    score_path = scores_dir / "model__layers__0__self_attn__q_proj.pt"
    entry = torch.load(score_path, map_location="cpu", weights_only=False)
    assert set(entry) == {"wanda"}
    torch.testing.assert_close(entry["wanda"], model.model.layers[0].self_attn.q_proj.weight.abs() * activation_stats["model.layers.0.self_attn.q_proj"].view(1, -1))

    run_args = json.loads((scores_dir / "run_args.json").read_text())
    assert run_args["model"] == "tiny-model"
    assert run_args["calibration"] == "tiny-calibration"
    assert run_args["output_dir"] == str(scores_dir)
    assert run_args["prune_ops"] == ["q_proj"]
    assert run_args["enable_thinking"] == "auto"

    metadata = json.loads((scores_dir / "metadata.json").read_text())
    assert metadata["score_key"] == "wanda"
    assert "score_keys" not in metadata
    assert metadata["modules"] == {"model.layers.0.self_attn.q_proj": "model__layers__0__self_attn__q_proj.pt"}
    assert metadata["summaries"]["model.layers.0.self_attn.q_proj"]["shape"] == [2, 3]
    assert metadata["num_modules"] == 1
    assert metadata["num_total_scores"] == 6

    masks = _build_masks_from_saved_scores(model, method="wanda", sparsity=1 / 3, scores_dir=scores_dir, prune_ops=["q_proj"], granularity="rowwise")
    assert set(masks) == {"model.layers.0.self_attn.q_proj"}
    assert masks["model.layers.0.self_attn.q_proj"].shape == model.model.layers[0].self_attn.q_proj.weight.shape


def test_runner_saves_combined_scores_for_requested_methods(tmp_path):
    model = TinyModel()
    scores_dir = tmp_path / "scores"
    weight = model.model.layers[0].self_attn.q_proj.weight.detach().cpu()
    gradient_stats = {
        "model.layers.0.self_attn.q_proj": {
            "g": torch.tensor([[0.5, -0.25, 0.125], [1.0, -1.5, 2.0]]),
            "h": torch.tensor([[4.0, 9.0, 16.0], [1.0, 0.25, 0.0]]),
            "abs_g": torch.tensor([[0.5, 0.25, 0.125], [1.0, 1.5, 2.0]]),
        }
    }
    activation_stats = {"model.layers.0.self_attn.q_proj": torch.tensor([10.0, 100.0, 1000.0])}
    config = ExperimentConfig(
        model=ModelConfig(model_name_or_path="tiny-model"),
        pruning=PruningConfig(prune_ops=["q_proj"]),
        methods=["magnitude", "gradient_norm", "signed_first_order", "signed_taylor", "hybrid_wanda_signed_taylor"],
    )
    config.hybrid.lambda_values = [0.1, 1.0]

    _save_representative_scores(model, gradient_stats=gradient_stats, activation_stats=activation_stats, scores_dir=scores_dir, config=config)

    entry = torch.load(scores_dir / "model__layers__0__self_attn__q_proj.pt", map_location="cpu", weights_only=False)
    expected_keys = {
        "magnitude",
        "gradient_norm",
        "signed_first_order",
        "signed_taylor",
        "hybrid_wanda_signed_taylor__lambda=0.1",
        "hybrid_wanda_signed_taylor__lambda=1.0",
        "wanda",
    }
    assert set(entry) == expected_keys
    torch.testing.assert_close(entry["magnitude"], magnitude_score(weight))
    torch.testing.assert_close(entry["gradient_norm"], gradient_norm_score(weight, gradient_stats["model.layers.0.self_attn.q_proj"]["h"]))
    torch.testing.assert_close(entry["signed_first_order"], signed_first_order_score(weight, gradient_stats["model.layers.0.self_attn.q_proj"]["g"]))
    torch.testing.assert_close(entry["signed_taylor"], signed_taylor_score(weight, gradient_stats["model.layers.0.self_attn.q_proj"]["g"], gradient_stats["model.layers.0.self_attn.q_proj"]["h"]))
    torch.testing.assert_close(entry["wanda"], wanda_score(weight, activation_stats["model.layers.0.self_attn.q_proj"]))
    torch.testing.assert_close(entry["hybrid_wanda_signed_taylor__lambda=0.1"], hybrid_wanda_signed_taylor_score(weight, activation_stats["model.layers.0.self_attn.q_proj"], gradient_stats["model.layers.0.self_attn.q_proj"]["g"], gradient_stats["model.layers.0.self_attn.q_proj"]["h"], 0.1))

    metadata = json.loads((scores_dir / "metadata.json").read_text())
    assert metadata["score_key"] is None
    assert metadata["score_keys"] == list(entry.keys())
    assert metadata["num_total_scores_by_key"]["hybrid_wanda_signed_taylor__lambda=0.1"] == 6

    direct_masks = build_masks_for_model(
        model,
        method="hybrid_wanda_signed_taylor",
        sparsity=1 / 3,
        prune_ops=["q_proj"],
        gradient_stats=gradient_stats,
        activation_stats=activation_stats,
        lambda_value=0.1,
        granularity="rowwise",
    )
    saved_masks = _build_masks_from_saved_scores(
        model,
        method="hybrid_wanda_signed_taylor",
        sparsity=1 / 3,
        scores_dir=scores_dir,
        prune_ops=["q_proj"],
        granularity="rowwise",
        lambda_value=0.1,
    )
    torch.testing.assert_close(saved_masks["model.layers.0.self_attn.q_proj"], direct_masks["model.layers.0.self_attn.q_proj"])
