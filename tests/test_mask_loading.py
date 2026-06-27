import torch

from apply_pruning import load_masks, save_masks
from experiment_runner import _resolve_mask_dir
from types import SimpleNamespace


def test_save_and_load_masks_round_trip(tmp_path):
    masks = {"model.layers.0.self_attn.q_proj": torch.tensor([[True, False], [False, True]])}
    save_masks(masks, tmp_path, {"method": "m"})
    loaded = load_masks(tmp_path)
    assert torch.equal(loaded["model.layers.0.self_attn.q_proj"], masks["model.layers.0.self_attn.q_proj"])


def test_resolve_mask_dir_uses_configured_mask_root(tmp_path):
    cfg = SimpleNamespace(pruning=SimpleNamespace(mask_root=str(tmp_path)), output=SimpleNamespace(root_dir="unused"))
    path = _resolve_mask_dir(cfg, "signed_taylor", 0.5, None)
    assert path == tmp_path / "method=signed_taylor" / "sparsity=0.5" / "lambda=none"


def test_resolve_mask_dir_defaults_to_output_root():
    cfg = SimpleNamespace(pruning=SimpleNamespace(mask_root=None), output=SimpleNamespace(root_dir="results/x"))
    path = _resolve_mask_dir(cfg, "hybrid", 0.1, 0.01)
    assert path.as_posix() == "results/x/masks/method=hybrid/sparsity=0.1/lambda=0.01"

from experiment_runner import GRAD_METHODS, ACT_METHODS


def test_loaded_masks_do_not_require_score_stats():
    methods = ["signed_taylor", "hybrid_wanda_signed_taylor"]
    load_masks = True
    need_grad = (not load_masks) and any(method in GRAD_METHODS for method in methods)
    need_act = (not load_masks) and any(method in ACT_METHODS for method in methods)
    assert need_grad is False
    assert need_act is False
