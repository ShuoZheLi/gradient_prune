import torch.nn as nn

from layer_utils import find_prunable_linears, iter_prunable_modules, normalize_prune_ops
from masks import compute_actual_sparsity


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.rotary_emb = nn.Identity()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.norm = nn.LayerNorm(4)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block()])
        self.lm_head = nn.Linear(4, 4, bias=False)


def test_normalize_prune_ops_aliases():
    assert normalize_prune_ops(["q", "up_proj"]) == ("self_attn.q_proj", "mlp.up_proj")


def test_find_prunable_linears_excludes_non_prunable():
    found = find_prunable_linears(Block())
    assert set(found) == {"self_attn.q_proj", "self_attn.k_proj", "mlp.up_proj"}


def test_actual_sparsity_on_prunable_modules():
    model = Model()
    for _, module in iter_prunable_modules(model):
        module.weight.data[:, :2] = 0
    stats = compute_actual_sparsity(model)
    assert abs(stats["actual_sparsity"] - 0.5) < 1e-6
