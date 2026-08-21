from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from recoverability_pruning import hvp as hvp_module
from recoverability_pruning.hvp import compute_batch_hvp, compute_dataset_hvp
from recoverability_pruning.losses import causal_sft_nll
from recoverability_pruning.params import build_parameter_space


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.input_anchor = nn.Embedding(vocab_size, 1)
        self.input_anchor.weight.requires_grad_(False)
        self.transition = nn.Linear(vocab_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.input_anchor

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        features = F.one_hot(input_ids, num_classes=self.transition.in_features).float()
        return SimpleNamespace(logits=self.transition(features))


def _functional_loss(flat_weight, input_ids, labels, vocab_size):
    weight = flat_weight.view(vocab_size, vocab_size)
    features = F.one_hot(input_ids, num_classes=vocab_size).float()
    logits = F.linear(features, weight)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        labels[:, 1:].reshape(-1),
        reduction="mean",
    )


def test_reverse_over_reverse_hvp_matches_explicit_hessian():
    torch.manual_seed(7)
    vocab_size = 3
    model = TinyCausalLM(vocab_size)
    parameter_space = build_parameter_space(
        model,
        candidate_modules=["transition"],
        hvp_parameter_scope="candidates",
    )
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 1]], dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2, 1]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    probe_tensor = torch.tensor(
        [[1, -1, 1], [-1, 1, -1], [1, 1, -1]],
        dtype=torch.int8,
    )
    probe = {"transition.weight": probe_tensor}
    estimated, num_tokens = compute_batch_hvp(model, batch, probe, parameter_space, causal_sft_nll)
    flat_weight = model.transition.weight.detach().clone().requires_grad_(True).reshape(-1)
    explicit_hessian = torch.autograd.functional.hessian(
        lambda value: _functional_loss(value, batch["input_ids"], batch["labels"], vocab_size),
        flat_weight,
    )
    exact = explicit_hessian @ probe_tensor.float().reshape(-1)
    assert num_tokens == 3
    torch.testing.assert_close(estimated["transition.weight"].reshape(-1), exact, rtol=2e-4, atol=2e-5)


def test_causal_nll_respects_shifted_label_mask():
    model = TinyCausalLM(vocab_size=3)
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 2, 1]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    result = causal_sft_nll(model, batch)
    logits = model(batch["input_ids"]).logits
    expected = F.cross_entropy(logits[:, 1:3, :].reshape(-1, 3).float(), torch.tensor([2, 1]))
    assert result.num_tokens == 2
    torch.testing.assert_close(result.loss, expected)


def test_dataset_hvp_is_weighted_by_supervised_token_count(monkeypatch):
    model = TinyCausalLM(vocab_size=2)
    parameter_space = build_parameter_space(
        model,
        candidate_modules=["transition"],
        hvp_parameter_scope="candidates",
    )
    responses = iter(
        [
            ({"transition.weight": torch.full((2, 2), 2.0)}, 1),
            ({"transition.weight": torch.full((2, 2), 5.0)}, 3),
        ]
    )

    def fake_batch_hvp(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(hvp_module, "compute_batch_hvp", fake_batch_hvp)
    dataloader = [{"unused": torch.tensor(0)}, {"unused": torch.tensor(1)}]
    result, info = compute_dataset_hvp(
        model,
        dataloader,
        {"transition.weight": torch.ones(2, 2, dtype=torch.int8)},
        parameter_space,
        causal_sft_nll,
        objective_name="test",
    )
    torch.testing.assert_close(result["transition.weight"], torch.full((2, 2), 4.25))
    assert info["num_tokens"] == 4.0


def test_dataset_hvp_matches_explicit_token_normalized_dataset_hessian():
    torch.manual_seed(19)
    vocab_size = 3
    model = TinyCausalLM(vocab_size)
    parameter_space = build_parameter_space(
        model,
        candidate_modules=["transition"],
        hvp_parameter_scope="candidates",
    )
    batches = [
        {
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "labels": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[1, 2, 0, 1, 0]], dtype=torch.long),
            "labels": torch.tensor([[1, 2, 0, 1, 0]], dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        },
    ]
    probe_tensor = torch.tensor(
        [[1, -1, 1], [-1, 1, -1], [1, 1, -1]],
        dtype=torch.int8,
    )
    estimated, info = compute_dataset_hvp(
        model,
        batches,
        {"transition.weight": probe_tensor},
        parameter_space,
        causal_sft_nll,
        objective_name="explicit-test",
    )

    flat_weight = model.transition.weight.detach().clone().requires_grad_(True).reshape(-1)

    def dataset_loss(value):
        loss_sum = value.new_zeros(())
        token_count = 0
        for batch in batches:
            weight = value.view(vocab_size, vocab_size)
            features = F.one_hot(batch["input_ids"], num_classes=vocab_size).float()
            logits = F.linear(features, weight)
            targets = batch["labels"][:, 1:]
            loss_sum = loss_sum + F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
            token_count += targets.numel()
        return loss_sum / token_count

    explicit_hessian = torch.autograd.functional.hessian(dataset_loss, flat_weight)
    exact = explicit_hessian @ probe_tensor.float().reshape(-1)
    assert info["num_tokens"] == 6.0
    torch.testing.assert_close(estimated["transition.weight"].reshape(-1), exact, rtol=2e-4, atol=2e-5)
