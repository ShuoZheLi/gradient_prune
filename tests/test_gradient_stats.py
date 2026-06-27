import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gradient_stats import _iter_single_examples


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(0.0)

    def forward(self, x):
        return self.linear(x).mean()


def _microbatch_stats(model, values, batch_size):
    grad_sum = torch.zeros_like(model.linear.weight)
    grad_sq_sum = torch.zeros_like(model.linear.weight)
    count = 0
    for (x,) in DataLoader([(torch.tensor([[v]], dtype=torch.float32),) for v in values], batch_size=batch_size):
        loss = model(x)
        loss.backward()
        grad = model.linear.weight.grad.detach().clone()
        grad_sum += grad
        grad_sq_sum += grad.pow(2)
        count += 1
        model.zero_grad(set_to_none=True)
    return grad_sum / count, grad_sq_sum / count


def _per_example_stats(model, values, loader_batch_size):
    grad_sum = torch.zeros_like(model.linear.weight)
    grad_sq_sum = torch.zeros_like(model.linear.weight)
    count = 0
    for (x,) in DataLoader([(torch.tensor([[v]], dtype=torch.float32),) for v in values], batch_size=loader_batch_size):
        batch = {"x": x}
        for example in _iter_single_examples(batch):
            loss = model(example["x"])
            loss.backward()
            grad = model.linear.weight.grad.detach().clone()
            grad_sum += grad
            grad_sq_sum += grad.pow(2)
            count += 1
            model.zero_grad(set_to_none=True)
    return grad_sum / count, grad_sq_sum / count


def test_iter_single_examples_splits_batch_dimension():
    batch = {"input_ids": torch.arange(6).view(3, 2), "labels": torch.arange(3)}
    examples = list(_iter_single_examples(batch))
    assert len(examples) == 3
    assert examples[1]["input_ids"].tolist() == [[2, 3]]
    assert examples[1]["labels"].tolist() == [1]


def test_per_example_h_independent_of_loader_batch_size():
    values = [1.0, -2.0, 3.0, -4.0]
    g1, h1 = _per_example_stats(ToyModel(), values, loader_batch_size=1)
    g4, h4 = _per_example_stats(ToyModel(), values, loader_batch_size=4)
    torch.testing.assert_close(g1, g4)
    torch.testing.assert_close(h1, h4)


def test_microbatch_h_changes_with_batch_size():
    values = [1.0, -1.0]
    _, h1 = _microbatch_stats(ToyModel(), values, batch_size=1)
    _, h2 = _microbatch_stats(ToyModel(), values, batch_size=2)
    assert not torch.allclose(h1, h2)
