import torch

from pruning_scores import signed_taylor_score


def test_signed_taylor_formula_exact_values():
    w = torch.tensor([1.0, -2.0])
    g = torch.tensor([0.5, -0.25])
    h = torch.tensor([2.0, 0.5])
    expected = -g * w + 0.5 * h * w.pow(2)
    actual = signed_taylor_score(w, g, h)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([0.5, 0.5]))

from pruning_scores import gradient_norm_score, wanda_score, hybrid_wanda_signed_taylor_score


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
