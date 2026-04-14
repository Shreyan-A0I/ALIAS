"""
Gradient Reversal Layer for domain-adversarial training.
Implements Ganin et al. (2016) - forward pass is identity,
backward pass flips and scales gradients by -lambda.
"""

import torch
from torch.autograd import Function
import math


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    """Wrapper module for the GRL function."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def compute_grl_lambda(epoch, total_epochs):
    """
    Progressive lambda schedule from Ganin et al. (2016).
    λ(p) = 2 / (1 + exp(-10p)) - 1
    where p = epoch / total_epochs (0 → 1).
    Starts near 0 and saturates near 1.
    """
    p = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
