"""
Gradient Reversal Layer (GRL) for domain-adversarial training.
Acts as an identity during forward propagation and flips gradients during 
backpropagation to force invariant feature learning.
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
    """Module wrapper for the autograd GRL function."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def compute_grl_lambda(epoch, total_epochs):
    """
    Standard progressive lambda schedule (Ganin et al. 2016).
    Starts at 0 and saturates at 1 as training progresses.
    """
    p = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

