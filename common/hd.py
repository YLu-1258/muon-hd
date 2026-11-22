"""Baydin et al., equations 6 and 8, for updates of the form w -= lr * d."""
import math

import torch


def adapt(lr, beta, mode, pairs):
    if not pairs or beta == 0 or mode == "fixed":
        return lr
    dot = sum(torch.dot(g.reshape(-1), d.reshape(-1)) for g, d in pairs)
    if mode == "multiplicative":
        g2 = sum(torch.linalg.vector_norm(g).square() for g, _ in pairs)
        d2 = sum(torch.linalg.vector_norm(d).square() for _, d in pairs)
        denom = (g2 * d2).sqrt()
        alignment = torch.where(denom == 0, 0., dot / denom)
        lr *= 1 + beta * alignment.item()
    else:
        lr += beta * dot.item()
    if not math.isfinite(lr):
        raise FloatingPointError("Learning rate became nonfinite")
    return lr
