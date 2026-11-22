import math

import torch

from common.hd import adapt
from vendor.muon.muon import muon_update


class Muon(torch.optim.Optimizer):
    def __init__(self, matrices, lr, mode, beta, granularity):
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(beta) or beta < 0:
            raise ValueError("Expected positive initial LR and nonnegative finite beta")
        if mode == "multiplicative" and beta >= 1:
            raise ValueError("Multiplicative beta must be below 1")
        groups = (
            [dict(params=[p], name=name) for name, p in matrices]
            if granularity == "matrix"
            else [dict(params=[p for _, p in matrices], name="global")]
        )
        super().__init__(groups, dict(lr=lr))
        self.mode, self.beta = mode, beta

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            pairs = [
                (p.grad, self.state[p]["previous"])
                for p in params if "previous" in self.state[p]
            ]
            group["lr"] = adapt(group["lr"], self.beta, self.mode, pairs)
            for p in params:
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                direction = muon_update(p.grad, state["momentum"], beta=.95, ns_steps=5)
                p.add_(direction, alpha=-group["lr"])
                if self.mode != "fixed" and self.beta:
                    state["previous"] = direction.float()

    def rates(self):
        return {g["name"]: g["lr"] for g in self.param_groups}
