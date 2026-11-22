import torch

from common.hd import adapt


class AdamWHD(torch.optim.AdamW):
    """Additive HD with the actual AdamW direction, including decoupled decay."""

    def __init__(self, params, *, lr, weight_decay, beta):
        super().__init__(params, lr=lr, weight_decay=weight_decay, foreach=False)
        self.beta = beta

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        params = [p for p in group["params"] if p.grad is not None]
        group["lr"] = adapt(group["lr"], self.beta, "additive", [
            (p.grad, self.state[p]["previous"])
            for p in params if "previous" in self.state[p]
        ])
        decay = group["weight_decay"]
        before = [p.detach().mul(decay) for p in params]
        super().step()
        b1, b2 = group["betas"]
        for p, decay_direction in zip(params, before):
            state = self.state[p]
            t = state["step"].item()
            denom = state["exp_avg_sq"].sqrt() / (1 - b2**t)**.5 + group["eps"]
            state["previous"] = decay_direction.addcdiv_(
                state["exp_avg"], denom, value=1 / (1 - b1**t)
            )
