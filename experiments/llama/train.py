import argparse
import contextlib
import math

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from common.log import Log
from experiments.llama import data
from experiments.llama.optim import Muon


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.hidden = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.head = nn.Linear(16, 32)

    def forward(self, input_ids, attention_mask, labels):
        logits = self.head(self.hidden(self.embedding(input_ids).cumsum(1)))
        return type("Output", (), {"loss": cross_entropy(
            logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1)
        )})

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


def loss_pass(model, rows, a, pad, backward=False):
    counts = [sum(token != -100 for token in row["labels"][1:]) for row in rows]
    total = sum(counts)
    losses = []
    for start in range(0, len(rows), a.microbatch):
        end = start + a.microbatch
        batch = data.collate(rows[start:end], pad, a.device)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if a.device.startswith("cuda") else contextlib.nullcontext())
        with autocast:
            loss = model(**batch).loss * (sum(counts[start:end]) / total)
        if backward:
            loss.backward()
        losses.append(loss.detach())
    loss = torch.stack(losses).sum().item()
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    return loss


def train(a, log):
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.smoke:
        torch.set_num_threads(1)
        model, tokenizer, pad = TinyLM(), None, 0
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision)
        pad = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            a.model, revision=a.model_revision, dtype=torch.float32, attn_implementation="sdpa"
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model.to(a.device)
    excluded = {id(model.get_input_embeddings().weight), id(model.get_output_embeddings().weight)}
    matrices = [(n, p) for n, p in model.named_parameters() if p.ndim == 2 and id(p) not in excluded]
    ids = {id(p) for _, p in matrices}
    aux = [p for p in model.parameters() if id(p) not in ids]
    opt = Muon(matrices, a.lr, a.mode, a.beta, a.granularity)
    fallback = torch.optim.AdamW(aux, lr=a.aux_lr, betas=(.9, .95), weight_decay=0)
    rows, split = data.prepare(a, tokenizer)
    log.write("split.json", split)
    log.write("model.json", {"muon_parameters": [n for n, _ in matrices],
                            "revision": getattr(getattr(model, "config", None), "_commit_hash", None)})
    generator = torch.Generator().manual_seed(a.seed + 1)

    @torch.no_grad()
    def validate():
        model.eval()
        return loss_pass(model, rows["validation"], a, pad)

    log.record(step=0, validation_loss=validate(), rates=opt.rates())
    for step in range(1, a.steps + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        train_loss = loss_pass(model, data.sample(rows["train"], a.batch_size, generator), a, pad, True)
        opt.step()
        fallback.step()
        row = dict(step=step, train_loss=train_loss, rates=opt.rates())
        if step % a.eval_every == 0 or step == a.steps:
            row["validation_loss"] = validate()
            if a.save_best:
                log.checkpoint(model, row["validation_loss"])
        log.record(**row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fixed", "additive", "multiplicative"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--beta", type=float)
    p.add_argument("--granularity", choices=["global", "matrix"], default="global")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--aux-lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--model-revision", default="4e20de362430cd3b72f300e6b0f18e50e7166e08")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if min(a.steps, a.eval_every, a.batch_size, a.microbatch) < 1:
        p.error("step counts and batch sizes must be positive")
    if a.beta is None:
        a.beta = {"fixed": 0, "additive": .001, "multiplicative": .02}[a.mode]
    log = Log(a.output, dict(vars(a), ns_steps=5, muon_momentum=.95, warmup_steps=0,
                             weight_decay=0, auxiliary_betas=[.9, .95], torch_version=torch.__version__))
    try:
        train(a, log)
    except BaseException as error:
        log.status("failed", error=str(error))
        raise
    log.status("complete")


if __name__ == "__main__":
    main()
