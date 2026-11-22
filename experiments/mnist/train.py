import argparse
import gzip
import tempfile
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, TensorDataset

from common.adam_hd import AdamHD
from common.log import Log
from experiments.mnist.optim import AdamWHD

METHODS = ("adam", "adam-hd", "adamw", "adamw-hd")


def dataset(root, train):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "train" if train else "t10k"
    values = []
    for suffix, header in (("images-idx3", 16), ("labels-idx1", 8)):
        name = f"{prefix}-{suffix}-ubyte.gz"
        path = root / name
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                tmp = Path(temporary) / name
                urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + name, tmp)
                tmp.replace(path)
        raw = gzip.decompress(path.read_bytes())
        values.append(torch.frombuffer(bytearray(raw[header:]), dtype=torch.uint8))
    x, y = values
    return TensorDataset((x.reshape(-1, 784).float() / 255 - .1307) / .3081, y.long())


def optimizer(model, method):
    kwargs = dict(lr=.001, weight_decay=.0001)
    cls = {"adam": torch.optim.Adam, "adam-hd": AdamHD,
           "adamw": torch.optim.AdamW, "adamw-hd": AdamWHD}[method]
    if method == "adam-hd":
        kwargs["hypergrad_lr"] = 1e-7
    if method == "adamw-hd":
        kwargs["beta"] = 1e-7
    if method in ("adam", "adamw"):
        kwargs["foreach"] = False
    return cls(model.parameters(), **kwargs)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return dict(validation_loss=loss / n, test_error=1 - correct / n)


def train(a, log):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(784, 1000), nn.ReLU(), nn.Linear(1000, 1000),
                          nn.ReLU(), nn.Linear(1000, 10)).to(a.device)
    opt = optimizer(model, a.method)
    if a.smoke:
        torch.set_num_threads(1)
        tr, va = [TensorDataset(torch.randn(256, 784), torch.randint(10, (256,))) for _ in range(2)]
    else:
        tr, va = dataset(a.data, True), dataset(a.data, False)
    tr = DataLoader(tr, batch_size=128, shuffle=True)
    va = DataLoader(va, batch_size=128)
    step = 0

    def rates():
        return {"global": opt.param_groups[0]["lr"]}

    model.eval()
    with torch.no_grad():
        x, y = next(iter(tr))
        initial = cross_entropy(model(x.to(a.device)), y.to(a.device)).item()
    log.record(step=step, epoch=0, train_loss=initial, rates=rates(), **evaluate(model, va, a.device))
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0
        for batch, (x, y) in enumerate(tr, 1):
            opt.zero_grad()
            loss = cross_entropy(model(x.to(a.device)), y.to(a.device))
            loss.backward()
            opt.step()
            step += 1
            total += loss.item()
            row = dict(step=step, epoch=epoch, train_loss=loss.item(), rates=rates())
            if batch == len(tr):
                row.update(epoch_train_loss=total / len(tr), **evaluate(model, va, a.device))
            log.record(**row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data", default="data/mnist")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.epochs < 1:
        p.error("--epochs must be positive")
    log = Log(a.output, dict(vars(a), seed=1, lr=.001, batch_size=128,
                             weight_decay=.0001, beta=1e-7 if a.method.endswith("-hd") else 0,
                             architecture=[784, 1000, 1000, 10], torch_version=torch.__version__))
    try:
        train(a, log)
    except BaseException as error:
        log.status("failed", error=str(error))
        raise
    log.status("complete")


if __name__ == "__main__":
    main()
