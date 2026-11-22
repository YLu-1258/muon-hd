import json
import time
from pathlib import Path

import torch


class Log:
    def __init__(self, output, config):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        self.best = float("inf")
        self.write("config.json", config)
        self.status("running")

    def write(self, name, data):
        tmp = self.path / (name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
        tmp.replace(self.path / name)

    def status(self, state, **row):
        self.write("status.json", {"state": state, **row})

    def record(self, **row):
        row["seconds"] = time.monotonic() - self.start
        with (self.path / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        self.status("running", **row)

    def checkpoint(self, model, loss):
        if loss < self.best:
            tmp = self.path / "best.pt.tmp"
            torch.save(model.state_dict(), tmp)
            tmp.replace(self.path / "best.pt")
            self.best = loss
