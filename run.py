"""Queue experiment jobs with one process per GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

RATES = [.002, .006, .02, .06, .2]
MODES = ["fixed", "additive", "multiplicative"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=["mnist", "llama"])
    p.add_argument("--output", type=Path)
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--rates", nargs=5, type=float, default=RATES)
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--train-args", nargs=argparse.REMAINDER, default=[])
    a = p.parse_args()
    if len(set(a.gpus)) != len(a.gpus):
        p.error("--gpus must list distinct devices")
    if len(set(a.rates)) != 5 or len(set(a.modes)) != len(a.modes):
        p.error("Rates and modes must be distinct")
    if a.suite == "mnist":
        jobs = {m: ["--method", m] for m in ("adam", "adam-hd", "adamw", "adamw-hd")}
        smoke = ["--epochs", "1"]
    else:
        jobs = {f"{mode}-{lr:g}": ["--mode", mode, "--lr", str(lr)]
                for lr in a.rates for mode in a.modes}
        smoke = ["--steps", "3", "--eval-every", "1", "--batch-size", "4"]

    extra = (["--smoke", "--device", "cpu", *smoke] if a.smoke else []) + a.train_args
    root = (a.output or Path(f"runs/{a.suite}-{datetime.now():%Y%m%d-%H%M%S-%f}")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "suite.json").write_text(json.dumps({"suite": a.suite, "jobs": jobs, "extra": extra}, indent=2))
    print(f"Run directory: {root}", flush=True)
    print(f"Monitor: {sys.executable} -m report {root} --watch", flush=True)

    pending = deque(jobs)
    active = {}
    failed = []
    try:
        while pending or active:
            for gpu, (name, proc) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    if code:
                        failed.append(name)
                        (root / "logs" / f"{name}.failed").write_text(str(code))
                    print(f"{name}: {'failed' if code else 'complete'}", flush=True)
                    del active[gpu]
            for gpu in a.gpus:
                if gpu in active or not pending:
                    continue
                name = pending.popleft()
                cmd = [sys.executable, "-m", f"experiments.{a.suite}.train",
                       *extra, *jobs[name], "--output", str(root / name)]
                with (root / "logs" / f"{name}.log").open("w") as log:
                    proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                            stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (name, proc)
            if active:
                time.sleep(.25)
    finally:
        for _, proc in active.values():
            proc.terminate()
        for _, proc in active.values():
            proc.wait()
    from report import plot
    plot(root)
    if failed:
        raise SystemExit(f"Failed runs: {', '.join(failed)}. See {root / 'logs'}.")


if __name__ == "__main__":
    main()
