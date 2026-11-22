import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def runs(root):
    return json.loads((root / "suite.json").read_text())


def metrics(path):
    if not path.exists():
        return []
    # An in-progress JSONL append may leave an incomplete last row.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n")]


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suite = runs(root)
    fields = ("train_loss", "validation_loss", "rates")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    for name in suite["jobs"]:
        rows = metrics(root / name / "metrics.jsonl")
        for axis, field in zip(axes, fields):
            key = "epoch_train_loss" if field == "train_loss" and suite["suite"] == "mnist" else field
            points = [(r["step"], r[key]) for r in rows if key in r]
            if not points:
                continue
            x, y = zip(*points)
            if field == "rates":
                line, = axis.plot(x, [statistics.mean(v.values()) for v in y], label=name)
                if len(y[0]) > 1:
                    axis.fill_between(x, [min(v.values()) for v in y], [max(v.values()) for v in y],
                                      color=line.get_color(), alpha=.12)
            else:
                axis.plot(x, y, label=name)
    for axis, title in zip(axes, ["Training cross-entropy", "Validation cross-entropy", "Learning rate"]):
        axis.set(title=title, xlabel="Optimizer step")
        axis.grid(alpha=.2)
    if axes[-1].has_data():
        axes[-1].set_yscale("symlog", linthresh=1e-4)
    if axes[0].has_data():
        axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)


def display(root):
    print(root)
    print(f"{'Run':<24} {'State':<11} {'Step':>7} {'Train CE':>10} {'Val CE':>10} {'Mean LR':>11}")
    colors = dict(complete=32, running=36, failed=31, starting=33)
    for name in runs(root)["jobs"]:
        status = root / name / "status.json"
        output = root / "logs" / f"{name}.log"
        state = json.loads(status.read_text())["state"] if status.exists() else (
            "starting" if output.exists() else "queued"
        )
        if (root / "logs" / f"{name}.failed").exists():
            state = "failed"
        rows = metrics(root / name / "metrics.jsonl")
        row = rows[-1] if rows else {}
        val = next((r["validation_loss"] for r in reversed(rows) if "validation_loss" in r), None)
        lr = statistics.mean(row["rates"].values()) if rows else None
        values = ["-" if v is None else f"{v:.5g}" for v in (row.get("train_loss"), val, lr)]
        line = f"{name:<24} {state:<11} {str(row.get('step', '-')):>7} {values[0]:>10} {values[1]:>10} {values[2]:>11}"
        print(f"\033[{colors.get(state, 0)}m{line}\033[0m" if sys.stdout.isatty() else line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--watch", action="store_true")
    group.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot:
        plot(args.output)
    else:
        try:
            while True:
                if args.watch and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                display(args.output)
                if not args.watch:
                    break
                time.sleep(3)
        except KeyboardInterrupt:
            pass
