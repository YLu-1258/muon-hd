# Muon hypergradient experiments

The MNIST suite is in `experiments/mnist/`. The three GSM8K experiments share the training loop in `experiments/llama/`. Both suites log their settings and metrics, and the launcher queues one job on each listed GPU.

| Experiment | Model and task | Comparison |
| --- | --- | --- |
| 1 | Paper MLP on MNIST | Adam vs. Adam-HD as the paper reference, plus AdamW vs. AdamW-HD |
| 2 | Llama 3.2 1B base on GSM8K | Fixed learning rate at five values |
| 3 | Same Llama and data | Additive HD from the same five initial rates |
| 4 | Same Llama and data | Multiplicative HD from the same five initial rates |

## Run on arti

From the repo root, use the environment's interpreter directly. The Llama run uses the pinned Hugging Face model snapshot and needs cache access or prior access to the gated model. MNIST downloads its IDX files on first use.

```bash
cd /data/alexl/muon-hd
PY=/data/alexl/miniconda3/env/muonp/bin/python
"$PY" run.py mnist --gpus 1
"$PY" run.py llama --gpus 1 2 3 4
```

Run either suite independently. The launcher prints an absolute run path and a monitor command. Open the monitor in another terminal from the repo root.

```bash
"$PY" -m report /absolute/path/to/run --watch
```

Each suite writes `comparison.png` after its jobs finish. Use `"$PY" -m report /absolute/path/to/run --plot` to regenerate it. Each job writes `config.json`, `metrics.jsonl`, and a status file. Training logs are under `logs/`. Llama's optional `--save-best` saves one full model state dict per job by lowest post-update development loss. Enable it via `--train-args --save-best`.

`--modes fixed`, `--modes additive`, and `--modes multiplicative` select individual sweeps. `--rates` takes exactly five distinct values. Put shared training overrides last, for example `--train-args --steps 120 --eval-every 10 --microbatch 1`. Use `--train-args --granularity matrix` for separate adaptive rates per eligible weight matrix. The default is one shared Muon rate. Every actual rate is recorded, and the plot shows the matrix mean and min–max range when applicable.

`run.py mnist --smoke` and `run.py llama --smoke` exercise the full launcher and plotting paths using synthetic CPU data without model or dataset downloads.

## Experimental settings

**MNIST.** The official PyTorch MLP is 784 → 1000 → 1000 → 10 with ReLU. We use MNIST normalization (0.1307, 0.3081), 100 epochs, batch size 128, initial LR 0.001, decay 0.0001, Adam betas (0.9, 0.999), and HD beta 1e-7, matching the published MLP Adam commands. The validation series is cross-entropy on the official MNIST test split, following the authors' terminology. We log test error as well. Adam-HD retains the authors' per-parameter sequential rate updates. Its compatibility port changes the legacy PyTorch calls and turns the scalar hypergradient into a Python number. AdamW is the requested extension with decoupled decay. AdamW-HD sums the gradient dot products before applying one shared rate, using the previous AdamW direction including the decay term.

**GSM8K.** All three modes run 120 updates with validation at step 0 and every 10 steps, effective batch 128, microbatch 1, and max length 1024. The objective is next-token cross-entropy on answer tokens including EOS. Prompts and padding are masked. A seeded split holds out 256 examples from GSM8K main's training split as development data. All jobs see the same sampled training examples. The official GSM8K test split is unused, and the plotted score is answer-token loss. Training uses BF16 autocast and gradient checkpointing with FP32 parameters.

The default Muon rates are **0.002, 0.006, 0.02, 0.06, 0.2**, a screening grid around upstream Muon's 0.02 default. They are explicit, changeable settings. Muon uses upstream's five Newton–Schulz iterations, momentum 0.95 with Nesterov, and upstream shape scaling on hidden matrices. Embeddings, the output head, and vector parameters use fixed AdamW at 2e-5 with betas (0.9, 0.95). Both optimizers use zero weight decay in the Llama runs. There is no warmup, scheduler, or clipping.

Let `D_prev` be the previous Muon direction after Newton–Schulz and shape scaling, and `g` the current training gradient. For a weight update `W -= lr * D`, the estimated hypergradient is `-dot(g, D_prev)`. Additive HD uses `lr += beta * dot(g, D_prev)`, with a starting beta of 0.001. Multiplicative HD uses `lr *= 1 + beta * cosine(g, D_prev)`, with beta 0.02. Global cosine uses the concatenated gradient and direction across all eligible matrices. The first step uses the initial rate. Later steps reuse the normal training backward pass. Additive HD can produce negative rates, which are recorded directly.

The upstream source revisions are recorded in `vendor/sources.json`. The Baydin repo is the authors' PyTorch reimplementation of their original LuaTorch experiments. See the [paper](https://arxiv.org/abs/1703.04782), [authors' code](https://github.com/gbaydin/hypergradient-descent), and [upstream Muon](https://github.com/KellerJordan/Muon).
