# mmlu-reasoning-benchmark

Paired **reasoning vs. non-reasoning** evaluation of local GGUF LLMs on the
official **MMLU** test set, using
[`llama.cpp`](https://github.com/ggml-org/llama.cpp)'s `llama-server`.

This is the follow-up to **[`mmlu-local-benchmark`](https://github.com/Jorgen-Bergstrom/mmlu-local-benchmark)**
(Part 1), where every model answered with a grammar-constrained single letter
over the raw `/completion` endpoint and never got to deliberate. Here the model
is allowed to think: the harness uses the chat path so the model's native
thinking mode can engage, lets it generate freely, and parses the final answer
out of the text.

> The companion blog write-up lives on
> [bergstrom.org](https://bergstrom.org/posts/mmlu_reasoning/).

## The paired design

Comparing a reasoning run against a *different* set of questions is confounded:
MMLU difficulty varies a lot, so any gap could just be which questions were
sampled. This harness answers **the same questions twice** and counts the flips:

1. Sample `N` questions round-robin across all 57 subjects (seeded, so it is
   reproducible).
2. Obtain both arms for each question:
   - **arm "off" (control)** — by default the *published* Part 1 prediction for
     the same model, read from `../Benchmark_MMLU/results/<model>/predictions.jsonl`
     and matched by its `subject:idx` id. Only the reasoning arm is executed,
     which roughly halves the wall-clock.
   - **arm "on" (reasoning)** — `/v1/chat/completions` with
     `enable_thinking: true` and `reasoning_effort: xhigh`; thoughts land in
     `reasoning_content`, the final answer in `content`.
3. Parse the final `Answer: X` line and run an exact two-sided **McNemar** test
   on the discordant pairs.

Pass `--baseline-results none` to instead run a fresh chat-based non-reasoning
control (`enable_thinking: false`, asked to answer directly).

**Honest framing:** by default this compares two *methods* — a one-token grammar
baseline vs. chat + a "reason first" instruction — not two settings of a single
switch. Some of the gap is method as well as mechanism.

## Results

`qwen3.8-27B-IQ3_S`, 1,000 questions sampled round-robin across all 57 subjects
(~17–18 each), single NVIDIA RTX 5060 Ti (16 GB), greedy decoding:

| Arm | Overall | Macro | Avg tokens | Avg latency |
|---|---:|---:|---:|---:|
| non-reasoning (Part 1 baseline) | 82.5% | 82.6% | 1 | 2.9 s |
| reasoning (`xhigh`) | **91.2%** | 91.1% | 1,188 | 42.2 s |

Reasoning adds **+8.7 points** (95% CI +6.3 to +11.1), exact McNemar
**p ≈ 3.7 × 10⁻¹³** across 1,000 paired questions. It **fixed 118** questions
and **broke 31**. Gains concentrate in mathematics, formal logic and physics.

Run outputs are intentionally **not committed** — they are regenerable by
re-running the harness (see [Outputs](#outputs)). Only the headline numbers are
recorded here.

## Requirements

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) built with `llama-server`,
  including `--jinja`, `--reasoning-format` and `--chat-template-kwargs` support
  (default path `~/llama.cpp/build/bin/llama-server`)
- Python 3 with the `requests` package
- The MMLU CSVs under `mmlu/data/` (see below)

## Getting the MMLU data

The dataset is **not** committed (it is large and publicly available). Download
the canonical Berkeley tarball and extract it into `mmlu/`:

```bash
cd mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar          # creates mmlu/data/{test,dev,val,auxiliary_train}/
```

The `dev` split supplies the standard 5-shot examples; the `test` split is the
scoring set (14,042 questions across 57 subjects). Each CSV row is
`question, A, B, C, D, answer`. (Equivalently: `load_dataset("cais/mmlu", "all")`
on Hugging Face.)

## Running it

```bash
python3 benchmark_mmlu_reasoning.py --list      # show registered models
python3 benchmark_mmlu_reasoning.py --dry-run   # print the exact server command
LIMIT=20  ./run_reasoning.sh                    # smoke test
LIMIT=1000 ./run_reasoning.sh                   # the paired study (~11 h)
```

For a long run, use the detached launcher so it survives the launching shell:

```bash
LIMIT=1000 ./run_detached.sh     # logs to real_run.log, PID in results/<model>/run.pid
tail -f real_run.log             # watch progress
```

Runs are **resumable**: predictions stream to
`results/<model>/predictions.jsonl`, keyed by `(question id, arm)`. Re-running
resumes where it left off (`--retry-incomplete` re-runs truncated/unparseable
records by default).

> `llama-server` is started in its own session, so if the benchmark process is
> killed abruptly the server is orphaned and keeps holding the GPU. Stop both:
> `pkill -f "[b]enchmark_mmlu_reasoning.py"; pkill -f "[l]lama-server"`

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `qwen3.8-27B-IQ3_S` | registry key (`--list`) |
| `--limit N` | all 14,042 | **total** questions, sampled round-robin across subjects |
| `--seed N` | `0` | makes `--limit` sampling reproducible |
| `--arms on\|off\|both` | `both` | run one arm only |
| `--baseline-results PATH` | `auto` | Part 1 `predictions.jsonl` used as the non-reasoning arm (`auto` finds `../Benchmark_MMLU/results/<model>/`; `none` runs a fresh chat control) |
| `--shots N` | `5` | few-shot examples from the dev split |
| `--ctx N` | `131072` | server context window |
| `--max-tokens N` | `16384` | generation cap per request (includes the reasoning trace) |
| `--reasoning-budget N` | `8192` | thinking-token budget; when hit the server makes the model finalise instead of hard-truncating (`-1` = unrestricted) |
| `--reasoning-effort L` | `xhigh` | `xhigh` / `medium` / `low` |
| `--concurrency N` | `1` | parallel slots; **keep at 1** for reasoning |
| `--fresh` | off | ignore previous results, start over |

Model launch configurations live in `benchmark_mmlu_reasoning.py` under
`MODEL_REGISTRY`. Edit them there for your own models.

## Outputs

Per model in `results/<model>/` (created at runtime, gitignored):

- `predictions.jsonl` — one line per `(question, arm)` with prediction, gold,
  `correct`, `parse_method`, `finish_reason`, `truncated`, token counts, latency,
  and the raw content/reasoning.
- `run.json` — config, server command, per-arm stats, paired/McNemar stats,
  per-subject deltas.
- `server.log` — raw `llama-server` output.

Cross-arm: `results/summary.md` — human-readable table + paired comparison.

### Reproducing the paired numbers from a fresh clone

Because results are not committed, the default `auto` baseline
(`../Benchmark_MMLU/results/<model>/predictions.jsonl`) is not present in a
fresh clone. Either:

- fetch the matching `predictions.jsonl` from the
  [`mmlu-local-benchmark`](https://github.com/Jorgen-Bergstrom/mmlu-local-benchmark)
  results/releases and pass `--baseline-results /path/to/predictions.jsonl`, or
- re-run both arms from scratch with `--baseline-results none`.

The Part 1 baseline scores 80.8% on the full test set and 82.5% on this
1,000-question sample, so the sample is slightly easier. Compare **paired
deltas**, never the reasoning sample against the full-set number.

## Layout

```
benchmark_mmlu_reasoning.py   # paired harness (server lifecycle + reasoning/baseline join + McNemar)
run_reasoning.sh              # smoke test / study runner
run_detached.sh               # setsid launcher for multi-hour runs
mmlu/data/{test,dev,...}/     # MMLU CSVs (not committed; see above)
results/<model>/              # run.json, predictions.jsonl, server.log (not committed)
results/summary.md            # cross-arm table + paired report (not committed)
```

## License

MIT — see [`LICENSE`](LICENSE).
