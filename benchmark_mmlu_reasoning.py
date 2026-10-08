#!/usr/bin/env python3
"""Paired reasoning vs non-reasoning MMLU benchmark for local LLMs (llama.cpp).

This is the follow-up to Benchmark_MMLU. The original harness forced a single
letter with a GBNF grammar over the raw /completion endpoint, so a model could
never deliberate. Here we use the *chat* path so the model's thinking mode can
engage, let it generate freely, and parse the final answer out of the text.

The non-reasoning arm is, by default, the *published* Benchmark_MMLU result
for the same model: this project reads ../Benchmark_MMLU/results/<model>/
predictions.jsonl and matches every question by its `subject:idx` id. That
makes the comparison "reasoning follow-up vs the original one-token grammar
baseline" and means only the reasoning arm has to be run (~half the time).

    arm "off"  -> published Benchmark_MMLU prediction (raw /completion, grammar)
    arm "on"   -> thinking enabled, prompt asks it to reason first

Pass --baseline-results none to instead run a fresh non-reasoning control over
the chat path (thinking disabled, asked to answer directly). Because the two
arms always see the exact same questions, we can report a proper paired
comparison (wrong->right vs right->wrong flips) and a McNemar test, which stays
meaningful even when only a subset is run.

The paired "off" arm is a chat/grammar baseline, not a re-implementation of the
one-token grammar run; both are documented references.

Examples:
  python3 benchmark_mmlu_reasoning.py --list
  python3 benchmark_mmlu_reasoning.py --dry-run
  python3 benchmark_mmlu_reasoning.py --limit 20                 # smoke test
  python3 benchmark_mmlu_reasoning.py --limit 1000               # paired study
  python3 benchmark_mmlu_reasoning.py                            # full test set
"""

import argparse
import csv
import json
import math
import os
import random
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("This script requires the 'requests' package (pip install requests)")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MMLU_DATA_DIR = PROJECT_DIR / "mmlu" / "data"
RESULTS_DIR = PROJECT_DIR / "results"

LLAMA_SERVER = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
GGUF_CACHE = Path.home() / ".cache" / "llama.cpp"

# Reasoning runs need room for a long trace, so the read timeout is generous.
REQUEST_TIMEOUT = (10, 1800)  # (connect, read) seconds
REQUEST_RETRIES = 3

ARMS = ("off", "on")

# --------------------------------------------------------------------------
# Model registry — reasoning-capable models only.
#
# These server args are model-loading args. The script appends
#   -c <ctx>  -np <concurrency>  --reasoning-format deepseek
# so the context window, slot count and thought extraction are controlled by
# CLI flags rather than being baked in here.
# --------------------------------------------------------------------------

MODEL_REGISTRY = {
    "qwen3.8-27B-IQ3_S": {
        "gguf": "Qwen3.8-27B-UD-IQ3_S.gguf",
        "server_args": "-ngl 99 -ctk q4_0 -ctv q4_0 -fa on --jinja",
        "notes": "alias qwen3.8_27B | hybrid SSM+attention; q4_0 KV",
    },
    "qwen3.8-27B-IQ3_XXS": {
        "gguf": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "server_args": "-ngl 99 -ctk q4_0 -ctv q4_0 -fa on --jinja",
        "notes": "more aggressive quant, for the quant-recovery comparison",
    },
}

# --------------------------------------------------------------------------
# MMLU data
# --------------------------------------------------------------------------


def load_split(split):
    """Return {subject: [[question, A, B, C, D, answer], ...]} for a split."""
    split_dir = MMLU_DATA_DIR / split
    if not split_dir.is_dir():
        sys.exit(f"MMLU {split} split not found: {split_dir}")
    suffix = f"_{split}.csv"
    data = {}
    for path in sorted(split_dir.glob(f"*{suffix}")):
        subject = path.name[: -len(suffix)]
        with path.open(newline="", encoding="utf-8") as f:
            data[subject] = [row for row in csv.reader(f) if row]
    return data


FORMAT_INSTRUCTION = (
    "Give the final answer on its own line in exactly this format:\n"
    "Answer: X\n"
    "where X is one of A, B, C, D. Do not write anything after that line."
)

# The reasoning arm is told to think first; the direct control arm is not.
# Both share the same answer-format requirement, so parsing is identical.
REASONING_INSTRUCTION = (
    "Respond with your reasoning, then give the final answer on its own line "
    "in exactly this format:\n"
    "Answer: X\n"
    "where X is one of A, B, C, D. Do not write anything after that line."
)


def build_prompt(subject, dev_rows, test_row, shots, arm="on"):
    """5-shot MMLU prompt.

    Arm "on" gets the reasoning instruction; arm "off" is asked to answer
    directly. The final-answer format requirement is shared.
    """
    subj = subject.replace("_", " ")
    lines = [f"The following are multiple choice questions (with answers) about {subj}.", ""]
    for row in dev_rows[:shots]:
        q, a, b, c, d, ans = row
        lines += [f"Q: {q}", f"A. {a}", f"B. {b}", f"C. {c}", f"D. {d}", f"Answer: {ans}", ""]
    q, a, b, c, d, _gold = test_row
    instruction = REASONING_INSTRUCTION if arm == "on" else FORMAT_INSTRUCTION
    lines += [
        f"Q: {q}", f"A. {a}", f"B. {b}", f"C. {c}", f"D. {d}", "",
        instruction,
    ]
    return "\n".join(lines)


def build_scope(test_data, limit, seed, subjects=None):
    """Build the list of (subject, question_index) pairs to run.

    `limit` is the TOTAL number of questions, sampled round-robin across the
    selected subjects so a small study still covers all of MMLU. Sampling is
    seeded, so the same --limit/--seed always selects the same questions
    (which keeps resume working).
    """
    if subjects:
        test_data = {s: rows for s, rows in test_data.items() if s in subjects}
        missing = set(subjects) - set(test_data)
        if missing:
            sys.exit(f"Unknown subject(s): {', '.join(sorted(missing))}")

    rng = random.Random(seed)
    shuffled = {}
    for subject, rows in test_data.items():
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        shuffled[subject] = idxs

    if limit is None:
        scope = []
        for subject in sorted(shuffled):
            scope += [(subject, i) for i in shuffled[subject]]
        return scope

    scope = []
    order = sorted(shuffled)
    pos = {s: 0 for s in order}
    while len(scope) < limit:
        progressed = False
        for subject in order:
            if pos[subject] < len(shuffled[subject]):
                scope.append((subject, shuffled[subject][pos[subject]]))
                pos[subject] += 1
                progressed = True
                if len(scope) >= limit:
                    break
        if not progressed:
            break  # ran out of questions
    return scope


# --------------------------------------------------------------------------
# llama-server lifecycle
# --------------------------------------------------------------------------


class LlamaServer:
    def __init__(self, cmd, host, port, log_path, expected_model=None):
        self.cmd = cmd
        self.host = host
        self.port = port
        self.log_path = Path(log_path)
        self.expected_model = Path(expected_model) if expected_model else None
        self.proc = None
        self._log_file = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def _port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def start(self):
        # If something already listens on the port, our llama-server would fail
        # to bind, exit, and the benchmark would silently query the existing
        # server (i.e. benchmark the wrong model). Fail loudly instead.
        if self._port_in_use():
            raise RuntimeError(
                f"Something is already listening on {self.host}:{self.port}; "
                f"refusing to start. Stop that process first (or pass --port to "
                f"use a free port)."
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", buffering=1)
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _verify_serving_expected_model(self):
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited (code {self.proc.returncode}) while "
                f"starting; see {self.log_path}"
            )
        if self.expected_model is None:
            return
        try:
            r = requests.get(f"{self.base_url}/props", timeout=5)
            r.raise_for_status()
            served = r.json().get("model_path")
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not verify the model served at {self.base_url}: {e}"
            )
        if served and Path(served).name != self.expected_model.name:
            raise RuntimeError(
                f"Server at {self.base_url} is serving {Path(served).name!r}, "
                f"but expected {self.expected_model.name!r}; refusing to run "
                f"against the wrong model."
            )

    def wait_ready(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self.proc.returncode}); "
                    f"see {self.log_path}"
                )
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.ok:
                    self._verify_serving_expected_model()
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(
            f"llama-server not ready after {timeout_s}s; see {self.log_path}"
        )

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.close()


# --------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------

THINK_RE = re.compile(r"<think[^>]*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)
# "Answer: A", "Answer - B", "The answer is C", "Answer: **D**"
ANSWER_RE = re.compile(r"answer\s*(?:[:\-]|\bis\b)[\s*_(]*([ABCD])\b", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{\s*([ABCD])\s*\}")
PAREN_RE = re.compile(r"\(([ABCD])\)")
BOLD_RE = re.compile(r"\*\*([ABCD])\*\*")
STANDALONE_RE = re.compile(r"\b([ABCD])\b")


def split_think(content, reasoning):
    """Some templates inline  thinking...</think> in content instead of a
    separate reasoning_content field; normalise both cases."""
    if not reasoning and content and THINK_RE.search(content):
        reasoning = "\n".join(m.group(1) for m in THINK_RE.finditer(content))
        content = THINK_RE.sub("", content)
    return content, reasoning


def parse_answer(content, reasoning=""):
    """Extract a single A/B/C/D letter. Returns (letter|None, method)."""
    content, reasoning = split_think(content or "", reasoning or "")
    text = content.strip()

    for pattern, method in (
        (ANSWER_RE, "answer_label"),
        (BOXED_RE, "boxed"),
        (PAREN_RE, "paren"),
        (BOLD_RE, "bold"),
        (STANDALONE_RE, "standalone"),
    ):
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).upper(), method

    # Last resort: look at the tail of the reasoning trace.
    if reasoning:
        tail = reasoning[-500:]
        for pattern in (ANSWER_RE, STANDALONE_RE):
            matches = list(pattern.finditer(tail))
            if matches:
                return matches[-1].group(1).upper(), "reasoning_fallback"

    return None, "unparseable"


# --------------------------------------------------------------------------
# Chat client
# --------------------------------------------------------------------------


def ask_chat(base_url, prompt, enable_thinking, effort, max_tokens):
    """One /v1/chat/completions request. Returns a dict of the response."""
    kwargs = {"enable_thinking": bool(enable_thinking)}
    if enable_thinking and effort:
        kwargs["reasoning_effort"] = effort

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,        # greedy
        "top_k": 1,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": kwargs,
    }
    last_exc = None
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                              timeout=REQUEST_TIMEOUT)
            if r.status_code == 400:
                raise RuntimeError(f"HTTP 400 from llama-server: {r.text[:400]}")
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {}) or {}
            usage = data.get("usage", {}) or {}
            return {
                "content": msg.get("content") or "",
                "reasoning_content": msg.get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except requests.RequestException as e:
            last_exc = e
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30))
    raise last_exc


def run_one(base_url, subject, idx, test_row, dev_rows, shots, arm,
            effort, max_tokens, store_reasoning):
    qid = f"{subject}:{idx}"
    question, a, b, c, d, gold = test_row
    record = {
        "id": qid,
        "subject": subject,
        "question": question,
        "choices": [a, b, c, d],
        "gold": gold,
        "arm": arm,
        "prompt_style": "reasoning" if arm == "on" else "direct",
    }
    prompt = build_prompt(subject, dev_rows, test_row, shots, arm)
    t0 = time.monotonic()
    try:
        resp = ask_chat(base_url, prompt, enable_thinking=(arm == "on"),
                        effort=effort, max_tokens=max_tokens)
        record["latency_s"] = round(time.monotonic() - t0, 3)
        record["finish_reason"] = resp["finish_reason"]
        record["truncated"] = resp["finish_reason"] == "length"
        record["prompt_tokens"] = resp["prompt_tokens"]
        record["completion_tokens"] = resp["completion_tokens"]
        record["reasoning_chars"] = len(resp["reasoning_content"])
        record["content_chars"] = len(resp["content"])

        pred, method = parse_answer(resp["content"], resp["reasoning_content"])
        record["parse_method"] = method
        if pred is None:
            record["pred"] = None
            record["correct"] = False
            record["error"] = "unparseable response"
        else:
            record["pred"] = pred
            record["correct"] = (pred == gold)

        if store_reasoning:
            record["reasoning_content"] = resp["reasoning_content"]
            record["content"] = resp["content"]
    except Exception as e:
        record["pred"] = None
        record["correct"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        record["latency_s"] = round(time.monotonic() - t0, 3)
    return record


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(records, scope, arm):
    """Per-subject + overall accuracy for one arm.

    `records` must already be filtered to `arm`; `scope` is (subject, idx) pairs.
    """
    per_subject = {}
    for subject in sorted({s for s, _ in scope}):
        recs = [records[f"{subject}:{i}|{arm}"] for s, i in scope
                if s == subject and f"{subject}:{i}|{arm}" in records]
        answered = [r for r in recs if "error" not in r]
        correct = sum(1 for r in answered if r["correct"])
        per_subject[subject] = {
            "total": len(recs),
            "answered": len(answered),
            "failed": len(recs) - len(answered),
            "correct": correct,
            "accuracy": (correct / len(answered)) if answered else None,
        }
    total_answered = sum(v["answered"] for v in per_subject.values())
    total_correct = sum(v["correct"] for v in per_subject.values())
    total_failed = sum(v["failed"] for v in per_subject.values())
    accs = [v["accuracy"] for v in per_subject.values() if v["accuracy"] is not None]
    lat = [r["latency_s"] for r in records.values() if "latency_s" in r]
    comp = [r["completion_tokens"] for r in records.values()
            if r.get("completion_tokens") is not None]
    trunc = sum(1 for r in records.values() if r.get("truncated"))
    unparse = sum(1 for r in records.values()
                  if r.get("parse_method") == "unparseable")
    return {
        "total_questions": len(scope),
        "total_answered": total_answered,
        "total_failed": total_failed,
        "total_correct": total_correct,
        "overall_accuracy": (total_correct / total_answered) if total_answered else None,
        "macro_accuracy": (sum(accs) / len(accs)) if accs else None,
        "unparseable": unparse,
        "truncated": trunc,
        "avg_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "avg_completion_tokens": round(sum(comp) / len(comp), 1) if comp else None,
        "per_subject": per_subject,
    }


def mcnemar(off_correct, on_correct):
    """Paired comparison for questions answered in BOTH arms.

    b = off right / on wrong, c = off wrong / on right. Exact two-sided
    binomial p-value on the discordant pairs, plus the continuity-corrected
    chi-square statistic.
    """
    both_right = off_only = on_only = both_wrong = 0
    for o, n in zip(off_correct, on_correct):
        if o and n:
            both_right += 1
        elif o and not n:
            off_only += 1
        elif n and not o:
            on_only += 1
        else:
            both_wrong += 1
    n = off_only + on_only
    if n == 0:
        p_exact = 1.0
        chi2 = 0.0
    else:
        k = min(off_only, on_only)
        p_exact = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
        chi2 = (abs(off_only - on_only) - 1) ** 2 / n
    return {
        "pairs": both_right + off_only + on_only + both_wrong,
        "both_correct": both_right,
        "off_only_correct": off_only,
        "on_only_correct": on_only,
        "both_wrong": both_wrong,
        "delta_on_minus_off": on_only - off_only,
        "mcnemar_chi2": round(chi2, 4),
        "mcnemar_p_exact": round(p_exact, 6),
    }


def paired_stats(records, scope):
    """Accuracy delta and McNemar over questions completed in both arms."""
    off_c, on_c = [], []
    per_subject = {}
    for subject in sorted({s for s, _ in scope}):
        oc, nc = [], []
        for s, i in scope:
            if s != subject:
                continue
            ko, kn = f"{subject}:{i}|off", f"{subject}:{i}|on"
            ro, rn = records.get(ko), records.get(kn)
            if ro and rn and "error" not in ro and "error" not in rn:
                oc.append(ro["correct"])
                nc.append(rn["correct"])
        off_c += oc
        on_c += nc
        if oc:
            per_subject[subject] = {
                "pairs": len(oc),
                "off_accuracy": sum(oc) / len(oc),
                "on_accuracy": sum(nc) / len(nc),
                "delta": (sum(nc) - sum(oc)) / len(oc),
            }
    overall = mcnemar(off_c, on_c)
    if off_c:
        overall["off_accuracy"] = sum(off_c) / len(off_c)
        overall["on_accuracy"] = sum(on_c) / len(on_c)
        overall["delta"] = overall["on_accuracy"] - overall["off_accuracy"]
    return {"overall": overall, "per_subject": per_subject}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, float) else "n/a"


def _num(x, suffix=""):
    return f"{x}{suffix}" if x is not None else "n/a"


def write_summary(run_info):
    off = run_info["arms"]["off"]
    on = run_info["arms"]["on"]
    p = run_info["paired"]["overall"]
    lines = [
        f"# MMLU reasoning vs non-reasoning — {run_info['model']}",
        "",
        f"- Questions per arm: {off['total_answered']} "
        f"(failed: off={off['total_failed']}, on={on['total_failed']})",
        f"- Effort: `{run_info['config']['reasoning_effort']}` | "
        f"ctx: {run_info['config']['ctx']} | max_tokens: "
        f"{run_info['config']['max_tokens']} | shots: {run_info['config']['shots']}",
        f"- Non-reasoning arm: "
        + (f"published baseline `{run_info['config']['baseline_results']}`"
           if run_info['config'].get('baseline_results')
           else "fresh chat/direct run (thinking disabled)"),
        f"- Reasoning arm: native thinking, `reasoning_effort="
        f"{run_info['config']['reasoning_effort']}`",
        "",
        "| Arm | Overall | Macro | Unparseable | Truncated | Avg tokens | Avg latency |",
        "|---|---|---|---|---|---|---|",
        f"| non-reasoning | {_pct(off['overall_accuracy'])} | {_pct(off['macro_accuracy'])} "
        f"| {off['unparseable']} | {off['truncated']} | {_num(off['avg_completion_tokens'])} "
        f"| {_num(off['avg_latency_s'], 's')} |",
        f"| reasoning | {_pct(on['overall_accuracy'])} | {_pct(on['macro_accuracy'])} "
        f"| {on['unparseable']} | {on['truncated']} | {_num(on['avg_completion_tokens'])} "
        f"| {_num(on['avg_latency_s'], 's')} |",
        "",
        "## Paired comparison (questions answered in both arms)",
        "",
        f"- Paired questions: {p['pairs']}",
        f"- Both correct: {p['both_correct']}",
        f"- Only non-reasoning correct: {p['off_only_correct']}",
        f"- Only reasoning correct: {p['on_only_correct']}",
        f"- Both wrong: {p['both_wrong']}",
        f"- Delta (reasoning - non-reasoning): "
        f"{(p.get('delta', 0) * 100):+.1f} points",
        f"- McNemar chi^2: {p['mcnemar_chi2']} | exact p: {p['mcnemar_p_exact']}",
        "",
        "## Per-subject delta (reasoning - non-reasoning)",
        "",
        "| Subject | Non-reasoning | Reasoning | Delta |",
        "|---|---:|---:|---:|",
    ]
    for subject, st in run_info["paired"]["per_subject"].items():
        lines.append(
            f"| {subject} | {_pct(st['off_accuracy'])} | "
            f"{_pct(st['on_accuracy'])} | {st['delta'] * 100:+.1f} |"
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------


def resolve_baseline_path(arg, model_key):
    """Locate the Benchmark_MMLU predictions used as the non-reasoning arm.

    'auto' looks in ../Benchmark_MMLU/results/<model>/predictions.jsonl;
    'none' disables baseline sourcing; anything else is a path.
    """
    if arg == "none":
        return None
    if arg == "auto":
        candidate = (PROJECT_DIR.parent / "Benchmark_MMLU" / "results"
                     / model_key / "predictions.jsonl")
        return candidate if candidate.exists() else None
    path = Path(arg)
    if not path.exists():
        sys.exit(f"--baseline-results file not found: {path}")
    return path


def load_baseline_records(path, scope):
    """Load Benchmark_MMLU predictions for the questions in `scope`, as arm 'off'.

    The original project keys predictions by the same `subject:idx` id, so we
    can match each reasoning question to its published non-reasoning answer.
    """
    want = {f"{s}:{i}" for s, i in scope}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] not in want:
                continue
            out[f"{rec['id']}|off"] = {
                "id": rec["id"],
                "subject": rec["subject"],
                "arm": "off",
                "prompt_style": "grammar_baseline",
                "baseline": True,
                "gold": rec["gold"],
                "pred": rec.get("pred"),
                "correct": rec["correct"],
                "latency_s": rec.get("latency_s"),
                "parse_method": "grammar_baseline",
            }
    return out


def run_benchmark(args, model_key, model_path, server_args, notes):
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    run_dir = RESULTS_DIR / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "server.log"

    # -np 1 keeps the whole context for the single reasoning trace; the
    # --reasoning-format deepseek flag separates thoughts from the answer, and
    # --reasoning-budget makes an over-thinking model finalise gracefully
    # instead of being hard-truncated at max_tokens.
    server_cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "--host", args.host,
        "--port", str(args.port),
        "-np", str(args.concurrency),
        "-c", str(args.ctx),
        "--reasoning-format", "deepseek",
    ] + (["--reasoning-budget", str(args.reasoning_budget)]
         if args.reasoning_budget >= 0 else []) + server_args

    if args.dry_run:
        print("Server command (not started):")
        print("  " + " ".join(server_cmd))
        print(f"Benchmark: MMLU test split, {args.shots}-shot, paired arms "
              f"{args.arms}, concurrency {args.concurrency}"
              + (f", limit {args.limit}" if args.limit else ""))
        print(f"Results would be written to: {run_dir}")
        return

    # Record our own PID so a detached run can be stopped reliably.
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")

    test_data = load_split("test")
    dev_data = load_split("dev")
    subjects = set(args.subjects.split(",")) if args.subjects else None
    scope = build_scope(test_data, args.limit, args.seed, subjects)

    # Reporting arms are what we compare; run_arms are what we execute. If a
    # published Benchmark_MMLU baseline is available, the non-reasoning arm is
    # sourced from it (matched by question id) instead of being re-run.
    arms = list(ARMS) if args.arms == "both" else [args.arms]
    baseline_path = resolve_baseline_path(args.baseline_results, model_key)
    baseline_loaded = baseline_path is not None
    run_arms = [a for a in arms if not (a == "off" and baseline_loaded)]

    # Resume: preload previously answered (id, arm) records. Incomplete
    # results (truncated, unparseable, or errored) are dropped so they are
    # re-run under the current settings, and the file is rewritten without
    # them to avoid duplicate ids.
    records = {}
    resume = pred_path.exists() and not args.fresh
    if resume:
        kept, dropped = [], 0
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                incomplete = (rec.get("truncated")
                              or rec.get("parse_method") == "unparseable"
                              or "error" in rec)
                if incomplete and args.retry_incomplete:
                    dropped += 1
                    continue
                kept.append(rec)
                records[f"{rec['id']}|{rec['arm']}"] = rec
        if dropped:
            with pred_path.open("w", encoding="utf-8") as f:
                for rec in kept:
                    f.write(json.dumps(rec) + "\n")
            print(f"Re-queued {dropped} incomplete record(s) for re-run")

    if baseline_loaded:
        records.update(load_baseline_records(baseline_path, scope))

    todo = [(s, i, arm) for s, i in scope for arm in run_arms
            if f"{s}:{i}|{arm}" not in records]
    print(f"Model: {model_key} ({model_path.name})")
    if baseline_loaded:
        matched = sum(1 for s, i in scope if f"{s}:{i}|off" in records)
        print(f"Non-reasoning arm: published baseline {baseline_path}")
        print(f"  baseline questions matched: {matched}/{len(scope)}")
    else:
        print("Non-reasoning arm: run fresh (chat, direct)")
    print(f"Paired questions in scope: {len(scope)} | arms: {arms} | "
          f"tasks to run: {len(todo)}")
    if not todo:
        print("Nothing to run (use --fresh to start over).")

    started_at = datetime.now(timezone.utc).isoformat()
    server = LlamaServer(server_cmd, args.host, args.port, log_path,
                         expected_model=model_path)
    server.start()
    print(f"Started llama-server (pid {server.proc.pid}), waiting for model to load...")
    try:
        server.wait_ready(args.startup_timeout)
        print(f"Server ready at {server.base_url} (ctx={args.ctx}, "
              f"np={args.concurrency})")

        elapsed = 0.0
        if todo:
            t0 = time.monotonic()
            done = 0
            lock = threading.Lock()
            pred_file = open(pred_path, "a" if resume else "w", encoding="utf-8")
            try:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = {
                        pool.submit(run_one, server.base_url, s, i,
                                    test_data[s][i], dev_data.get(s, []),
                                    args.shots, arm, args.reasoning_effort,
                                    args.max_tokens, args.store_reasoning): (s, i, arm)
                        for s, i, arm in todo
                    }
                    for fut in as_completed(futures):
                        rec = fut.result()
                        with lock:
                            records[f"{rec['id']}|{rec['arm']}"] = rec
                            pred_file.write(json.dumps(rec) + "\n")
                            pred_file.flush()
                            done += 1
                            if args.progress_every and (
                                    done % args.progress_every == 0
                                    or done == len(todo)):
                                elapsed_now = max(time.monotonic() - t0, 1e-9)
                                rate = done / elapsed_now
                                eta = (len(todo) - done) / rate if rate else 0
                                print(
                                    f"  {datetime.now().strftime('%H:%M:%S')}  "
                                    f"{done}/{len(todo)}  "
                                    f"{rec['id']}|{rec['arm']}  "
                                    f"pred={rec.get('pred')} gold={rec['gold']} "
                                    f"{'ok' if rec.get('correct') else 'XX'}  "
                                    f"tok={rec.get('completion_tokens')}  "
                                    f"{rec.get('latency_s')}s  "
                                    f"{rate:.3f} task/s  ETA {eta / 60:.1f} min",
                                    flush=True)
                print()
            finally:
                pred_file.close()
            elapsed = time.monotonic() - t0
    finally:
        server.stop()
        print("llama-server stopped")

    # Only aggregate over arms that were actually requested.
    arm_records = {arm: {k: v for k, v in records.items() if v["arm"] == arm}
                   for arm in arms}
    run_info = {
        "model": model_key,
        "gguf": str(model_path),
        "notes": notes,
        "server_command": server_cmd,
        "config": {
            "shots": args.shots,
            "concurrency": args.concurrency,
            "ctx": args.ctx,
            "max_tokens": args.max_tokens,
            "reasoning_budget": args.reasoning_budget,
            "reasoning_effort": args.reasoning_effort,
            "endpoint": "/v1/chat/completions",
            "temperature": 0,
            "limit": args.limit,
            "seed": args.seed,
            "subjects": args.subjects,
            "arms": arms,
            "baseline_results": str(baseline_path) if baseline_loaded else None,
            "host": args.host,
            "port": args.port,
        },
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "gpu": gpu_info(),
        "arms": {arm: aggregate(arm_records[arm], scope, arm) for arm in arms},
    }
    if "off" in arms and "on" in arms:
        run_info["paired"] = paired_stats(records, scope)

    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    if "off" in arms and "on" in arms:
        write_summary(run_info)

    print("\nPer-arm accuracy:")
    for arm in arms:
        a = run_info["arms"][arm]
        print(f"  {arm:4s} overall {_pct(a['overall_accuracy'])}  "
              f"macro {_pct(a['macro_accuracy'])}  "
              f"unparseable {a['unparseable']}  truncated {a['truncated']}  "
              f"avg {a['avg_completion_tokens']} tok / {a['avg_latency_s']}s")
    if "paired" in run_info:
        p = run_info["paired"]["overall"]
        print(f"\nPaired: off={_pct(p.get('off_accuracy'))} "
              f"on={_pct(p.get('on_accuracy'))} delta={p.get('delta', 0) * 100:+.1f} pts "
              f"| regressions(off-only) {p['off_only_correct']} "
              f"vs improvements(on-only) {p['on_only_correct']} "
              f"| McNemar p={p['mcnemar_p_exact']}")
    print(f"Results written to {run_dir / 'run.json'}")


def main():
    ap = argparse.ArgumentParser(
        description="Paired reasoning vs non-reasoning MMLU benchmark.")
    ap.add_argument("--model", default="qwen3.8-27B-IQ3_S",
                    help="model key from the registry (see --list)")
    ap.add_argument("--model-file",
                    help="path to a GGUF not in the registry (use with --server-args)")
    ap.add_argument("--server-args", default="",
                    help="extra llama-server args for --model-file")
    ap.add_argument("--list", action="store_true", help="list registered models")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact server command without starting anything")
    ap.add_argument("--limit", type=int, default=None,
                    help="TOTAL number of questions (round-robin across subjects; "
                         "default: all 14042)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed for --limit (default: 0)")
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subject filter, e.g. abstract_algebra,astronomy")
    ap.add_argument("--arms", choices=["both", "on", "off"], default="both",
                    help="which arm(s) to run (default: both, paired)")
    ap.add_argument("--baseline-results", default="auto",
                    help="Benchmark_MMLU predictions.jsonl to use as the "
                         "non-reasoning arm, matched by question id. 'auto' "
                         "(default) looks in ../Benchmark_MMLU/results/<model>/, "
                         "'none' runs the non-reasoning arm fresh instead")
    ap.add_argument("--shots", type=int, default=5, choices=range(1, 6),
                    help="few-shot examples from the dev split (default: 5)")
    ap.add_argument("--ctx", type=int, default=131072,
                    help="server context size; whole window goes to the single "
                         "slot when concurrency=1 (default: 131072)")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="max tokens to generate per request (default: 16384; "
                         "includes the thinking trace)")
    ap.add_argument("--reasoning-budget", type=int, default=8192,
                    help="token budget for the thinking phase; when reached the "
                         "server prompts the model to finalise, avoiding hard "
                         "truncation. -1 = unrestricted (default: 8192)")
    ap.add_argument("--reasoning-effort", default="xhigh",
                    help="reasoning_effort passed to the chat template "
                         "(xhigh|medium|low; default: xhigh)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel requests / server slots (default: 1). Note: "
                         "each slot splits --ctx, so keep this at 1 for reasoning.")
    ap.add_argument("--no-store-reasoning", dest="store_reasoning",
                    action="store_false", default=True,
                    help="omit the raw reasoning traces from predictions.jsonl")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--startup-timeout", type=int, default=1200,
                    help="seconds to wait for the model to load (default: 1200)")
    ap.add_argument("--progress-every", type=int, default=1,
                    help="print a progress line every N completed tasks "
                         "(0 disables; default: 1, i.e. one line per task)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore previous results and overwrite predictions")
    ap.add_argument("--retry-incomplete", dest="retry_incomplete",
                    action="store_true", default=True,
                    help="on resume, re-run truncated/unparseable/errored "
                         "records instead of keeping them (default: on)")
    ap.add_argument("--no-retry-incomplete", dest="retry_incomplete",
                    action="store_false",
                    help="keep incomplete records on resume")
    args = ap.parse_args()

    if not LLAMA_SERVER.exists():
        sys.exit(f"llama-server not found at {LLAMA_SERVER}")
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")

    if args.list:
        for key, cfg in MODEL_REGISTRY.items():
            print(f"{key:22s} {GGUF_CACHE / cfg['gguf']}")
            print(f"{'':22s} server args: {cfg['server_args']}")
            print(f"{'':22s} note: {cfg['notes']}")
        return

    if args.model_file:
        model_path = Path(args.model_file).resolve()
        model_key = model_path.stem
        server_args = shlex.split(args.server_args)
        notes = "custom --model-file run"
    else:
        if args.model not in MODEL_REGISTRY:
            sys.exit(f"Unknown model {args.model!r} (see --list)")
        cfg = MODEL_REGISTRY[args.model]
        model_key, model_path = args.model, GGUF_CACHE / cfg["gguf"]
        server_args, notes = cfg["server_args"].split(), cfg["notes"]

    try:
        run_benchmark(args, model_key, model_path, server_args, notes)
    except KeyboardInterrupt:
        print("\nInterrupted; server stopped. Re-run to resume where it left off.")
        sys.exit(130)


if __name__ == "__main__":
    main()
