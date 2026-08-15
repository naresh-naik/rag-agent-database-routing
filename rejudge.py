"""
Re-judge an existing evaluation results file with a (cheaper) judge model.

Used to put two runs judged by different models on equal footing: the stored
question/reference/answer triples are re-scored without touching the pipeline.

Usage:
  uv run python rejudge.py --source run1_baseline --label run1_judged8b \
      --judge-model llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import os

from openai import OpenAI

ARTIFACTS = Path(__file__).parent / "eval_artifacts"
DEFECTIVE_QIDS = {32, 33}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def llm_call(fn, attempts: int = 10):
    delay = 2.0
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            if not any(k in msg for k in ("429", "rate", "limit", "500", "502", "503", "timeout", "connection")):
                raise
            import re as _re
            wait = delay
            m = _re.search(r"try again in (\d+)m([\d.]+)s", msg)
            if m:
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 5
            else:
                m = _re.search(r"try again in ([\d.]+)s", msg)
                if m:
                    wait = float(m.group(1)) + 2
            log(f"  rate-limited, retrying in {wait:.0f}s ...")
            time.sleep(wait)
            delay = min(delay * 2, 32.0)
    raise RuntimeError(f"judge call failed: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source results label")
    ap.add_argument("--label", required=True, help="output results label")
    ap.add_argument("--judge-model", default="llama-3.1-8b-instant")
    args = ap.parse_args()

    key = os.getenv("GROQ_API_KEY", "").strip()
    src = ARTIFACTS / f"mod01_results_{args.source}.json"
    out = ARTIFACTS / f"mod01_results_{args.label}.json"
    rows = json.loads(src.read_text())
    done = {}
    if out.exists():
        done = {r["qid"]: r for r in json.loads(out.read_text())}
        log(f"Resuming: {len(done)} already re-judged")

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    results = []
    for row in rows:
        qid = row["qid"]
        if qid in done:
            results.append(done[qid])
            continue
        prompt = (
            "You are grading a RAG system's answer against a reference answer taken from "
            "the source document.\n"
            "Score 1.0 if the answer is factually correct and covers the reference "
            "(matching numbers/values exactly when the reference contains them).\n"
            "Score 0.5 if partially correct or missing a key detail.\n"
            "Score 0.0 if incorrect, contradicts the reference, is off-topic, or says it "
            "cannot answer.\n"
            "Accounting notation: parentheses mean negative, so '(123)' equals '-123'; "
            "treat these as matching.\n"
            'Respond ONLY as JSON: {"score":1.0,"verdict":"correct|partial|incorrect","reason":"short"}\n\n'
            f"QUESTION: {row['question']}\n\nREFERENCE ANSWER: {row['reference'][:300]}\n\n"
            f"SYSTEM ANSWER: {row['answer'][:600]}"
        )
        resp = llm_call(
            lambda: client.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        )
        data = json.loads(resp.choices[0].message.content)
        new = dict(row)
        new["judge_score"] = float(data.get("score", 0.0))
        new["judge_verdict"] = str(data.get("verdict", "?"))
        new["judge_reason"] = str(data.get("reason", ""))
        results.append(new)
        log(f"  Q{qid:>3} {new['judge_verdict']} ({new['judge_score']:.1f}) {row['question'][:55]}")
        out.write_text(json.dumps(results, indent=2))
        time.sleep(0.4)

    valid = [r for r in results if r["qid"] not in DEFECTIVE_QIDS]
    acc = statistics.mean(r["judge_score"] for r in valid)
    correct = sum(1 for r in valid if r["judge_score"] == 1.0)
    partial = sum(1 for r in valid if r["judge_score"] == 0.5)
    incorrect = sum(1 for r in valid if r["judge_score"] == 0.0)
    print("=" * 60)
    print(f"RE-JUDGED ({args.judge_model}) accuracy: {acc*100:.1f}%  "
          f"({correct} correct / {partial} partial / {incorrect} incorrect of {len(valid)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
