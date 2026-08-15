"""
Second-pass repair: re-answer only the questions a previous run scored < 1.0,
using the latest RAG_SYSTEM prompt, then re-judge with the same judge model.

Keeps every passing answer untouched, so the result isolates the prompt fix.

Usage:
  uv run python fix_failed.py --source run2_improved_8b --label run3_fixed_8b \
      --gen-model llama-3.1-8b-instant --judge-model llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in evaluation/)
sys.path.insert(0, str(ROOT))  # keep `rag_agent` importable when run directly
load_dotenv(ROOT / ".env")

import os

from openai import OpenAI

from evaluate_pdf_module import (
    ARTIFACTS, COLLECTION, DEFECTIVE_QIDS, TOP_K, generate_answer, judge_answer, log,
    resolve_pdf_path,
)
from rag_agent.databases import add_documents, build_databases
from rag_agent.evaluator import FaithfulnessEvaluator
from rag_agent.parser import DocumentParser
from rag_agent.retriever import retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gen-model", default="llama-3.1-8b-instant")
    ap.add_argument("--judge-model", default="llama-3.1-8b-instant")
    ap.add_argument("--pdf", default=None, help="path to the benchmark PDF (default: auto-resolved)")
    args = ap.parse_args()

    key = os.getenv("GROQ_API_KEY", "").strip()
    groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)

    rows = json.loads((ARTIFACTS / f"mod01_results_{args.source}.json").read_text())
    out_file = ARTIFACTS / f"mod01_results_{args.label}.json"
    done: dict[int, dict] = {}
    if out_file.exists():
        done = {r["qid"]: r for r in json.loads(out_file.read_text())}
        log(f"Resuming: {len(done)} rows already in {out_file.name}")

    log("Building index (consolidated hybrid) ...")
    client, embeddings = build_databases()
    pdf_path = resolve_pdf_path(args.pdf)
    chunks, engine = DocumentParser.parse_file(str(pdf_path), pdf_path.name)
    ingested = add_documents(client, embeddings, COLLECTION, chunks)
    log(f"  {len(chunks)} parser chunks -> {ingested} indexed chunks ({engine})")

    targets = [r for r in rows if r["judge_score"] < 1.0 and r["qid"] not in DEFECTIVE_QIDS]
    log(f"Re-answering {len(targets)} previously-failed questions ...")

    results = list(rows)
    by_qid = {r["qid"]: i for i, r in enumerate(results)}

    for row in targets:
        qid = row["qid"]
        if qid in done:
            results[by_qid[qid]] = done[qid]
            continue
        docs = retrieve(row["question"], COLLECTION, client, embeddings, top_k=TOP_K)
        answer = generate_answer(groq, row["question"], docs, gen_model=args.gen_model)
        time.sleep(1.0)
        ev = FaithfulnessEvaluator.evaluate(answer, [d.text for d in docs])
        jscore, jverdict, jreason = judge_answer(
            groq, row["question"], row["reference"], answer, judge_model=args.judge_model
        )
        new = dict(row)
        new.update(
            docs_found=len(docs),
            top_score=round(docs[0].score, 4) if docs else 0.0,
            answer=answer,
            groundedness=ev.groundedness_score,
            ground_label=ev.status_label,
            judge_score=jscore, judge_verdict=jverdict, judge_reason=jreason,
        )
        results[by_qid[qid]] = new
        done[qid] = new
        out_file.write_text(json.dumps(results, indent=2))
        arrow = "FIXED" if jscore == 1.0 else ("same" if jscore == row["judge_score"] else f"{row['judge_score']:.1f}->{jscore:.1f}")
        log(f"  Q{qid:>3} [{jverdict} {jscore:.1f}] {arrow} | {row['question'][:60]}")
        time.sleep(1.0)

    valid = [r for r in results if r["qid"] not in DEFECTIVE_QIDS]
    acc = statistics.mean(r["judge_score"] for r in valid)
    c = sum(1 for r in valid if r["judge_score"] == 1.0)
    p = sum(1 for r in valid if r["judge_score"] == 0.5)
    i = sum(1 for r in valid if r["judge_score"] == 0.0)
    print("=" * 64)
    print(f"FINAL ({args.label}) accuracy: {acc*100:.1f}%  "
          f"({c} correct / {p} partial / {i} incorrect of {len(valid)})")
    print("=" * 64)


if __name__ == "__main__":
    main()
