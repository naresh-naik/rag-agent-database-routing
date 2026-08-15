"""
Zero-API retrieval coverage check: baseline (fragmented) chunks vs consolidated chunks.

For every benchmark question, retrieves top-4 from both indexes and checks whether
the reference answer's content (exact number or key words) is present. This isolates
retrieval quality from generation and costs no LLM tokens.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from evaluate_pdf_module import resolve_pdf_path
from rag_agent.embeddings import FastEmbeddings
from rag_agent.parser import DocumentParser, consolidate_chunks

ROOT = Path(__file__).parent
QUESTIONS_FILE = ROOT / "eval_artifacts" / "mod01_questions.json"


def make_index(client: QdrantClient, texts: list[str], embeddings) -> None:
    client.create_collection("t", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    vecs = embeddings.embed_documents(texts)
    client.upsert(
        "t",
        points=[
            PointStruct(id=i, vector=v, payload={"text": t})
            for i, (t, v) in enumerate(zip(texts, vecs))
        ],
    )


def covered(client: QdrantClient, embeddings, q: str, ref: str) -> tuple[bool, float]:
    hits = client.query_points("t", query=embeddings.embed_query(q), limit=4).points
    ctx = " ".join(h.payload["text"] for h in hits).lower()

    refn = ref.strip().strip("()").replace(",", "")
    if re.fullmatch(r"[\d.]+", refn):  # numeric reference -> exact number must appear
        ok = any(refn in h.payload["text"].replace(",", "") for h in hits)
    else:  # textual reference -> majority of key content words present
        words = [w for w in re.findall(r"\b\w{5,}\b", ref.lower())][:6]
        ok = (sum(1 for w in words if w in ctx) / len(words) >= 0.6) if words else True
    return ok, hits[0].score if hits else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None, help="path to the benchmark PDF (default: auto-resolved)")
    args = ap.parse_args()
    qa_pairs = json.loads(QUESTIONS_FILE.read_text())
    pdf_path = resolve_pdf_path(args.pdf)
    chunks, engine = DocumentParser.parse_file(str(pdf_path), pdf_path.name)
    consolidated = consolidate_chunks(chunks)
    print(f"parser: {engine} | chunks: baseline={len(chunks)} consolidated={len(consolidated)}")

    emb = FastEmbeddings()
    idx_base = QdrantClient(":memory:")
    make_index(idx_base, chunks, emb)
    idx_cons = QdrantClient(":memory:")
    make_index(idx_cons, consolidated, emb)

    base_ok = cons_ok = 0
    flips: list[tuple[int, str]] = []
    still_missing: list[tuple[int, str, str]] = []

    for i, pair in enumerate(qa_pairs, 1):
        q, ref = pair["question"], pair["reference_answer"]
        b, _ = covered(idx_base, emb, q, ref)
        c, score = covered(idx_cons, emb, q, ref)
        base_ok += b
        cons_ok += c
        if not b and c:
            flips.append((i, q[:64]))
        if not c:
            still_missing.append((i, q[:64], ref[:40]))

    n = len(qa_pairs)
    print(f"\nRETRIEVAL COVERAGE (reference answer present in top-4 retrieved chunks):")
    print(f"  baseline fragments:   {base_ok}/{n} = {base_ok/n*100:.0f}%")
    print(f"  consolidated chunks:  {cons_ok}/{n} = {cons_ok/n*100:.0f}%")
    print(f"\nFixed by consolidation ({len(flips)}):")
    for qid, q in flips:
        print(f"  Q{qid}: {q}")
    print(f"\nStill missing after consolidation ({len(still_missing)}):")
    for qid, q, r in still_missing:
        print(f"  Q{qid}: {q} | ref: {r}")


if __name__ == "__main__":
    main()
