"""
Evaluation harness for the RAG Agent with Database Routing.

Tests the REAL pipeline components (parser -> embeddings -> router ->
retriever -> Groq generator -> faithfulness evaluator) against two
materials-science PDFs and measures whether the system returns accurate
answers.

Method (self-contained, no external ground truth needed):
  1. Parse each PDF with the app's DocumentParser.
  2. Ingest each paper into its own in-memory Qdrant collection.
  3. Use the LLM to generate reference QA pairs grounded in the paper text.
  4. For every question:
       - record what the real query router would pick (routing behaviour),
       - retrieve top-k chunks from the correct paper collection,
       - generate a grounded answer with the app's RAG prompt + model,
       - score deterministic groundedness (app's FaithfulnessEvaluator),
       - LLM-as-judge correctness vs the reference answer (1 / 0.5 / 0).
  5. Aggregate and write EVALUATION_REPORT.md.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in evaluation/)
sys.path.insert(0, str(ROOT))  # keep `rag_agent` importable when run directly
load_dotenv(ROOT / ".env")


def _load_groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


from openai import OpenAI
from qdrant_client import QdrantClient

from rag_agent.databases import add_documents, build_databases
from rag_agent.evaluator import FaithfulnessEvaluator
from rag_agent.parser import DocumentParser
from rag_agent.pipeline import MODEL, RAG_SYSTEM
from rag_agent.retriever import retrieve
from rag_agent.router import route_query

PAPERS_DIR = Path("/Volumes/Hard Disk/Laxman BTP-1/data/raw/papers")

PAPERS = [
    {
        "id": "paper1",
        "collection": "products",
        "file": "3D Study of Microstructural Influences on Retained Austenite Transformation in Q&P 1180 Steel.pdf",
        "short": "3D Q&P 1180 Retained Austenite",
    },
    {
        "id": "paper2",
        "collection": "support",
        "file": "A CNN-Based Method for Quantitative Assessment of Steel Microstructures in Welded Zones .pdf",
        "short": "CNN Steel Microstructure (Welded Zones)",
    },
]

QA_PER_PAPER = 6
TOP_K = 4
MAX_INGEST_CHUNKS = 80           # cap embedding workload (external-drive CPU limit)
MAX_CHUNK_CHARS = 2000           # truncate huge LiteParse blocks before embedding
QA_SOURCE_CHUNKS = 30            # chunks fed to the QA generator
MAX_CTX_CHARS_FOR_QA = 14000     # context budget for QA generation


@dataclass
class QAResult:
    question: str
    reference: str
    routed_db: str
    routing_reason: str
    docs_found: int
    top_score: float
    answer: str
    groundedness: float
    ground_label: str
    judge_score: float
    judge_verdict: str
    judge_reason: str


@dataclass
class PaperResult:
    short: str
    file: str
    collection: str
    engine: str
    total_chunks: int
    ingested: int
    qa: list[QAResult] = field(default_factory=list)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_paper(path: Path, groq_client: OpenAI) -> tuple[list[str], str]:
    log(f"Parsing: {path.name} ({path.stat().st_size/1e6:.1f} MB) ...")
    t0 = time.time()
    chunks, engine = DocumentParser.parse_file(str(path), path.name, groq_client=groq_client)
    log(f"  -> {len(chunks)} chunks via {engine} in {time.time()-t0:.1f}s")
    return chunks, engine


def generate_reference_qa(groq_client: OpenAI, chunks: list[str], n: int) -> list[dict]:
    """Ask the LLM to produce fact-based QA pairs grounded strictly in the text."""
    context = ""
    for c in chunks[:QA_SOURCE_CHUNKS]:
        if len(context) + len(c) > MAX_CTX_CHARS_FOR_QA:
            break
        context += c + "\n\n"

    prompt = (
        "You are building a factual QA benchmark from a scientific paper excerpt.\n"
        f"Generate exactly {n} specific, factual question/answer pairs that can be "
        "answered ONLY from the text below. Prefer concrete facts: methods used, "
        "materials/steel grade, techniques, key numeric results, and conclusions.\n"
        "Each reference answer must be a short, verifiable statement taken from the text.\n"
        'Respond with ONLY JSON: {"pairs":[{"question":"...","reference_answer":"..."}]}\n\n'
        f"TEXT:\n{context}"
    )
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    data = json.loads(resp.choices[0].message.content)
    pairs = data.get("pairs", [])[:n]
    return [p for p in pairs if p.get("question") and p.get("reference_answer")]


def generate_answer(groq_client: OpenAI, query: str, docs) -> str:
    context_str = "\n\n".join(f"[{i+1}] {d.text}" for i, d in enumerate(docs))
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def judge_answer(groq_client: OpenAI, question: str, reference: str, answer: str) -> tuple[float, str, str]:
    """LLM-as-judge: compare the RAG answer to the reference answer."""
    prompt = (
        "You are grading a RAG system answer against a reference answer.\n"
        "Return score 1.0 if the answer is factually correct and covers the reference, "
        "0.5 if partially correct or incomplete, 0.0 if incorrect / unsupported / says it "
        "cannot answer.\n"
        'Respond ONLY as JSON: {"score":1.0,"verdict":"correct|partial|incorrect","reason":"short"}\n\n'
        f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference}\n\nSYSTEM ANSWER: {answer}"
    )
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    data = json.loads(resp.choices[0].message.content)
    score = float(data.get("score", 0.0))
    return score, str(data.get("verdict", "?")), str(data.get("reason", ""))


def run() -> None:
    key = _load_groq_key()
    if not key:
        log("ERROR: No GROQ_API_KEY found. Aborting.")
        sys.exit(1)
    os.environ["GROQ_API_KEY"] = key
    log("Groq API key loaded.")

    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)

    log("Building in-memory Qdrant + FastEmbed embeddings ...")
    client, embeddings = build_databases()

    results: list[PaperResult] = []

    for spec in PAPERS:
        path = PAPERS_DIR / spec["file"]
        if not path.exists():
            log(f"MISSING FILE: {path}")
            continue

        chunks, engine = parse_paper(path, groq_client)
        if not chunks:
            log(f"  !! No text extracted from {spec['short']}; skipping.")
            results.append(PaperResult(spec["short"], spec["file"], spec["collection"], engine, 0, 0))
            continue

        to_ingest = [c[:MAX_CHUNK_CHARS] for c in chunks[:MAX_INGEST_CHUNKS]]
        log(f"  Embedding + ingesting {len(to_ingest)} chunks into '{spec['collection']}' ...")
        t_emb = time.time()
        ingested = add_documents(client, embeddings, spec["collection"], to_ingest)
        log(f"  Ingested {ingested} chunks in {time.time()-t_emb:.1f}s")

        paper_res = PaperResult(
            short=spec["short"], file=spec["file"], collection=spec["collection"],
            engine=engine, total_chunks=len(chunks), ingested=ingested,
        )

        log("  Generating reference QA pairs ...")
        try:
            qa_pairs = generate_reference_qa(groq_client, chunks, QA_PER_PAPER)
        except Exception as e:
            log(f"  QA generation failed: {e}")
            qa_pairs = []

        for idx, pair in enumerate(qa_pairs, 1):
            q = pair["question"].strip()
            ref = pair["reference_answer"].strip()
            log(f"  Q{idx}: {q[:70]}")

            routing = route_query(groq_client, q)
            docs = retrieve(q, spec["collection"], client, embeddings, top_k=TOP_K)

            if docs:
                answer = generate_answer(groq_client, q, docs)
            else:
                answer = "No relevant documents retrieved (below score threshold)."

            ev = FaithfulnessEvaluator.evaluate(answer, [d.text for d in docs])
            try:
                jscore, jverdict, jreason = judge_answer(groq_client, q, ref, answer)
            except Exception as e:
                jscore, jverdict, jreason = 0.0, "error", str(e)

            paper_res.qa.append(QAResult(
                question=q, reference=ref,
                routed_db=routing.database, routing_reason=routing.reasoning,
                docs_found=len(docs), top_score=(docs[0].score if docs else 0.0),
                answer=answer, groundedness=ev.groundedness_score, ground_label=ev.status_label,
                judge_score=jscore, judge_verdict=jverdict, judge_reason=jreason,
            ))
            time.sleep(0.4)  # gentle on rate limits

        results.append(paper_res)

    write_report(results)


def write_report(results: list[PaperResult]) -> None:
    lines: list[str] = []
    w = lines.append

    all_scores = [qa.judge_score for p in results for qa in p.qa]
    all_ground = [qa.groundedness for p in results for qa in p.qa]
    overall_acc = statistics.mean(all_scores) if all_scores else 0.0
    overall_ground = statistics.mean(all_ground) if all_ground else 0.0

    w("# RAG Agent — Accuracy Evaluation Report")
    w("")
    w(f"_Generated: {time.strftime('%Y-%m-%d %H:%M')}_  ")
    w(f"_Model: `{MODEL}` · Embeddings: `BAAI/bge-small-en-v1.5` · Retriever top-k: {TOP_K}_")
    w("")
    w("## Executive Summary")
    w("")
    verdict = (
        "ACCURATE" if overall_acc >= 0.8 else
        "MOSTLY ACCURATE" if overall_acc >= 0.6 else
        "PARTIALLY ACCURATE" if overall_acc >= 0.4 else "INACCURATE"
    )
    w(f"- **Overall answer accuracy (LLM-judge): {overall_acc*100:.0f}%** → **{verdict}**")
    w(f"- **Mean deterministic groundedness: {overall_ground*100:.0f}%**")
    w(f"- Questions evaluated: {len(all_scores)} across {len(results)} papers")
    w("")
    w("| Metric | Meaning |")
    w("|---|---|")
    w("| Answer accuracy | LLM-judge: is the answer factually correct vs a reference drawn from the paper (1 / 0.5 / 0) |")
    w("| Groundedness | App's deterministic word-overlap score between answer and retrieved chunks |")
    w("| Routed DB | Which of products/support/financial the real router chose (domain-fit signal) |")
    w("")

    for p in results:
        acc = statistics.mean([qa.judge_score for qa in p.qa]) if p.qa else 0.0
        gnd = statistics.mean([qa.groundedness for qa in p.qa]) if p.qa else 0.0
        hit = sum(1 for qa in p.qa if qa.docs_found > 0)
        w(f"## {p.short}")
        w("")
        w(f"- File: `{p.file}`")
        w(f"- Parser engine: **{p.engine}** · chunks extracted: **{p.total_chunks}** · ingested: **{p.ingested}** (collection `{p.collection}`)")
        if not p.qa:
            w(f"- ⚠️ No QA evaluated (extraction or generation failed).")
            w("")
            continue
        w(f"- **Answer accuracy: {acc*100:.0f}%** · groundedness: {gnd*100:.0f}% · retrieval hit rate: {hit}/{len(p.qa)}")
        w("")
        for i, qa in enumerate(p.qa, 1):
            w(f"### Q{i}. {qa.question}")
            w(f"- **Reference:** {qa.reference}")
            w(f"- **System answer:** {qa.answer.strip()}")
            w(f"- **Judge:** `{qa.judge_verdict}` ({qa.judge_score:.1f}) — {qa.judge_reason}")
            w(f"- Retrieval: {qa.docs_found} docs, top score `{qa.top_score:.3f}` · groundedness `{qa.groundedness:.2f}` ({qa.ground_label})")
            w(f"- Router picked: `{qa.routed_db}` — _{qa.routing_reason}_")
            w("")

    w("## Notes & Caveats")
    w("")
    w("- The app router only classifies into **products / support / financial**. These are "
      "materials-science papers, so the 'Routed DB' column is expected to be an arbitrary/best-fit "
      "bucket, not a true topical match. For this test each paper was ingested into a dedicated "
      "collection and retrieval was forced there, isolating **retrieval + generation accuracy** "
      "from the domain-router mismatch.")
    w("- Reference answers are LLM-generated from the paper text (self-consistency benchmark), then "
      "a separate LLM-judge grades the RAG answer. This measures internal accuracy/faithfulness, "
      "not correctness against external ground truth.")
    w("- Groundedness is a lexical word-overlap heuristic; treat it as a support signal, not a "
      "semantic correctness measure.")

    out = ROOT / "docs" / "EVALUATION_REPORT.md"
    out.write_text("\n".join(lines))
    log(f"Report written: {out}")
    print("\n" + "=" * 60)
    print(f"OVERALL ACCURACY: {overall_acc*100:.0f}%  |  GROUNDEDNESS: {overall_ground*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    run()
