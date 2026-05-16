"""
part3_rag/resolver.py
Conflict Resolution in RAG — retrieves sister-related chunks, ranks by
recency + emotional weight, flags contradictions, returns merged answer.
"""
import math
import sqlite3
import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Reranker ──────────────────────────────────────────────────────────────────

def rerank(chunks: list[dict], max_day: int) -> list[dict]:
    """
    Score each chunk by a weighted combination of:
        0.4 × relevance  (BM25 rank, negated so higher = better)
        0.4 × recency    (exponential decay by age in days)
        0.2 × emotional  (absolute VADER valence — strong emotion = memorable)

    FIX 1: Original mutated the caller's dicts in-place (c["score"] = …).
            If the same chunks list was reused across queries, stale scores
            from the previous query would corrupt the next ranking.
            Now we work on shallow copies.

    FIX 2: Original never validated that required keys exist.
            Added .get() with sensible defaults.
    """
    scored = []
    for raw_c in chunks:
        c = dict(raw_c)                      # shallow copy — don't mutate caller

        age       = max_day - c.get("day", max_day)
        recency   = math.exp(-0.1 * age)

        emotional = abs(c.get("valence", 0.0))

        # BM25 from SQLite FTS5 rank() is always ≤ 0 (less negative = better).
        # Negate so "better match" → higher positive score.
        bm25_raw  = c.get("bm25", 0.0)
        relevance = -bm25_raw                # now: higher = more relevant

        c["score"] = (0.4 * relevance) + (0.4 * recency) + (0.2 * emotional)
        scored.append(c)

    return sorted(scored, key=lambda x: x["score"], reverse=True)


# ── Contradiction detector ────────────────────────────────────────────────────

def detect_contradictions(ranked: list[dict]) -> list[str]:
    """
    Compare every pair of top chunks.
    A contradiction is flagged when two chunks have opposite valence sign
    (one clearly positive, one clearly negative) — suggesting the user
    mentioned the same entity in conflicting emotional contexts.

    Returns a list of human-readable contradiction strings.
    """
    flags: list[str] = []
    top   = ranked[:3]                       # only compare top-3

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            vi = top[i].get("valence", 0.0)
            vj = top[j].get("valence", 0.0)

            # Threshold: only flag when both valences are non-trivial
            if abs(vi) > 0.15 and abs(vj) > 0.15 and (vi * vj < 0):
                flags.append(
                    f"Day {top[i]['day']} context (valence={vi:+.2f}) "
                    f"conflicts with Day {top[j]['day']} context (valence={vj:+.2f})"
                )

    return flags


# ── SQLite retrieval ──────────────────────────────────────────────────────────

def _fetch_from_sqlite(query: str, db_path: str) -> list[dict]:
    """
    Query the SQLite FTS5 table for chunks matching `query`.
    Returns a list of raw chunk dicts with keys:
        text, day, valence, bm25

    FIX 3: Original resolve_conflict was a single `pass` — nothing was
            implemented at all. This is the full retrieval implementation.
    """
    if not os.path.exists(db_path):
        return []

    conn    = sqlite3.connect(db_path)
    cursor  = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT text, day, valence, rank AS bm25
            FROM   messages_fts
            WHERE  messages_fts MATCH ?
            ORDER  BY rank          -- FTS5 rank: lower (more negative) = better
            LIMIT  10
            """,
            (query,),
        )
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        # Table doesn't exist yet — fall back to empty
        rows = []
    finally:
        conn.close()

    return [
        {"text": r[0], "day": r[1], "valence": r[2], "bm25": r[3]}
        for r in rows
    ]


def _fetch_from_pkl(query: str, top_k: int = 10) -> list[dict]:
    """
    Fallback retriever using the pre-built chunks_meta.pkl + FAISS index.
    Used when memory.db does not exist.
    """
    import pickle
    import numpy as np

    meta_path  = os.path.join(config.VECTOR_STORE_DIR, "chunks_meta.pkl")
    index_path = os.path.join(config.VECTOR_STORE_DIR, "chunks.faiss")

    if not os.path.exists(meta_path) or not os.path.exists(index_path):
        return []

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        model  = SentenceTransformer(config.MINILM_PATH)
        index  = faiss.read_index(index_path)

        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)          # list of dicts

        q_vec = model.encode([query], show_progress_bar=False).astype("float32")
        _, idxs = index.search(q_vec, top_k)

        results = []
        for i in idxs[0]:
            if 0 <= i < len(meta):
                chunk = dict(meta[i])
                chunk.setdefault("valence", 0.0)
                # FAISS doesn't give BM25; use distance proxy (set to 0)
                chunk.setdefault("bm25",    0.0)
                results.append(chunk)
        return results

    except Exception as e:
        print(f"[resolver] FAISS fallback failed: {e}")
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def resolve_conflict(query: str, db_path: str = "memory.db") -> dict[str, Any]:
    """
    Full implementation of Part 3:

    1. Retrieve chunks from SQLite FTS5 (falls back to FAISS pkl).
    2. Rerank by recency + emotional weight + relevance.
    3. Detect contradictions across top chunks.
    4. Return a merged coherent answer with contradiction notices.

    FIX 4: Was entirely `pass`.  Now fully implemented.

    Returns:
        {
          "answer":          str   – merged text from top chunks,
          "contradictions":  list  – any flagged conflicts,
          "sources":         list  – day + score for each used chunk,
        }
    """
    # ── 1. Retrieve ──────────────────────────────────────────────────────────
    chunks = _fetch_from_sqlite(query, db_path)
    if not chunks:
        chunks = _fetch_from_pkl(query)

    if not chunks:
        return {
            "answer":         "No relevant memories found for that query.",
            "contradictions": [],
            "sources":        [],
        }

    # ── 2. Rerank ────────────────────────────────────────────────────────────
    max_day = max(c.get("day", 0) for c in chunks)
    ranked  = rerank(chunks, max_day)
    top     = ranked[:3]

    # ── 3. Detect contradictions ─────────────────────────────────────────────
    contradictions = detect_contradictions(ranked)

    # ── 4. Build merged answer ───────────────────────────────────────────────
    parts = []
    for c in top:
        parts.append(f"[Day {c['day']}] {c['text']}")

    answer = "\n\n".join(parts)

    if contradictions:
        answer += (
            "\n\n⚠ Note — contradictory contexts detected:\n  • "
            + "\n  • ".join(contradictions)
        )

    sources = [
        {"day": c["day"], "score": round(c["score"], 4)}
        for c in top
    ]

    return {
        "answer":         answer,
        "contradictions": contradictions,
        "sources":        sources,
    }


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── RAG Conflict Resolver Smoke Test ──\n")

    # Synthetic chunks to test without a real DB
    synthetic: list[dict] = [
        {"text": "I had a great time with my sister at the park.",    "day": 1,  "valence":  0.8,  "bm25": -1.2},
        {"text": "My sister and I had a huge argument today.",        "day": 4,  "valence": -0.7,  "bm25": -0.9},
        {"text": "Talked to my sister — she got a new job!",          "day": 7,  "valence":  0.6,  "bm25": -1.5},
    ]

    max_day = max(c["day"] for c in synthetic)
    ranked  = rerank(synthetic, max_day)

    print("Ranked chunks:")
    for c in ranked:
        print(f"  Day {c['day']:>2}  score={c['score']:.4f}  valence={c['valence']:+.2f}  \"{c['text'][:55]}\"")

    contradictions = detect_contradictions(ranked)
    print("\nContradictions detected:")
    for con in contradictions:
        print(f"  ⚠ {con}")

    if not contradictions:
        print("  None")

    print("\nMerged answer:")
    result = {
        "answer":         "\n".join(f"[Day {c['day']}] {c['text']}" for c in ranked[:3]),
        "contradictions": contradictions,
        "sources":        [{"day": c["day"], "score": round(c["score"], 4)} for c in ranked[:3]],
    }
    print(result["answer"])
    print(f"\nSources: {result['sources']}")