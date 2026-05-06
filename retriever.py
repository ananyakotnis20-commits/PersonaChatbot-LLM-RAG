from __future__ import annotations
import json
import os

import config
from embedder import EmbedderFAISS


class Retriever:
    """
    Loads both FAISS indexes and exposes a single `retrieve(query)` call
    that returns combined, deduplicated context.
    """

    def __init__(self):
        self._summaries: EmbedderFAISS | None = None
        self._chunks: EmbedderFAISS | None = None

    def load(self) -> "Retriever":
        print("[retriever] Loading FAISS indexes ...")
        self._summaries = EmbedderFAISS("summaries").load_or_create()
        self._chunks = EmbedderFAISS("chunks").load_or_create()
        return self

    #Public API

    def retrieve(
            self,
            query: str,
            top_k_summaries: int = config.TOP_K_SUMMARIES,
            top_k_chunks: int = config.TOP_K_CHUNKS,
    ) -> dict:
        """
        Returns:
          {
            "summaries": [...],   # list of summary dicts with _score
            "chunks":    [...],   # list of raw chunk dicts with _score
            "context_text": str,  # ready-to-use context for the LLM
          }
        """
        summaries = self._summaries.query(query, top_k=top_k_summaries)
        chunks = self._chunks.query(query, top_k=top_k_chunks)

        context_text = self._build_context(query, summaries, chunks)
        return {
            "summaries": summaries,
            "chunks": chunks,
            "context_text": context_text,
        }

    #Helpers

    @staticmethod
    def _build_context(
            query: str,
            summaries: list[dict],
            chunks: list[dict],
    ) -> str:
        parts = [f"## Query\n{query}\n"]

        if summaries:
            parts.append("## Relevant Conversation Summaries")
            for i, s in enumerate(summaries, 1):
                label = (
                    f"Topic {s.get('topic_id', '?')} "
                    f"(msgs {s.get('start_msg', '?')}–{s.get('end_msg', '?')})"
                    if s.get("type") == "topic"
                    else f"Chunk {s.get('chunk_id', '?')} "
                         f"(msgs {s.get('start_msg', '?')}–{s.get('end_msg', '?')})"
                )
                parts.append(f"**{i}. {label}** [score={s['_score']:.3f}]")
                parts.append(s.get("summary", ""))
                parts.append("")

        if chunks:
            parts.append("## Relevant Raw Conversation Excerpts")
            for i, c in enumerate(chunks, 1):
                parts.append(
                    f"**Excerpt {i}** "
                    f"(msgs {c.get('start_msg', '?')}–{c.get('end_msg', '?')}) "
                    f"[score={c['_score']:.3f}]"
                )
                parts.append(c.get("text", "")[:600])
                parts.append("")

        full = "\n".join(parts)
        # Hard-cap token approximation (4 chars ≈ 1 token)
        max_chars = config.MAX_CONTEXT_TOKENS * 4
        return full[:max_chars]