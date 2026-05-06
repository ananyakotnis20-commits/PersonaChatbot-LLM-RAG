"""
02_build_rag.py - Build all checkpoints and populate FAISS indexes.

Produces:
  checkpoints/topic_checkpoints.json      — topic segments + summaries
  checkpoints/message_checkpoints.json    — 100-msg chunk summaries
  vector_store/summaries.faiss + _meta    — embedded summaries
  vector_store/chunks.faiss    + _meta    — embedded raw message windows
"""

from __future__ import annotations
import json
import os
import time
import numpy as np
from tqdm import tqdm

import config
import llm_client
from embedder import EmbedderFAISS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer



def load_messages() -> list[dict]:
    path = os.path.join(config.DATA_DIR, "messages.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def window_text(messages: list[dict], start: int, end: int) -> str:
    """Render messages[start:end] as a readable dialogue string."""
    lines = [f"{m['speaker']}: {m['text']}" for m in messages[start:end]]
    return "\n".join(lines)


# Part A – Topic Checkpoints via Sliding Window + Embedding comparison


def detect_topic_change(prev_text: str, curr_text: str, embedder: EmbedderFAISS) -> bool:
    """Use cosine distance between embeddings — zero API calls."""
    distance = embedder.cosine_distance(prev_text, curr_text)
    return distance > config.TOPIC_CHANGE_THRESHOLD


def summarise(text: str) -> str:
    """Extractive summarization — no API calls."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        sentences = summarizer(parser.document, 4)  # 4 key sentences
        return " ".join(str(s) for s in sentences)
    except Exception:
        return text[:300]




def build_topic_checkpoints(messages: list[dict]) -> list[dict]:
    """
    Slide a window over all messages.
    When the embedding distance exceeds threshold → close the current segment,
    summarise it, start a new one.
    """
    from embedder import EmbedderFAISS
    embedder = EmbedderFAISS("summaries").load_or_create()

    W = config.SLIDING_WINDOW_SIZE
    OV = config.TOPIC_OVERLAP

    checkpoints: list[dict] = []
    topic_id = 0
    seg_start = 0  # global message index where current topic starts
    prev_window_text: str | None = None

    print(f"\n[build_rag] Building topic checkpoints "
          f"(window={W}, overlap={OV}) over {len(messages):,} messages ...")

    i = 0
    pbar = tqdm(total=len(messages), unit="msg")

    while i < len(messages):
        win_end = min(i + W, len(messages))
        curr_text = window_text(messages, i, win_end)

        if prev_window_text is not None:
            # Pass the local embedder to the detection function
            changed = detect_topic_change(prev_window_text, curr_text, embedder)
            if changed:
                # Close current topic segment
                seg_text = window_text(messages, seg_start, i)
                summary = summarise(seg_text)
                checkpoints.append({
                    "type": "topic",
                    "topic_id": topic_id,
                    "start_msg": seg_start,
                    "end_msg": i - 1,
                    "msg_count": i - seg_start,
                    "summary": summary,
                })
                print(f"\n  → Topic {topic_id} closed "
                      f"(msgs {seg_start}–{i - 1}): {summary[:60]}...")
                topic_id += 1
                seg_start = i

        prev_window_text = curr_text
        step = W - OV
        pbar.update(step)
        i += step

    pbar.close()

    # Close the last segment
    seg_text = window_text(messages, seg_start, len(messages))
    summary = summarise(seg_text)
    checkpoints.append({
        "type": "topic",
        "topic_id": topic_id,
        "start_msg": seg_start,
        "end_msg": len(messages) - 1,
        "msg_count": len(messages) - seg_start,
        "summary": summary,
    })
    print(f"\n  → Topic {topic_id} closed (last segment)")
    print(f"[build_rag] ✓ {len(checkpoints)} topic checkpoints created")
    return checkpoints


# Part B – 100-Message Checkpoints


def build_message_checkpoints(messages: list[dict]) -> list[dict]:
    """Create one summary checkpoint for every 100 messages."""
    N = config.MSG_CHUNK_SIZE
    checkpoints: list[dict] = []
    total = len(messages)

    print(f"\n[build_rag] Building {total // N + 1} ×100-msg checkpoints ...")

    for start in tqdm(range(0, total, N), desc="100-msg chunks"):
        end = min(start + N, total)
        text = window_text(messages, start, end)
        summary = summarise(text)
        checkpoints.append({
            "type": "message_chunk",
            "chunk_id": start // N,
            "start_msg": start,
            "end_msg": end - 1,
            "msg_count": end - start,
            "summary": summary,
        })

    print(f"[build_rag] ✓ {len(checkpoints)} message checkpoints created")
    return checkpoints


# Part C – Build FAISS vector indexes


def build_vector_indexes(
        topic_cps: list[dict],
        message_cps: list[dict],
        messages: list[dict],
):
    print("\n[build_rag] Building FAISS vector indexes ...")

    # ── Summaries index (topic + message checkpoints)
    summary_store = EmbedderFAISS("summaries").load_or_create()
    all_cps = topic_cps + message_cps
    texts = [cp["summary"] for cp in all_cps]
    metas = [{k: v for k, v in cp.items()} for cp in all_cps]

    BATCH = 256
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding summaries"):
        summary_store.add(texts[i:i + BATCH], metas[i:i + BATCH])
    summary_store.save()

    # ── Raw chunks index (sliding windows of raw messages)
    chunk_store = EmbedderFAISS("chunks").load_or_create()
    W = config.SLIDING_WINDOW_SIZE * 2  # wider windows for raw retrieval
    chunk_texts, chunk_metas = [], []

    for start in range(0, len(messages), W // 2):  # 50% overlap
        end = min(start + W, len(messages))
        text = window_text(messages, start, end)
        chunk_texts.append(text)
        chunk_metas.append({
            "type": "raw_chunk",
            "start_msg": start,
            "end_msg": end - 1,
            "text": text[:1500],  # store truncated for context use
        })

    for i in tqdm(range(0, len(chunk_texts), BATCH), desc="Embedding raw chunks"):
        chunk_store.add(chunk_texts[i:i + BATCH], chunk_metas[i:i + BATCH])
    chunk_store.save()

    print(f"[build_rag] ✓ Indexes: "
          f"{summary_store.size} summaries, {chunk_store.size} raw chunks")



def main():
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)

    messages = load_messages()

    # ── Topic checkpoints ──────────────────────────────────────────────────
    topic_path = os.path.join(config.CHECKPOINTS_DIR, "topic_checkpoints.json")
    if os.path.exists(topic_path):
        print(f"[build_rag] Loading existing topic checkpoints from {topic_path}")
        with open(topic_path) as f:
            topic_cps = json.load(f)
    else:
        topic_cps = build_topic_checkpoints(messages)
        with open(topic_path, "w") as f:
            json.dump(topic_cps, f, indent=2)

    #100-msg checkpoints
    msg_path = os.path.join(config.CHECKPOINTS_DIR, "message_checkpoints.json")
    if os.path.exists(msg_path):
        print(f"[build_rag] Loading existing msg checkpoints from {msg_path}")
        with open(msg_path) as f:
            msg_cps = json.load(f)
    else:
        msg_cps = build_message_checkpoints(messages)
        with open(msg_path, "w") as f:
            json.dump(msg_cps, f, indent=2)

    #Vector indexes
    if not os.path.exists(
            os.path.join(config.VECTOR_STORE_DIR, "summaries.faiss")
    ):
        build_vector_indexes(topic_cps, msg_cps, messages)
    else:
        print("[build_rag] Vector indexes already exist – skipping rebuild.")
        print("  (Delete vector_store/ to force a rebuild)")

    print("\n[build_rag] ✅ All checkpoints and indexes ready.")


if __name__ == "__main__":
    main()