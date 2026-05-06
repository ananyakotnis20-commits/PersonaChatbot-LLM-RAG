"""
01_ingest.py - Load, parse, and preprocess the conversations CSV.

Each CSV row = one "day" conversation.
Each row is a flat string with messages separated by \\n.
Output: data/messages.json  — flat ordered list of all messages with metadata.
"""

import json
import os
import pandas as pd
from tqdm import tqdm
import config


# ── Helpers ────────────────────────────────────────────────────────────────

def parse_row(row_text: str, day_idx: int) -> list[dict]:
    """Split a CSV row into individual messages with metadata."""
    messages = []
    raw_lines = [l.strip() for l in str(row_text).split("\\n") if l.strip()]
    for i, line in enumerate(raw_lines):
        # Detect speaker prefix  e.g. "User 1:" or "User 2:"
        speaker = "unknown"
        content = line
        if ":" in line:
            prefix, rest = line.split(":", 1)
            if len(prefix.split()) <= 3:  # short prefix = speaker tag
                speaker = prefix.strip()
                content = rest.strip()
        if content:
            messages.append({
                "global_idx": None,  # filled below
                "day_idx": day_idx,
                "local_idx": i,
                "speaker": speaker,
                "text": content,
            })
    return messages


def ingest(csv_path: str, out_path: str) -> list[dict]:
    print(f"[ingest] Loading {csv_path} ...")
    df = pd.read_csv(csv_path, header=0)
    col = df.columns[0]  # single-column CSV

    all_messages: list[dict] = []
    for day_idx, row_text in enumerate(tqdm(df[col], desc="Parsing rows")):
        msgs = parse_row(row_text, day_idx)
        all_messages.extend(msgs)

    # Assign sequential global index
    for g, msg in enumerate(all_messages):
        msg["global_idx"] = g

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=2)

    print(f"[ingest] ✓ {len(all_messages):,} messages from "
          f"{len(df):,} days → {out_path}")
    return all_messages


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    import shutil

    # Copy CSV into data/ if not already there
    if not os.path.exists(config.CSV_PATH):
        src = "conversations.csv"
        if os.path.exists(src):
            shutil.copy(src, config.CSV_PATH)
        else:
            raise FileNotFoundError(
                f"Put conversations.csv at {config.CSV_PATH} or in current dir"
            )
    ingest(config.CSV_PATH, os.path.join(config.DATA_DIR, "messages.json"))