"""
part1_drift/persona_drift.py

Adaptive Persona Engine
Detects mood/tone drift across conversations and identifies trigger words.

Works with:
- CSV datasets where each row is a conversation
- JSON datasets with {"messages": [...]}
- JSON datasets with [...]

Author: Fixed + generalized version
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────

analyzer = SentimentIntensityAnalyzer()

tfidf = TfidfVectorizer(
    max_features=500,
    stop_words="english"
)

# ─────────────────────────────────────────────────────────────────────────────
# Score helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_scores(texts: list[str]) -> tuple[float, float]:
    """
    Returns:
        (formality, valence)

    Formality:
        Higher = more formal

    Valence:
        Sentiment polarity from VADER
        Range [-1, 1]
    """

    if not texts:
        return 0.5, 0.0

    informal_words = {
        "i'm", "lol", "haha", "u", "r",
        "can't", "won't",
        "gonna", "wanna", "gotta",
        "idk", "omg", "tbh", "brb"
    }

    tokens = " ".join(texts).lower().split()

    if tokens:
        informal_count = sum(
            1 for t in tokens
            if t in informal_words
        )

        formality = 1.0 - (informal_count / len(tokens))
    else:
        formality = 0.5

    valence = sum(
        analyzer.polarity_scores(t)["compound"]
        for t in texts
    ) / len(texts)

    return formality, valence

# ─────────────────────────────────────────────────────────────────────────────
# Persona labels
# ─────────────────────────────────────────────────────────────────────────────

def label_persona(f: float, v: float) -> str:
    """
    Convert (formality, valence) → human-readable label
    """

    if f > 0.85 and v > 0.1:
        return "curious & formal"

    if f < 0.55 and v < -0.1:
        return "casual & frustrated"

    if f < 0.65 and v > 0.2:
        return "playful"

    if v > 0.3:
        return "enthusiastic"

    if v < -0.3:
        return "distressed"

    return "neutral"

# ─────────────────────────────────────────────────────────────────────────────
# Trigger detector
# ─────────────────────────────────────────────────────────────────────────────

def find_trigger(prev_msgs: list[str], curr_msgs: list[str]) -> list[str]:
    """
    Finds the words that increased most between
    previous and current conversations.
    """

    if not prev_msgs or not curr_msgs:
        return []

    corpus = [
        " ".join(prev_msgs),
        " ".join(curr_msgs)
    ]

    try:
        matrix = tfidf.fit_transform(corpus).toarray()
    except ValueError:
        return []

    if matrix.shape[0] < 2:
        return []

    diff = matrix[1] - matrix[0]

    feature_names = tfidf.get_feature_names_out()

    top_indices = np.argsort(diff)[-3:]

    triggers = [
        feature_names[i]
        for i in top_indices
        if diff[i] > 0.05
    ]

    return triggers

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_dataset(path: str) -> dict[int, list[str]]:
    """
    Loads CSV dataset.
    Dynamically splits raw dialogue lines into a fixed number of
    chronological blocks to prevent terminal overflow.
    """
    df = pd.read_csv(path)

    if df.empty:
        return {}

    # Automatically identify the text column (assumes the first column)
    text_column = df.columns[0]

    # Filter out empty or NaN rows first
    valid_rows = df[df[text_column].astype(str).str.strip().str.lower() != "nan"]
    valid_rows = valid_rows[valid_rows[text_column].astype(str).str.strip() != ""]

    total_messages = len(valid_rows)
    if total_messages == 0:
        return {}

    # Define how many total "days"/steps you want to display in your terminal
    TARGET_TOTAL_DAYS = 30

    # Calculate how many messages should go into each chunk
    chunk_size = max(1, total_messages // TARGET_TOTAL_DAYS)

    days = defaultdict(list)

    for idx, (_, row) in enumerate(valid_rows.iterrows()):
        text = str(row[text_column]).strip()

        # Calculate which block this message belongs to (1 to 30)
        day = (idx // chunk_size) + 1

        # Cap it at TARGET_TOTAL_DAYS in case of rounding
        if day > TARGET_TOTAL_DAYS:
            day = TARGET_TOTAL_DAYS

        days[day].append(text)

    return days


def load_json_dataset(path: str) -> dict[int, list[str]]:
    """
    Loads JSON dataset.
    Supports raw list structures and chunks messages into a
    fixed timeline to prevent massive terminal spam.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        messages = data.get("messages", [])
    elif isinstance(data, list):
        messages = data
    else:
        print("[ERROR] Unsupported JSON structure.")
        return {}

    # Filter down to valid messages that actually contain text
        # Filter down to valid messages that actually contain text
        valid_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            text = msg.get("text") or msg.get("message") or msg.get("content") or msg.get("summary") or ""
            text = str(text).strip()

            # SKIP API ERROR STRINGS
            if "Summary failed" in text or "404" in text:
                continue

            if text:
                explicit_day = msg.get("day") or msg.get("date")
                valid_messages.append((explicit_day, text))

    if not valid_messages:
        return {}

    days = defaultdict(list)
    total_msgs = len(valid_messages)

    # Target 30 timeline stages maximum
    TARGET_TOTAL_DAYS = 30
    chunk_size = max(1, total_msgs // TARGET_TOTAL_DAYS)

    for idx, (explicit_day, text) in enumerate(valid_messages):
        if explicit_day:
            # If the JSON object contains a specific day label, respect it
            day = explicit_day
        else:
            # Fallback: Smoothly distribute across 30 blocks
            day = (idx // chunk_size) + 1
            if day > TARGET_TOTAL_DAYS:
                day = TARGET_TOTAL_DAYS

        days[day].append(text)

    return days



# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_drift() -> list[dict]:
    """
    Main persona drift analysis pipeline.
    """

    if not os.path.exists(config.message_path):

        print(
            f"[ERROR] Dataset not found:\n"
            f"    {config.message_path}"
        )

        return []

    ext = os.path.splitext(config.message_path)[1].lower()

    # ─────────────────────────────────────────
    # Load dataset
    # ─────────────────────────────────────────

    if ext == ".csv":

        days = load_csv_dataset(config.message_path)

    elif ext == ".json":

        days = load_json_dataset(config.message_path)

    else:

        print(
            "[ERROR] Unsupported file type.\n"
            "Use CSV or JSON."
        )

        return []

    if not days:

        print("[WARN] No usable conversations found.")
        return []

    # ─────────────────────────────────────────
    # Drift analysis
    # ─────────────────────────────────────────

    timeline = []

    prev_label = None

    sorted_days = sorted(days.keys())

    for idx, day in enumerate(sorted_days):

        texts = days[day]

        f, v = get_scores(texts)

        lbl = label_persona(f, v)

        drift = (
            prev_label is not None
            and lbl != prev_label
        )

        entry = {
            "day": day,
            "label": lbl,
            "drift": drift,
            "formality": round(f, 3),
            "valence": round(v, 3),
        }

        # ─────────────────────────────────────
        # Trigger detection
        # ─────────────────────────────────────

        if drift:

            prev_day = sorted_days[idx - 1]

            triggers = find_trigger(
                days[prev_day],
                days[day]
            )

            entry["prev_label"] = prev_label
            entry["trigger"] = triggers

        timeline.append(entry)

        prev_label = lbl

        # ─────────────────────────────────────
        # Pretty printing
        # ─────────────────────────────────────

        drift_tag = " ⚡ DRIFT!" if drift else ""

        trigger_str = ""

        if drift and entry.get("trigger"):

            trigger_str = (
                f" trigger → {entry['trigger']}"
            )

        print(
            f"Day {str(day):>3} │ "
            f"{lbl:<25} "
            f"(formality={f:.2f}, "
            f"valence={v:+.2f})"
            f"{drift_tag}"
            f"{trigger_str}"
        )

    return timeline

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    result = analyze_drift()

    print(
        f"\n[Done] {len(result)} day(s) analysed."
    )