"""
03_extract_persona.py - One-time pass to extract a structured user persona
from all topic summaries.

Output: persona.json
"""

from __future__ import annotations
import json
import os
import time

from tqdm import tqdm
import config
import llm_client



EXTRACT_SYSTEM = """You are a psychologist and behavioural analyst.
Extract factual, evidence-based insights about User 1 from conversation
summaries. Only include what is explicitly mentioned or strongly implied.
Never fabricate or guess. Return ONLY valid JSON — no preamble, no markdown."""

EXTRACT_PROMPT = """Analyse these conversation summaries about User 1.
Extract what you can infer about them. Return this exact JSON structure
(use null for unknown fields, empty arrays [] when nothing found):

{{
  "habits": {{
    "sleep":    [],
    "food":     [],
    "exercise": [],
    "work":     [],
    "hobbies":  [],
    "other":    []
  }},
  "personal_facts": {{
    "occupation":  null,
    "location":    null,
    "family":      [],
    "relationships": [],
    "life_events": [],
    "interests":   []
  }},
  "personality_traits": [],
  "communication_style": {{
    "tone":          null,
    "message_length": null,
    "emoji_usage":   null,
    "humour":        null,
    "formality":     null,
    "notable_patterns": []
  }},
  "evidence_quotes": []
}}

Summaries:
{summaries}"""

MERGE_SYSTEM = """You are merging multiple partial persona JSON objects
into one final consolidated persona. Deduplicate, resolve conflicts by
taking the most specific/frequent value. Return ONLY valid JSON."""

MERGE_PROMPT = """Merge these partial persona objects into one final persona.
Keep the same structure. Combine arrays, pick best scalar values.

Partials:
{partials}"""


def safe_json_parse(text: str) -> dict | None:
    """Try to parse LLM output as JSON, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_partial(summaries_batch: list[str]) -> dict | None:
    combined = "\n\n---\n\n".join(summaries_batch)
    time.sleep(4)        # pause to avoid rate limits
    raw = llm_client.complete(
        EXTRACT_PROMPT.format(summaries=combined[:3500]),
        system=EXTRACT_SYSTEM,
        max_tokens=800,
    )
    return safe_json_parse(raw)


def merge_partials(partials: list[dict]) -> dict:
    partials_json = json.dumps(partials, indent=2)[:4000]
    time.sleep(4)        # pause to avoid rate limits
    raw = llm_client.complete(
        MERGE_PROMPT.format(partials=partials_json),
        system=MERGE_SYSTEM,
        max_tokens=1000,
    )
    merged = safe_json_parse(raw)
    if merged is None:
        return partials[0] if partials else {}
    return merged



def main():
    topic_path = os.path.join(config.CHECKPOINTS_DIR, "topic_checkpoints.json")
    if not os.path.exists(topic_path):
        raise FileNotFoundError(
            f"Run 02_build_rag.py first – {topic_path} not found"
        )

    with open(topic_path) as f:
        topic_cps = json.load(f)

    summaries = [cp["summary"] for cp in topic_cps if cp.get("summary")]
    print(f"[persona] Extracting persona from {len(summaries)} topic summaries ...")

    BATCH = config.PERSONA_BATCH_SIZE
    partials: list[dict] = []

    for i in tqdm(range(0, len(summaries), BATCH), desc="Extracting"):
        batch = summaries[i : i + BATCH]
        partial = extract_partial(batch)
        if partial:
            partials.append(partial)

    print(f"[persona] Merging {len(partials)} partial extractions ...")

    # Merge in rounds (tree reduction)
    while len(partials) > 1:
        next_round: list[dict] = []
        for i in range(0, len(partials), 4):
            chunk = partials[i : i + 4]
            if len(chunk) == 1:
                next_round.append(chunk[0])
            else:
                next_round.append(merge_partials(chunk))
        partials = next_round

    final_persona = partials[0] if partials else {}

    # Add metadata
    final_persona["_meta"] = {
        "source_summaries": len(summaries),
        "extraction_model": config.SUMMARY_MODEL,
    }

    with open(config.PERSONA_PATH, "w") as f:
        json.dump(final_persona, f, indent=2)

    print(f"\n[persona] ✅ Persona saved to {config.PERSONA_PATH}")
    print(json.dumps(final_persona, indent=2)[:1500])


if __name__ == "__main__":
    main()