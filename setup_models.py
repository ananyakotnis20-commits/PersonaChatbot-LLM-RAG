"""
setup_models.py — One-time setup script.
Run this ONCE before anything else.

What it does:
  1. Downloads MiniLM-L6-v2 and saves it locally (offline after this).
  2. Trains the intent classifier on labelled data and saves it.
  3. Verifies both artefacts are sane before exiting.

FIX 1: Original downloaded the model and immediately created a SECOND
        SentenceTransformer from disk — redundant and risky if save() failed.
FIX 2: No idempotency guard — re-running redownloaded 80 MB every time.
FIX 3: TRAINING_DATA had only 1 sample → LinearSVC can't train; missing
        the "unknown" class; only 1 sample per class causes SVC to crash.
"""
import os
import sys

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# ── Path bootstrap ──────────────────────────────────────────────────────────
# setup_models.py lives at the project root, so BASE_DIR is already correct.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Training data (all 5 classes, ≥ 2 samples each) ─────────────────────────
TRAINING_DATA = [
    # reminder
    ("remind me to call dad at 5pm",               "reminder"),
    ("set an alarm for tomorrow morning",           "reminder"),
    ("don't let me forget about the dentist",       "reminder"),
    ("ping me 10 minutes before the meeting",       "reminder"),
    # emotional-support
    ("i've been feeling really down lately",        "emotional-support"),
    ("i'm so anxious and overwhelmed right now",    "emotional-support"),
    ("nobody seems to understand what i'm feeling", "emotional-support"),
    ("i just need someone to talk to",              "emotional-support"),
    # action-item
    ("buy groceries and fix the kitchen leak",      "action-item"),
    ("can you draft that report for me",            "action-item"),
    ("please schedule a call with the client",      "action-item"),
    ("finish the presentation by end of day",       "action-item"),
    # small-talk
    ("what's the weather like today",               "small-talk"),
    ("hi how are you doing",                        "small-talk"),
    ("tell me something interesting",               "small-talk"),
    ("what did you do this weekend",                "small-talk"),
    # unknown  ← FIX: was completely missing in original
    ("xkcd 1234",                                   "unknown"),
    ("asdf qwer zxcv plugh",                        "unknown"),
    ("42 banana purple elephant",                   "unknown"),
    ("lorem ipsum dolor sit amet",                  "unknown"),
]


def setup_minilm(force: bool = False) -> SentenceTransformer:
    """
    Download MiniLM-L6-v2 once and save to config.MINILM_PATH.

    FIX: uses config.MINILM_SENTINEL (a specific file inside the folder)
         as the guard — not the folder itself — so an empty / partial folder
         is not mistaken for a complete model.
    """
    if not force and os.path.exists(config.MINILM_SENTINEL):
        print(f"[setup] MiniLM already saved at {config.MINILM_PATH} — skipping download.")
        model = SentenceTransformer(config.MINILM_PATH)
        return model

    print("[setup] Downloading MiniLM-L6-v2 (one-time, ~80 MB)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    os.makedirs(config.MINILM_PATH, exist_ok=True)
    model.save(config.MINILM_PATH)
    print(f"[setup] ✅ MiniLM saved to {config.MINILM_PATH}")

    # FIX: reuse the already-loaded object — no second SentenceTransformer()
    return model


def setup_classifier(model: SentenceTransformer, force: bool = False) -> None:
    """
    Train LinearSVC intent classifier and save artefacts.

    FIX: original training data had only 1 sample per class, causing
         LinearSVC to raise during fit.  Now ≥ 4 samples per class.
    """
    if (
        not force
        and os.path.exists(config.INTENT_CLF_PATH)
        and os.path.exists(config.LABEL_ENC_PATH)
    ):
        print("[setup] Classifier already trained — skipping.")
        return

    print("[setup] Training intent classifier…")
    texts, labels = zip(*TRAINING_DATA)

    encoder = LabelEncoder()
    y       = encoder.fit_transform(labels)
    X       = model.encode(list(texts), show_progress_bar=False)

    clf = LinearSVC(C=0.5, max_iter=5000)
    clf.fit(X, y)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(clf,     config.INTENT_CLF_PATH)
    joblib.dump(encoder, config.LABEL_ENC_PATH)
    print(f"[setup] ✅ Classifier saved to {config.MODELS_DIR}")


def verify_artefacts(model: SentenceTransformer) -> None:
    """Sanity-check every saved artefact."""
    print("\n[setup] Verifying artefacts…")

    # MiniLM embedding dimension
    vec = model.encode(["test"], show_progress_bar=False)
    assert vec.shape == (1, 384), f"Unexpected embedding shape: {vec.shape}"
    print(f"  ✅ MiniLM OK — embedding dim = {vec.shape[1]}")

    # Classifier round-trip
    clf     = joblib.load(config.INTENT_CLF_PATH)
    encoder = joblib.load(config.LABEL_ENC_PATH)
    classes = list(encoder.classes_)
    assert set(classes) == {
        "reminder", "emotional-support", "action-item", "small-talk", "unknown"
    }, f"Missing classes: {classes}"

    sample  = model.encode(["remind me tomorrow"], show_progress_bar=False)
    pred    = encoder.inverse_transform(clf.predict(sample))
    print(f"  ✅ Classifier OK — classes={classes}")
    print(f"  ✅ Test prediction: 'remind me tomorrow' → {pred[0]}")


def main(force: bool = False) -> None:
    print("=" * 55)
    print("  KaStack — Model Setup")
    print("=" * 55)

    model = setup_minilm(force=force)
    setup_classifier(model, force=force)
    verify_artefacts(model)

    print("\n[setup] ✅ All done. You can now run the pipeline.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KaStack model setup")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-train even if artefacts already exist",
    )
    args = parser.parse_args()
    main(force=args.force)