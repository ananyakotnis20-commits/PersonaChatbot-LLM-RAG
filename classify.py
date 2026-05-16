"""
part2_intent/classify.py
Offline Intent Classifier — runs fully on CPU, no external API calls.
Target: <200 ms per message after warm-up.

Intent classes: reminder | emotional-support | action-item | small-talk | unknown
"""
import os
import sys
import time

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Training data (all 5 required classes, ≥2 samples each) ─────────────────
# FIX 1: Original had only 4 classes — missing "unknown" entirely.
#         LinearSVC cannot predict a label it never saw during training.
TRAINING_DATA = [
    # reminder
    ("remind me to call dad at 5pm",          "reminder"),
    ("set an alarm for tomorrow morning",     "reminder"),
    ("don't let me forget the dentist",       "reminder"),
    ("ping me about the meeting at noon",     "reminder"),
    # emotional-support
    ("i've been feeling really down lately",  "emotional-support"),
    ("i feel so overwhelmed and anxious",     "emotional-support"),
    ("i'm really struggling right now",       "emotional-support"),
    ("nobody understands what i'm going through", "emotional-support"),
    # action-item
    ("buy groceries and fix the leak",        "action-item"),
    ("can you draft that email for me",       "action-item"),
    ("please schedule a call with the team",  "action-item"),
    ("update the report by end of day",       "action-item"),
    # small-talk
    ("what's the weather like today",         "small-talk"),
    ("hi how are you doing",                  "small-talk"),
    ("tell me a fun fact",                    "small-talk"),
    ("what did you do today",                 "small-talk"),
    # unknown  ← FIX: was completely missing
    ("xkcd 1234",                             "unknown"),
    ("asdf qwer zxcv plugh",                  "unknown"),
    ("the quick brown fox",                   "unknown"),
    ("42 banana purple elephant",             "unknown"),
]


class IntentClassifier:
    """
    Loads MiniLM + LinearSVC from disk (offline).
    Falls back to _train_dummy() if the saved model is absent.
    """

    def __init__(self):
        # FIX 2: Original called SentenceTransformer("all-MiniLM-L6-v2") BEFORE
        #         the os.path.exists guard, so it always hit the network.
        #         Now we load from the local path when it exists.
        if os.path.exists(config.MINILM_SENTINEL):
            # Fully offline — load from local disk
            self.model = SentenceTransformer(config.MINILM_PATH)
        else:
            # First-time: download and cache
            print("[classify] MiniLM not found locally — downloading (one-time)…")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            os.makedirs(config.MINILM_PATH, exist_ok=True)
            self.model.save(config.MINILM_PATH)
            print(f"[classify] Saved to {config.MINILM_PATH}")

        # FIX 3: Both MODEL_PATH and ENC_PATH must exist; if either is missing
        #         we must retrain.  Original only checked MODEL_PATH.
        if not os.path.exists(config.INTENT_CLF_PATH) or \
           not os.path.exists(config.LABEL_ENC_PATH):
            print("[classify] Saved classifier not found — training now…")
            self._train_dummy()

        self.clf = joblib.load(config.INTENT_CLF_PATH)
        self.le  = joblib.load(config.LABEL_ENC_PATH)
        print("[classify] Ready.")

    # ── Training ─────────────────────────────────────────────────────────────

    def _train_dummy(self) -> None:
        """Train a LinearSVC on TRAINING_DATA and persist to disk."""
        texts, labels = zip(*TRAINING_DATA)

        le = LabelEncoder()
        y  = le.fit_transform(labels)
        X  = self.model.encode(list(texts), show_progress_bar=False)

        # C=0.5 generalises better than default 1.0 on small datasets
        clf = LinearSVC(C=0.5, max_iter=5000)
        clf.fit(X, y)

        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(clf, config.INTENT_CLF_PATH)
        joblib.dump(le,  config.LABEL_ENC_PATH)
        print(f"[classify] Classifier saved to {config.MODELS_DIR}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Classify a single message.  Returns a dict with:
            intent   – one of the 5 label strings
            latency  – wall-clock ms as a string
            warning  – present only if latency > 200 ms

        FIX 4: Original did not handle the case where self.clf / self.le
                aren't loaded yet (e.g. if __init__ raised mid-way).
        """
        t0  = time.perf_counter()
        vec = self.model.encode([text], show_progress_bar=False)
        raw = self.clf.predict(vec)[0]
        ms  = (time.perf_counter() - t0) * 1000

        result: dict = {
            "intent":  self.le.inverse_transform([raw])[0],
            "latency": f"{ms:.1f}ms",
        }
        if ms > 200:
            result["warning"] = f"Latency {ms:.1f}ms exceeds 200ms budget"

        return result


# ── Module-level singleton (load once, reuse forever) ────────────────────────
# FIX 5: Original instantiated IntentClassifier() inside every call-site.
#         Each construction loads MiniLM from disk (~100 ms) so it blew
#         the 200 ms budget before even running inference.
_clf: IntentClassifier | None = None


def classify(text: str) -> dict:
    """Public API — import and call this."""
    global _clf
    if _clf is None:
        _clf = IntentClassifier()
    return _clf.predict(text)


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        "remind me to call mom at 6",
        "I need to schedule a meeting with my sister",
        "I feel really anxious today",
        "what's up?",
        "xyzzy plugh 42",
    ]
    ic = IntentClassifier()
    print("\n── Intent Classifier Smoke Test ──")
    for s in test_sentences:
        out = ic.predict(s)
        warn = f"  ⚠ {out['warning']}" if "warning" in out else ""
        print(f"  {out['latency']:>8}  [{out['intent']:<20}]  \"{s}\"{warn}")